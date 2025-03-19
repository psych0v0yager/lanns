"""
Optional vLLM-based enhancement for LANNS embeddings.
"""
import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class VLLMEnhancer:
    """
    Optional enhancer that uses vLLM to generate enhanced profiles
    for embedding data. Only loaded if vLLM is installed.
    """
    
    def __init__(self,
                model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                cache_dir: str = "./profiles_cache",
                output_dir: str = "./enhanced_profiles",
                prompt_template: Optional[str] = None,
                use_fp16: bool = False):
        """
        Initialize the vLLM enhancer.
        
        Args:
            model_name: Name of the vLLM model
            cache_dir: Directory to cache generated profiles
            output_dir: Directory to save enhanced profiles
            prompt_template: Template for generating profiles
            use_fp16: Use half precision (FP16)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        
        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default prompt template if not provided
        if prompt_template is None:
            self.prompt_template = (
                "Given the following user data, I would like you to psychoanalyze this person. "
                "Build a psychological profile of this user." 
                "Really try to get into their thought patterns."
                "Given the following data fields for an entity, create a comprehensive "
                "profile that captures the essential characteristics in natural language. "
                "Focus on creating a rich description that would be useful for semantic "
                "similarity matching.\n\n"
                "DATA:\n{data}\n\n"
                "PROFILE:"
            )
        else:
            self.prompt_template = prompt_template
        
        # Try to import vLLM
        try:
            from vllm import LLM, SamplingParams
            import torch
            
            # Initialize vLLM
            logger.info(f"Initializing vLLM with model {model_name}")
            
            # Configure for multi-GPU if available
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            dtype = "half" if use_fp16 else "auto"
            
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                trust_remote_code=True
            )
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            self.vllm_available = True
            logger.info("vLLM initialized successfully")
            
        except ImportError:
            logger.warning("vLLM not available. Install with: pip install vllm")
            self.vllm_available = False
    
    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format data dictionary into a string for the prompt"""
        formatted = []
        for key, value in data.items():
            if value:  # Skip empty values
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate a cache key from data"""
        # Sort keys for consistent hashing
        sorted_items = sorted(data.items())
        # Join key-value pairs
        data_str = "_".join(f"{k}_{v}" for k, v in sorted_items)
        # Create hash of the string
        import hashlib
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def generate_enhanced_profiles(self, 
                                data_list: List[Dict[str, Any]],
                                ids: Optional[List[Any]] = None,
                                batch_size: int = 16,
                                use_cache: bool = True) -> List[str]:
        """
        Generate enhanced profiles for a list of data records.
        
        Args:
            data_list: List of data dictionaries
            ids: Optional list of IDs for the data
            batch_size: Batch size for processing
            use_cache: Whether to use cached profiles
            
        Returns:
            List[str]: List of generated profiles
        """
        if not self.vllm_available:
            raise ImportError("vLLM is not available. Cannot generate enhanced profiles.")
        
        if not data_list:
            return []
        
        profiles = []
        
        # Process in batches
        with tqdm(total=len(data_list), desc="Generating profiles") as pbar:
            for i in range(0, len(data_list), batch_size):
                batch_end = min(i + batch_size, len(data_list))
                batch_data = data_list[i:batch_end]
                batch_ids = ids[i:batch_end] if ids else None
                
                # Check cache for each item in batch
                batch_prompts = []
                batch_indices = []
                batch_cache_keys = []
                
                for j, data in enumerate(batch_data):
                    cache_key = self._get_cache_key(data)
                    cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                    
                    if use_cache and os.path.exists(cache_file):
                        # Load from cache
                        with open(cache_file, 'r') as f:
                            profile = json.load(f)['profile']
                        profiles.append(profile)
                    else:
                        # Generate prompt
                        formatted_data = self._format_data(data)
                        prompt = self.prompt_template.format(data=formatted_data)
                        batch_prompts.append(prompt)
                        batch_indices.append(i + j)
                        batch_cache_keys.append(cache_key)
                
                # Process non-cached items
                if batch_prompts:
                    # Generate with vLLM
                    outputs = self.llm.generate(batch_prompts, self.sampling_params)
                    
                    # Process outputs
                    for j, output in enumerate(outputs):
                        profile = output.outputs[0].text.strip()
                        idx = batch_indices[j]
                        
                        # Extend profiles list if needed
                        while len(profiles) <= idx:
                            profiles.append(None)
                        
                        profiles[idx] = profile
                        
                        # Cache the result
                        if use_cache:
                            cache_key = batch_cache_keys[j]
                            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                            with open(cache_file, 'w') as f:
                                json.dump({'profile': profile}, f)
                
                # Save batch to output dir
                batch_file = os.path.join(self.output_dir, f"profiles_{i}_{batch_end}.json")
                with open(batch_file, 'w') as f:
                    json.dump({
                        'profiles': profiles[i:batch_end],
                        'ids': batch_ids
                    }, f)
                
                # Update progress
                pbar.update(batch_end - i)
                
        return profiles

def combine_embeddings(raw_embeddings: np.ndarray, 
                     enhanced_embeddings: np.ndarray,
                     method: str = 'concatenate',
                     raw_weight: float = 0.5) -> np.ndarray:
    """
    Combine raw and enhanced embeddings.
    
    Args:
        raw_embeddings: Raw embeddings
        enhanced_embeddings: Enhanced embeddings
        method: Combination method ('concatenate' or 'weighted_average')
        raw_weight: Weight for raw embeddings when using weighted_average
        
    Returns:
        np.ndarray: Combined embeddings
    """
    if method == 'concatenate':
        # Concatenate along feature dimension
        combined = np.concatenate([raw_embeddings, enhanced_embeddings], axis=1)
        # Normalize
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        return combined / norms
    else:  # weighted_average
        # Ensure same dimensions or project to smaller
        if raw_embeddings.shape[1] != enhanced_embeddings.shape[1]:
            min_dim = min(raw_embeddings.shape[1], enhanced_embeddings.shape[1])
            raw_embeddings = raw_embeddings[:, :min_dim]
            enhanced_embeddings = enhanced_embeddings[:, :min_dim]
        
        # Normalize first
        raw_norm = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        enhanced_norm = enhanced_embeddings / np.linalg.norm(enhanced_embeddings, axis=1, keepdims=True)
        
        # Weighted average
        enhanced_weight = 1.0 - raw_weight
        combined = (raw_weight * raw_norm) + (enhanced_weight * enhanced_norm)
        
        # Normalize again
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        return combined / norms