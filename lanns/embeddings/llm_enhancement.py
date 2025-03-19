"""
Smart embedding enhancement using vLLM for LANNS.
"""
import os
import time
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

# Import vLLM
try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("vLLM is required. Install with: pip install vllm")

import logging
logger = logging.getLogger(__name__)

class VLLMEnhancer:
    """
    Enhances data with LLM-generated profiles using vLLM.
    """
    
    def __init__(self,
                model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                device: str = "cuda",
                max_tokens: int = 512,
                cache_dir: str = "./llm_profiles",
                profile_template: Optional[str] = None,
                use_fp16: bool = False):
        """
        Initialize the vLLM enhancer.
        
        Args:
            model_name: Model name or path
            device: Device to use (cuda, cpu)
            max_tokens: Maximum tokens for generation
            cache_dir: Directory to cache generated profiles
            profile_template: Template for profile generation
            use_fp16: Whether to use half precision
        """
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup default profile template if not provided
        if profile_template is None:
            self.profile_template = (
                "You are an AI assistant that creates detailed profiles from data. "
                "Given the following data fields for an entity, create a comprehensive "
                "profile that captures the essential characteristics in natural language. "
                "Focus on creating a rich description that would be useful for semantic "
                "similarity matching.\n\n"
                "DATA:\n{data}\n\n"
                "PROFILE:"
            )
        else:
            self.profile_template = profile_template
        
        # Initialize vLLM
        logger.info(f"Initializing vLLM enhancer with model {model_name}")
        import torch
        tensor_parallel_size = torch.cuda.device_count() if device == "cuda" else 1
        dtype = "half" if use_fp16 else "auto"
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
        )
        
        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.3,  # Low temperature for stable outputs
            max_tokens=max_tokens,
            top_p=0.9
        )
    
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
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if profile exists in cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)['profile']
        return None
    
    def _save_to_cache(self, cache_key: str, profile: str) -> None:
        """Save profile to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump({'profile': profile}, f)
    
    def generate_profile(self, data: Dict[str, Any], use_cache: bool = True) -> str:
        """
        Generate a profile for a single data point.
        
        Args:
            data: Dictionary of data fields
            use_cache: Whether to use cached profiles
            
        Returns:
            str: Generated profile
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(data)
            cached = self._check_cache(cache_key)
            if cached:
                return cached
        
        # Format data for prompt
        formatted_data = self._format_data(data)
        prompt = self.profile_template.format(data=formatted_data)
        
        # Generate with vLLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        profile = outputs[0].outputs[0].text.strip()
        
        # Cache the result
        if use_cache:
            cache_key = self._get_cache_key(data)
            self._save_to_cache(cache_key, profile)
        
        return profile
    
    def generate_batch_profiles(self, 
                               data_list: List[Dict[str, Any]], 
                               batch_size: int = 16,
                               use_cache: bool = True,
                               show_progress: bool = True) -> List[str]:
        """
        Generate profiles for a batch of data points.
        
        Args:
            data_list: List of data dictionaries
            batch_size: Batch size for processing
            use_cache: Whether to use cached profiles
            show_progress: Whether to show progress bar
            
        Returns:
            List[str]: List of generated profiles
        """
        profiles = []
        
        # Check cache for all items first
        if use_cache:
            cached_profiles = {}
            for i, data in enumerate(data_list):
                cache_key = self._get_cache_key(data)
                cached = self._check_cache(cache_key)
                if cached:
                    cached_profiles[i] = cached
            
            if cached_profiles:
                logger.info(f"Found {len(cached_profiles)} profiles in cache")
        else:
            cached_profiles = {}
        
        # Prepare prompts for items not in cache
        prompts = []
        prompt_indices = []
        
        for i, data in enumerate(data_list):
            if i not in cached_profiles:
                formatted_data = self._format_data(data)
                prompt = self.profile_template.format(data=formatted_data)
                prompts.append(prompt)
                prompt_indices.append(i)
        
        # Generate for non-cached items
        if prompts:
            logger.info(f"Generating {len(prompts)} profiles with vLLM")
            
            # Process in batches
            all_generated = [None] * len(prompts)
            
            # Use tqdm if requested
            batch_iterator = range(0, len(prompts), batch_size)
            if show_progress:
                batch_iterator = tqdm(batch_iterator, desc="Generating profiles")
            
            for i in batch_iterator:
                end_idx = min(i + batch_size, len(prompts))
                batch_prompts = prompts[i:end_idx]
                batch_indices = prompt_indices[i:end_idx]
                
                # Generate with vLLM
                outputs = self.llm.generate(batch_prompts, self.sampling_params)
                
                # Process outputs
                for j, output in enumerate(outputs):
                    profile = output.outputs[0].text.strip()
                    idx = i + j
                    all_generated[idx] = profile
                    
                    # Cache the result
                    if use_cache:
                        data_idx = batch_indices[j]
                        cache_key = self._get_cache_key(data_list[data_idx])
                        self._save_to_cache(cache_key, profile)
        
        # Combine cached and generated results
        profiles = [None] * len(data_list)
        
        # Add cached profiles
        for i, profile in cached_profiles.items():
            profiles[i] = profile
        
        # Add generated profiles
        for i, idx in enumerate(prompt_indices):
            profiles[idx] = all_generated[i]
        
        return profiles