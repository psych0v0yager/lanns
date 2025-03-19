"""
Embeddings generator module for the LANNS system.
"""
import os
import numpy as np
import torch
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

from lanns.embeddings.models import get_embedding_model
from lanns.pipeline.logging import get_logger

logger = get_logger(__name__)

class EmbeddingsGenerator:
    """
    Core class for generating embeddings from processed data.
    """
    
    def __init__(self, 
                 model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
                 batch_size: int = 1024,
                 use_fp16: bool = False,
                 use_mrl: bool = False,
                 mrl_dimensions: int = 256,
                 output_dir: str = "./embeddings",
                 device: Optional[str] = None):
        """
        Initialize the embeddings generator.
        
        Args:
            model_name: Name or path of the embedding model
            batch_size: Batch size for embedding generation
            use_fp16: Use FP16 precision to reduce memory usage
            use_mrl: Use dimensionality reduction
            mrl_dimensions: Number of dimensions to keep if using MRL
            output_dir: Directory to save embeddings
            device: Device to use (cuda, cpu, or None for auto-selection)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.use_mrl = use_mrl
        self.mrl_dimensions = mrl_dimensions
        self.output_dir = output_dir
        
        # Auto-select device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize model
        logger.info(f"Initializing embedding model: {model_name} on {self.device}")
        self.model = get_embedding_model(model_name, self.device)
        
        # Use FP16 if requested
        if use_fp16:
            self.model.half()
            logger.info("Using FP16 precision")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self, 
                texts: List[str],
                ids: Optional[List[Any]] = None,
                save: bool = True,
                batch_idx: int = 0) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            ids: Optional list of IDs corresponding to texts
            save: Whether to save embeddings to disk
            batch_idx: Batch index for saving
            
        Returns:
            numpy.ndarray: Generated embeddings
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([])
            
        start_time = time.time()
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Generate embeddings
            with torch.inference_mode():
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    batch_size=min(self.batch_size, 32),
                    show_progress_bar=False
                )
            
            # Apply dimensionality reduction if needed
            if self.use_mrl:
                batch_embeddings = batch_embeddings[:, :self.mrl_dimensions]
                
            all_embeddings.append(batch_embeddings)
        
        # Combine batches
        embeddings = np.vstack(all_embeddings)
        
        # Save to disk if requested
        if save:
            # Create batch-specific filename
            batch_end = batch_idx + len(texts)
            embeddings_file = os.path.join(self.output_dir, f"embeddings_{batch_idx}_{batch_end}.npy")
            np.save(embeddings_file, embeddings)
            
            # Save IDs if provided
            if ids:
                ids_file = os.path.join(self.output_dir, f"ids_{batch_idx}_{batch_end}.json")
                import json
                with open(ids_file, 'w') as f:
                    json.dump(ids, f)
        
        duration = time.time() - start_time
        logger.info(f"Generated {len(texts)} embeddings in {duration:.2f}s ({len(texts)/duration:.2f} texts/sec)")
        
        return embeddings
    
    def generate_from_processor(self, processor, save=True):
        """
        Generate embeddings using a data processor that yields batches.
        
        Args:
            processor: Data processor instance that yields (texts, ids, batch_idx)
            save: Whether to save embeddings to disk
            
        Returns:
            int: Total number of processed texts
        """
        total_processed = 0
        
        for texts, ids, batch_idx, is_last_batch in processor:
            self.generate(texts, ids, save=save, batch_idx=batch_idx)
            total_processed += len(texts)
            
        return total_processed