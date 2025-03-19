"""
Embedding generator for LANNS.
"""
import os
import time
import numpy as np
import torch
import datetime
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers models.
    """
    
    def __init__(self, 
                model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
                batch_size: int = 1024,
                output_dir: str = "./embeddings",
                use_fp16: bool = False,
                use_mrl: bool = False,
                mrl_dimensions: int = 256,
                device: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model
            batch_size: Batch size for processing
            output_dir: Directory to save embeddings
            use_fp16: Use half precision (FP16)
            use_mrl: Apply dimensionality reduction
            mrl_dimensions: Number of dimensions to keep if using MRL
            device: Device to use (cuda, cpu, or None for auto)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.use_mrl = use_mrl
        self.mrl_dimensions = mrl_dimensions
        
        # Auto-select device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading model: {model_name} on {self.device}")
            
            start_load = time.time()
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Model loaded in {time.time() - start_load:.2f}s")
            
            # Enable half precision if requested
            if use_fp16:
                self.model.half()
                logger.info("Using half precision (FP16)")
                
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    def generate(self, 
                texts: List[str],
                ids: Optional[List[Any]] = None,
                batch_idx: int = 0,
                save: bool = True) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of text strings to embed
            ids: Optional list of IDs corresponding to texts
            batch_idx: Batch index for saving
            save: Whether to save embeddings to disk
            
        Returns:
            numpy.ndarray: Generated embeddings
        """
        start_time = time.time()
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Display GPU info if available
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory allocated: {mem_allocated:.2f} GB")
        
        # Generate embeddings
        with torch.inference_mode():
            batch_embeddings = self.model.encode(
                texts, 
                batch_size=min(self.batch_size, 1024),
                show_progress_bar=True
            )
        
        # Apply MRL compression if requested
        if self.use_mrl and batch_embeddings.shape[1] > self.mrl_dimensions:
            batch_embeddings = batch_embeddings[:, :self.mrl_dimensions]
            logger.info(f"Applied MRL, reduced dimensions to {self.mrl_dimensions}")
        
        # Save batch embeddings if requested
        if save:
            batch_end = batch_idx + len(texts)
            embeddings_file = f"{self.output_dir}/embeddings_{batch_idx}_{batch_end}.npy"
            np.save(embeddings_file, batch_embeddings)
            
            # Save IDs if provided
            if ids is not None:
                ids_file = f"{self.output_dir}/ids_{batch_idx}_{batch_end}.json"
                with open(ids_file, 'w') as f:
                    json.dump(ids, f)
                    
            logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Log stats
        duration = time.time() - start_time
        examples_per_second = len(texts) / duration if duration > 0 else 0
        logger.info(f"Generated {len(texts)} embeddings in {duration:.2f}s ({examples_per_second:.2f} texts/sec)")
        
        # Log GPU memory again if available
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory after embedding: {mem_allocated:.2f} GB")
        
        return batch_embeddings
    
    def process_with_checkpointing(self, 
                                 processor,
                                 checkpoint_file: str = None,
                                 checkpoint_frequency: int = 100):
        """
        Process data from a processor with checkpointing support.
        
        Args:
            processor: Data processor that yields batches
            checkpoint_file: Path to checkpoint file
            checkpoint_frequency: Save checkpoint after this many batches
            
        Returns:
            int: Total number of processed examples
        """
        start_time = time.time()
        total_processed = 0
        batch_num = 0
        
        # Resume from checkpoint if available
        start_batch_idx = 0
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_batch_idx = checkpoint.get('next_batch_start_idx', 0)
                total_processed = checkpoint.get('total_processed', 0)
                logger.info(f"Resuming from checkpoint at batch index {start_batch_idx}")
        
        # Process batches
        for batch_data, batch_texts, batch_ids, batch_idx, is_last_batch in processor.process():
            # Skip batches before resume point
            if batch_idx < start_batch_idx:
                continue
                
            batch_size = len(batch_texts)
            
            # Generate embeddings for this batch
            self.generate(batch_texts, batch_ids, batch_idx=batch_idx)
            
            # Update counters
            total_processed += batch_size
            batch_num += 1
            
            # Save checkpoint if requested
            if checkpoint_file and (batch_num % checkpoint_frequency == 0 or is_last_batch):
                next_batch_idx = batch_idx + batch_size
                checkpoint = {
                    "next_batch_start_idx": next_batch_idx,
                    "total_processed": total_processed,
                    "timestamp": str(datetime.datetime.now())
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                logger.info(f"Saved checkpoint after {total_processed:,} examples")
        
        # Final stats
        total_time = time.time() - start_time
        logger.info(f"Processing complete!")
        logger.info(f"Total examples processed: {total_processed:,}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average speed: {total_processed/total_time:.2f} examples/second")
        logger.info(f"Embeddings saved to {self.output_dir}")
        
        return total_processed