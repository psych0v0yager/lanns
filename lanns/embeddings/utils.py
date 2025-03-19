"""
Utility functions for LANNS embeddings.
"""
import os
import logging
import time
import json
import datetime
from typing import Dict, Any, Optional

def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Setup logging for LANNS embedding generation"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def get_gpu_memory_info():
    """Get GPU memory information"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            
            return {
                'gpu_name': gpu_name,
                'total_memory_gb': total_memory,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'available_gb': total_memory - reserved
            }
        else:
            return {'error': 'CUDA not available'}
    except ImportError:
        return {'error': 'PyTorch not installed'}
    except Exception as e:
        return {'error': str(e)}

def save_checkpoint(checkpoint_file: str, batch_idx: int, total_processed: int):
    """Save a checkpoint to file"""
    checkpoint = {
        "next_batch_start_idx": batch_idx,
        "total_processed": total_processed,
        "timestamp": str(datetime.datetime.now())
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)
    
    return checkpoint

def load_checkpoint(checkpoint_file: str):
    """Load a checkpoint from file"""
    if not os.path.exists(checkpoint_file):
        return None
        
    with open(checkpoint_file, 'r') as f:
        return json.load(f)