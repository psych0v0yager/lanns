"""
LANNS: Library for Large Scale Approximate Nearest Neighbor Search
"""

# Import main components for easy access
from lanns.indexing.builder import LANNSIndexBuilder
from lanns.indexing.storage import LANNSIndex
from lanns.core.segmenters import RandomSegmenter, RandomHyperplaneSegmenter, APDSegmenter 
from lanns.core.sharding import Sharder

# Import embeddings components
from lanns.embeddings.generator import EmbeddingGenerator
from lanns.embeddings.processor import NDJSONProcessor, get_processor
from lanns.embeddings.utils import setup_logging, get_gpu_memory_info

# Try to import vLLM enhancer components if available
try:
    from lanns.embeddings.enhancer import VLLMEnhancer, combine_embeddings
    vllm_available = True
except ImportError:
    vllm_available = False

# Package metadata
__version__ = '0.1.0'
__author__ = 'psych0v0yager'

# Define all exports
__all__ = [
    # Core components
    'LANNSIndexBuilder',
    'LANNSIndex',
    'RandomSegmenter', 
    'RandomHyperplaneSegmenter', 
    'APDSegmenter',
    'Sharder',
    
    # Embeddings components
    'EmbeddingGenerator',
    'NDJSONProcessor',
    'get_processor',
    'setup_logging',
    'get_gpu_memory_info',
    'vllm_available'
]

# Add vLLM components if available
if vllm_available:
    __all__.extend(['VLLMEnhancer', 'combine_embeddings'])