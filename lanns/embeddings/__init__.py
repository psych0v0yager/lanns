"""
LANNS embeddings package for generating and enhancing embeddings.
"""
from lanns.embeddings.processor import DataProcessor, NDJSONProcessor, get_processor
from lanns.embeddings.generator import EmbeddingGenerator
from lanns.embeddings.utils import setup_logging, get_gpu_memory_info

# Only import enhancer if vLLM is available
try:
    from lanns.embeddings.enhancer import VLLMEnhancer, combine_embeddings
    vllm_available = True
except ImportError:
    vllm_available = False

__all__ = [
    'DataProcessor',
    'NDJSONProcessor',
    'get_processor',
    'EmbeddingGenerator',
    'setup_logging',
    'get_gpu_memory_info',
    'vllm_available'
]

if vllm_available:
    __all__.extend(['VLLMEnhancer', 'combine_embeddings'])