"""
LANNS: Library for Large Scale Approximate Nearest Neighbor Search
"""

# Import main components for easy access
from lanns.indexing.builder import LANNSIndexBuilder
from lanns.indexing.storage import LANNSIndex
from lanns.core.segmenters import RandomSegmenter, RandomHyperplaneSegmenter, APDSegmenter 
from lanns.core.sharding import Sharder

# Package metadata
__version__ = '0.1.0'
__author__ = 'psych0v0yager'
