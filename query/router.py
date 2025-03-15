"""
router.py - Query routing logic for LANNS
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging

from lanns.core.segmenters import SegmenterBase
from lanns.core.sharding import Sharder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Handles routing of queries to appropriate shards and segments.
    """
    
    def __init__(self, sharder: Sharder, segmenters: Dict[int, SegmenterBase]):
        """
        Initialize the query router.
        
        Args:
            sharder: Sharder instance for routing to shards
            segmenters: Dictionary mapping shard IDs to segmenters
        """
        self.sharder = sharder
        self.segmenters = segmenters
        
        logger.debug(f"Initialized QueryRouter with {len(segmenters)} shard segmenters")
    
    def route_query(self, query_embedding: np.ndarray) -> Dict[int, List[int]]:
        """
        Route a query to the appropriate shards and segments.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            Dictionary mapping shard IDs to lists of segment IDs
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Route to shards
        shard_ids = self.sharder.route_queries(query_embedding)[0]
        
        # Route to segments within each shard
        routing_map = {}
        
        for shard_id in shard_ids:
            if shard_id in self.segmenters:
                # Get segmenter for this shard
                segmenter = self.segmenters[shard_id]
                
                # Route to segments
                segment_ids = segmenter.route_queries(query_embedding)[0]
                
                # Add to routing map
                routing_map[shard_id] = segment_ids
            else:
                logger.warning(f"No segmenter found for shard {shard_id}")
        
        return routing_map
    
    def batch_route_queries(self, query_embeddings: np.ndarray) -> List[Dict[int, List[int]]]:
        """
        Route multiple queries to the appropriate shards and segments.
        
        Args:
            query_embeddings: Query embeddings
            
        Returns:
            List of dictionaries mapping shard IDs to lists of segment IDs for each query
        """
        return [self.route_query(query) for query in query_embeddings]

    def calculate_per_shard_topk(self, k: int, confidence: float = 0.95) -> int:
        """
        Calculate the number of top-K results to fetch from each shard.
        
        Args:
            k: Number of nearest neighbors required in final result
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Number of results to fetch from each shard
        """
        return self.sharder.calculate_per_shard_topk(k, confidence)