"""
sharding.py - Implementation of sharding logic for LANNS
"""
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sharder:
    """
    Handles sharding of data points across multiple shards.
    In the LANNS system, sharding is the first level of partitioning.
    """
    
    def __init__(self, num_shards: int):
        """
        Initialize the sharder.
        
        Args:
            num_shards: Number of shards to create
        """
        self.num_shards = num_shards
        logger.info(f"Initialized sharder with {num_shards} shards")
    
    def assign_shards(self, ids: List[Any]) -> np.ndarray:
        """
        Assign data points to shards based on their IDs using a hash function.
        
        Args:
            ids: List of IDs to assign to shards
            
        Returns:
            shard_ids: Shard ID for each data point
        """
        if not ids:
            raise ValueError("Cannot assign empty list of IDs to shards")
        
        # Use hash function to determine shard
        shard_ids = np.array([hash(str(id_)) % self.num_shards for id_ in ids], dtype=int)
        
        # Log distribution statistics
        if logger.isEnabledFor(logging.DEBUG):
            unique, counts = np.unique(shard_ids, return_counts=True)
            distribution = dict(zip(unique, counts))
            logger.debug(f"Shard distribution: {distribution}")
        
        return shard_ids
    
    def route_queries(self, queries: np.ndarray) -> List[List[int]]:
        """
        Route queries to shards. In the basic sharder, all queries go to all shards.
        
        Args:
            queries: Query embeddings
            
        Returns:
            List of lists, where each inner list contains the shard IDs
            that the query should be routed to (all shards)
        """
        # Route all queries to all shards
        all_shards = list(range(self.num_shards))
        return [all_shards for _ in range(len(queries))]
    
    def calculate_per_shard_topk(self, k: int, confidence: float = 0.95) -> int:
        """
        Calculate the number of top-K results to fetch from each shard.
        
        Uses Normal Approximation Interval from the LANNS paper to
        reduce the number of nearest neighbors fetched from each randomly
        partitioned shard.
        
        Args:
            k: Number of nearest neighbors required in final result
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Number of results to fetch from each shard
        """
        if self.num_shards == 1:
            return k
        
        # Implementation of Normal Approximation Interval
        from scipy.stats import norm
        
        s_prime = 1.0 / self.num_shards
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        ci = s_prime + z_score * np.sqrt(s_prime * (1 - s_prime) / k)
        per_shard_topk = min(k, int(np.ceil(ci * k)))
        
        logger.info(f"Using per_shard_topk={per_shard_topk} for k={k} with {self.num_shards} shards")
        return per_shard_topk