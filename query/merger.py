"""
merger.py - Result merging logic for LANNS
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
import heapq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultMerger:
    """
    Handles merging of results from multiple shards and segments.
    """
    
    def __init__(self):
        """
        Initialize the result merger.
        """
        pass
    
    def merge_segment_results(self, segment_results: Dict[int, Tuple[List[Any], List[float]]], 
                            per_shard_topk: int) -> Tuple[List[Any], List[float]]:
        """
        Merge results from different segments within a shard.
        
        Args:
            segment_results: Dictionary mapping segment IDs to (ids, distances) tuples
            per_shard_topk: Number of top results to return per shard
            
        Returns:
            Tuple of (merged_ids, merged_distances)
        """
        # Collect all results
        all_ids = []
        all_distances = []
        
        for segment_id, (ids, distances) in segment_results.items():
            all_ids.extend(ids)
            all_distances.extend(distances)
        
        # Sort by distance
        if not all_ids:
            return [], []
        
        sorted_indices = np.argsort(all_distances)
        merged_ids = [all_ids[i] for i in sorted_indices[:per_shard_topk]]
        merged_distances = [all_distances[i] for i in sorted_indices[:per_shard_topk]]
        
        return merged_ids, merged_distances
    
    def merge_shard_results(self, shard_results: Dict[int, Tuple[List[Any], List[float]]], 
                           k: int) -> Tuple[List[Any], List[float]]:
        """
        Merge results from different shards.
        
        Args:
            shard_results: Dictionary mapping shard IDs to (ids, distances) tuples
            k: Number of top results to return
            
        Returns:
            Tuple of (merged_ids, merged_distances)
        """
        # Collect all results
        all_ids = []
        all_distances = []
        
        for shard_id, (ids, distances) in shard_results.items():
            all_ids.extend(ids)
            all_distances.extend(distances)
        
        # Sort by distance
        if not all_ids:
            return [], []
        
        sorted_indices = np.argsort(all_distances)
        merged_ids = [all_ids[i] for i in sorted_indices[:k]]
        merged_distances = [all_distances[i] for i in sorted_indices[:k]]
        
        return merged_ids, merged_distances
    
    def merge_results_efficient(self, 
                              segment_results: Dict[int, Dict[int, Tuple[List[Any], List[float]]]], 
                              k: int,
                              per_shard_topk: Optional[int] = None) -> Tuple[List[Any], List[float]]:
        """
        Efficiently merge results from multiple shards and segments using a two-level merging strategy.
        
        Args:
            segment_results: Dictionary mapping shard IDs to dictionaries mapping segment IDs to results
            k: Number of top results to return
            per_shard_topk: Number of top results to return per shard (optional)
            
        Returns:
            Tuple of (merged_ids, merged_distances)
        """
        # If per_shard_topk is not specified, use k
        if per_shard_topk is None:
            per_shard_topk = k
        
        # First level: Merge segments within each shard
        shard_results = {}
        
        for shard_id, shard_segment_results in segment_results.items():
            merged_ids, merged_distances = self.merge_segment_results(
                shard_segment_results, per_shard_topk
            )
            
            shard_results[shard_id] = (merged_ids, merged_distances)
        
        # Second level: Merge shards
        final_ids, final_distances = self.merge_shard_results(shard_results, k)
        
        return final_ids, final_distances
    
    def merge_batch_results(self, 
                           batch_segment_results: List[Dict[int, Dict[int, Tuple[List[Any], List[float]]]]], 
                           k: int,
                           per_shard_topk: Optional[int] = None) -> Tuple[List[List[Any]], List[List[float]]]:
        """
        Merge batch query results from multiple shards and segments.
        
        Args:
            batch_segment_results: List of segment_results dictionaries for each query
            k: Number of top results to return
            per_shard_topk: Number of top results to return per shard (optional)
            
        Returns:
            Tuple of (merged_ids_batch, merged_distances_batch)
        """
        merged_ids_batch = []
        merged_distances_batch = []
        
        for segment_results in batch_segment_results:
            ids, distances = self.merge_results_efficient(
                segment_results, k, per_shard_topk
            )
            
            merged_ids_batch.append(ids)
            merged_distances_batch.append(distances)
        
        return merged_ids_batch, merged_distances_batch


class ResultMergerHeap:
    """
    Alternative result merger implementation using heaps for more efficient merging.
    Useful for very large result sets.
    """
    
    def merge_results(self, all_results: List[Tuple[Any, float]], k: int) -> Tuple[List[Any], List[float]]:
        """
        Merge results using a min-heap for efficiency.
        
        Args:
            all_results: List of (id, distance) tuples
            k: Number of top results to return
            
        Returns:
            Tuple of (merged_ids, merged_distances)
        """
        # Use a min-heap to efficiently get the k smallest distances
        # Python's heapq is a min-heap, so we negate the distances to get a max-heap
        heap = []
        
        for id_, distance in all_results:
            # Push to heap
            heapq.heappush(heap, (distance, id_))
            
            # Keep only the k smallest elements
            if len(heap) > k:
                heapq.heappop(heap)
        
        # Sort results by distance (ascending)
        heap.sort()
        
        # Extract IDs and distances
        merged_ids = [id_ for distance, id_ in heap]
        merged_distances = [distance for distance, id_ in heap]
        
        return merged_ids, merged_distances
    
    def merge_segment_results(self, segment_results: Dict[int, Tuple[List[Any], List[float]]], 
                            k: int) -> Tuple[List[Any], List[float]]:
        """
        Merge results from different segments using a heap.
        
        Args:
            segment_results: Dictionary mapping segment IDs to (ids, distances) tuples
            k: Number of top results to return
            
        Returns:
            Tuple of (merged_ids, merged_distances)
        """
        # Collect all results as (id, distance) tuples
        all_results = []
        
        for segment_id, (ids, distances) in segment_results.items():
            all_results.extend(zip(ids, distances))
        
        return self.merge_results(all_results, k)
    
    def merge_batch_results(self, 
                           batch_segment_results: List[Dict[int, Dict[int, Tuple[List[Any], List[float]]]]], 
                           k: int) -> Tuple[List[List[Any]], List[List[float]]]:
        """
        Merge batch query results using a heap.
        
        Args:
            batch_segment_results: List of segment_results dictionaries for each query
            k: Number of top results to return
            
        Returns:
            Tuple of (merged_ids_batch, merged_distances_batch)
        """
        merged_ids_batch = []
        merged_distances_batch = []
        
        for query_result in batch_segment_results:
            # Flatten segment results for this query
            all_results = []
            
            for shard_id, shard_results in query_result.items():
                for segment_id, (ids, distances) in shard_results.items():
                    all_results.extend(zip(ids, distances))
            
            # Merge with heap
            ids, distances = self.merge_results(all_results, k)
            
            merged_ids_batch.append(ids)
            merged_distances_batch.append(distances)
        
        return merged_ids_batch, merged_distances_batch