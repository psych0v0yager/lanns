"""
storage.py - Utilities for loading and storing LANNS indices
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

from lanns.core.segmenters import SegmenterBase, RandomSegmenter, RandomHyperplaneSegmenter, APDSegmenter
from lanns.core.sharding import Sharder
from lanns.core.hnsw_utils import HNSWIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LANNSIndex:
    """
    LANNS Index class for loading and querying a built index.
    """
    
    def __init__(self, index_dir: str, ef_search: int = 100):
        """
        Load a LANNS index from disk.
        
        Args:
            index_dir: Directory containing the index
            ef_search: HNSW search parameter
        """
        self.index_dir = index_dir
        self.ef_search = ef_search
        
        # Load metadata
        metadata_path = os.path.join(index_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Index directory {index_dir} does not contain metadata.json")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize components
        self.num_shards = self.metadata['num_shards']
        self.num_segments = self.metadata['num_segments']
        self.segmenter_type = self.metadata['segmenter_type']
        self.spill = self.metadata['spill']
        self.space = self.metadata.get('space', 'l2')
        
        # Initialize sharder
        self.sharder = Sharder(self.num_shards)
        
        # Load segmenters (one per shard)
        self.segmenters = {}
        
        # Initialize HNSW indices (lazy loading)
        self.indices = {}
        
        logger.info(f"Loaded LANNS index from {index_dir}")
        logger.info(f"Index contains {self.metadata['total_indexed_points']} points")
        logger.info(f"Index has {self.num_shards} shards and {self.num_segments} segments per shard")
    
    def _get_segmenter(self, shard_id: int) -> SegmenterBase:
        """
        Get the segmenter for a shard (lazy loading).
        
        Args:
            shard_id: ID of the shard
            
        Returns:
            Segmenter for the shard
        """
        if shard_id not in self.segmenters:
            # Check if we have a common segmenter
            common_segmenter_path = os.path.join(self.index_dir, 'common_segmenter.pkl')
            if os.path.exists(common_segmenter_path):
                # Load common segmenter
                with open(common_segmenter_path, 'rb') as f:
                    self.segmenters[shard_id] = pickle.load(f)
            else:
                # Load shard-specific segmenter
                shard_dir = os.path.join(self.index_dir, f'shard_{shard_id}')
                segmenter_path = os.path.join(shard_dir, 'segmenter.pkl')
                
                if not os.path.exists(segmenter_path):
                    # If no segmenter found, create a random one
                    logger.warning(f"No segmenter found for shard {shard_id}. Creating a random segmenter.")
                    if self.segmenter_type == 'rs':
                        self.segmenters[shard_id] = RandomSegmenter(self.num_segments, self.spill)
                    elif self.segmenter_type == 'rh':
                        self.segmenters[shard_id] = RandomHyperplaneSegmenter(self.num_segments, self.spill)
                    elif self.segmenter_type == 'apd':
                        self.segmenters[shard_id] = APDSegmenter(self.num_segments, self.spill)
                    else:
                        raise ValueError(f"Unknown segmenter type: {self.segmenter_type}")
                else:
                    # Load segmenter
                    with open(segmenter_path, 'rb') as f:
                        self.segmenters[shard_id] = pickle.load(f)
        
        return self.segmenters[shard_id]
    
    def _get_index(self, shard_id: int, segment_id: int) -> Optional[HNSWIndex]:
        """
        Get the HNSW index for a (shard, segment) pair (lazy loading).
        
        Args:
            shard_id: ID of the shard
            segment_id: ID of the segment
            
        Returns:
            HNSW index for the (shard, segment) pair, or None if not found
        """
        key = (shard_id, segment_id)
        
        if key not in self.indices:
            # Load index
            segment_path = os.path.join(self.index_dir, f'shard_{shard_id}', f'segment_{segment_id}')
            index_path = os.path.join(segment_path, 'hnsw_index')
            
            if not os.path.exists(f"{index_path}.meta"):
                logger.debug(f"No index found for shard {shard_id}, segment {segment_id}")
                return None
            
            try:
                # Load index
                index = HNSWIndex.load(index_path)
                self.indices[key] = index
            except Exception as e:
                logger.error(f"Error loading index for shard {shard_id}, segment {segment_id}: {str(e)}")
                return None
        
        return self.indices[key]
    
    def query(self, 
              query_embedding: np.ndarray, 
              k: int = 10,
              ef_search: Optional[int] = None) -> Tuple[List[Any], List[float]]:
        """
        Query the LANNS index for k-nearest neighbors.
        
        Args:
            query_embedding: Query embedding
            k: Number of nearest neighbors to return
            ef_search: HNSW search parameter (overrides default)
            
        Returns:
            ids: IDs of nearest neighbors
            distances: Distances to nearest neighbors
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Set search parameter
        if ef_search is None:
            ef_search = self.ef_search
        
        # Route query to shards
        shard_ids = self.sharder.route_queries(query_embedding)[0]
        
        # Calculate per-shard topK
        per_shard_topk = self.sharder.calculate_per_shard_topk(k)
        
        # Query each shard
        all_ids = []
        all_distances = []
        
        for shard_id in shard_ids:
            # Get segmenter for this shard
            segmenter = self._get_segmenter(shard_id)
            
            # Route query to segments
            segment_ids = segmenter.route_queries(query_embedding)[0]
            
            # Query each segment
            shard_ids = []
            shard_distances = []
            
            for segment_id in segment_ids:
                # Get HNSW index for this segment
                index = self._get_index(shard_id, segment_id)
                
                if index is None:
                    # No index found for this segment (might be empty)
                    continue
                
                # Query the index
                segment_ids, segment_distances = index.query(
                    query_embedding, 
                    k=min(per_shard_topk, len(index.ids)),
                    ef_search=ef_search
                )
                
                # Add to shard results
                shard_ids.extend(segment_ids[0])
                shard_distances.extend(segment_distances[0])
            
            # Sort results for this shard
            if shard_ids:
                # Sort by distance
                sorted_indices = np.argsort(shard_distances)
                sorted_ids = [shard_ids[i] for i in sorted_indices]
                sorted_distances = [shard_distances[i] for i in sorted_indices]
                
                # Take top-k
                all_ids.extend(sorted_ids[:per_shard_topk])
                all_distances.extend(sorted_distances[:per_shard_topk])
        
        # Sort and return top-k overall
        if all_ids:
            sorted_indices = np.argsort(all_distances)
            sorted_ids = [all_ids[i] for i in sorted_indices]
            sorted_distances = [all_distances[i] for i in sorted_indices]
            
            return sorted_ids[:k], sorted_distances[:k]
        else:
            return [], []
    
    def batch_query(self, 
                   query_embeddings: np.ndarray, 
                   k: int = 10,
                   ef_search: Optional[int] = None) -> Tuple[List[List[Any]], List[List[float]]]:
        """
        Query the LANNS index for k-nearest neighbors for multiple queries.
        
        Args:
            query_embeddings: Query embeddings
            k: Number of nearest neighbors to return
            ef_search: HNSW search parameter (overrides default)
            
        Returns:
            ids: List of lists of IDs of nearest neighbors for each query
            distances: List of lists of distances to nearest neighbors for each query
        """
        results_ids = []
        results_distances = []
        
        for i in range(len(query_embeddings)):
            query_ids, query_distances = self.query(
                query_embeddings[i].reshape(1, -1),
                k=k,
                ef_search=ef_search
            )
            
            results_ids.append(query_ids)
            results_distances.append(query_distances)
        
        return results_ids, results_distances