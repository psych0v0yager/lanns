"""
hnsw_utils.py - Integration with HNSW for nearest neighbor search
"""
import numpy as np
import time
import os
import pickle
from typing import List, Dict, Tuple, Any, Optional
import logging
import importlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try different HNSW libraries in order of preference
def get_hnsw_library():
    """
    Get the best available HNSW library.
    Returns the module object.
    """
    # Try nmslib first
    try:
        import nmslib
        logger.info("Using nmslib for HNSW implementation")
        return nmslib
    except ImportError:
        logger.warning("nmslib not found, trying hnswlib")
    
    # Try hnswlib
    try:
        import hnswlib
        logger.info("Using hnswlib for HNSW implementation")
        return hnswlib
    except ImportError:
        logger.warning("hnswlib not found, trying faiss")
    
    # Try faiss
    try:
        import faiss
        logger.info("Using faiss for HNSW implementation")
        return faiss
    except ImportError:
        logger.error("No HNSW implementation found. Please install nmslib, hnswlib, or faiss.")
        raise ImportError("No HNSW implementation found. Please install nmslib, hnswlib, or faiss.")

# Detect which library to use
hnsw_lib_name = None
hnsw_lib = None

def initialize_hnsw():
    """Initialize the HNSW library."""
    global hnsw_lib, hnsw_lib_name
    
    if hnsw_lib is not None:
        return
    
    try:
        # Try to import nmslib first (best performance)
        import nmslib
        hnsw_lib = nmslib
        hnsw_lib_name = "nmslib"
        logger.info("Using nmslib for HNSW implementation")
    except ImportError:
        try:
            # Try hnswlib next
            import hnswlib
            hnsw_lib = hnswlib
            hnsw_lib_name = "hnswlib"
            logger.info("Using hnswlib for HNSW implementation")
        except ImportError:
            try:
                # Finally try faiss
                import faiss
                hnsw_lib = faiss
                hnsw_lib_name = "faiss"
                logger.info("Using faiss for HNSW implementation")
            except ImportError:
                logger.error("No HNSW implementation found. Please install nmslib, hnswlib, or faiss.")
                raise ImportError("No HNSW implementation found. Please install nmslib, hnswlib, or faiss.")

class HNSWIndex:
    """Wrapper for HNSW index that works with different libraries"""
    
    def __init__(self, space='l2', M=16, ef_construction=200):
        """
        Initialize HNSW index.
        
        Args:
            space: Distance metric ('l2', 'cosine', etc.)
            M: HNSW parameter for max number of connections
            ef_construction: HNSW parameter for index building
        """
        initialize_hnsw()
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.index = None
        self.dim = None
        self.ids = None
    
    def build(self, embeddings: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        """
        Build HNSW index for the given embeddings.
        
        Args:
            embeddings: Data points to index
            ids: Optional ID for each data point
        """
        n_samples, self.dim = embeddings.shape
        
        if ids is None:
            ids = list(range(n_samples))
        self.ids = ids
        
        start_time = time.time()
        
        # Create and build index based on available library
        if hnsw_lib_name == "nmslib":
            self.index = hnsw_lib.init(method='hnsw', space=self.space)
            self.index.addDataPointBatch(embeddings)
            
            index_params = {
                'M': self.M,
                'efConstruction': self.ef_construction,
                'post': 0
            }
            self.index.createIndex(index_params)
            
        elif hnsw_lib_name == "hnswlib":
            # Convert space name
            space_map = {'l2': 'l2', 'cosine': 'cosine', 'ip': 'ip'}
            hnswlib_space = space_map.get(self.space, 'l2')
            
            self.index = hnsw_lib.Index(space=hnswlib_space, dim=self.dim)
            self.index.init_index(max_elements=n_samples, ef_construction=self.ef_construction, M=self.M)
            
            # Add data points
            self.index.add_items(embeddings, ids)
            
        elif hnsw_lib_name == "faiss":
            # Create HNSW index
            if self.space == 'l2':
                self.index = hnsw_lib.IndexHNSWFlat(self.dim, self.M)
            elif self.space == 'cosine':
                # For cosine similarity, normalize vectors first
                embeddings = embeddings.copy()
                faiss.normalize_L2(embeddings)
                self.index = hnsw_lib.IndexHNSWFlat(self.dim, self.M)
            else:
                # Inner product
                self.index = hnsw_lib.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
            
            # Set parameters
            self.index.hnsw.efConstruction = self.ef_construction
            
            # Train and add vectors
            self.index.train(embeddings)
            self.index.add(embeddings)
            
            # Store IDs separately since faiss doesn't support custom IDs directly
            self.ids = ids
        
        build_time = time.time() - start_time
        logger.info(f"Built HNSW index with {n_samples} points in {build_time:.2f}s")
    
    def query(self, queries: np.ndarray, k: int, ef_search: int = 100) -> Tuple[List[List[Any]], List[List[float]]]:
        """
        Query the HNSW index.
        
        Args:
            queries: Query embeddings
            k: Number of nearest neighbors to return
            ef_search: HNSW search parameter
            
        Returns:
            ids: List of lists of IDs of nearest neighbors for each query
            distances: List of lists of distances to nearest neighbors for each query
        """
        if self.index is None:
            raise ValueError("Index has not been built yet")
        
        if k > len(self.ids):
            logger.warning(f"Requested k={k} is greater than number of indexed items ({len(self.ids)}). Limiting to {len(self.ids)}.")
            k = min(k, len(self.ids))
        
        start_time = time.time()
        result_ids = []
        result_distances = []
        
        # Query index based on available library
        if hnsw_lib_name == "nmslib":
            # Set search parameters
            self.index.setQueryTimeParams({'efSearch': ef_search})
            
            # Query each point
            for query in queries:
                ids, distances = self.index.knnQuery(query, k=k)
                # Convert indices to original IDs
                original_ids = [self.ids[idx] for idx in ids]
                result_ids.append(original_ids)
                result_distances.append(distances)
            
        elif hnsw_lib_name == "hnswlib":
            # Set search parameters
            self.index.set_ef(ef_search)
            
            # Query all at once
            labels, distances = self.index.knn_query(queries, k=k)
            
            # Convert to lists
            for i in range(len(queries)):
                result_ids.append(labels[i].tolist())
                result_distances.append(distances[i].tolist())
            
        elif hnsw_lib_name == "faiss":
            # Set search parameters
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = ef_search
            
            # Normalize queries if using cosine similarity
            if self.space == 'cosine':
                queries = queries.copy()
                faiss.normalize_L2(queries)
            
            # Query index
            distances, indices = self.index.search(queries, k)
            
            # Convert indices to original IDs
            for i in range(len(queries)):
                original_ids = [self.ids[idx] if idx >= 0 and idx < len(self.ids) else None for idx in indices[i]]
                result_ids.append(original_ids)
                result_distances.append(distances[i].tolist())
        
        query_time = time.time() - start_time
        if len(queries) > 0:
            logger.debug(f"Queried {len(queries)} points in {query_time:.2f}s ({query_time/len(queries)*1000:.2f}ms per query)")
        
        return result_ids, result_distances
    
    def save(self, filename: str) -> None:
        """
        Save the HNSW index to disk.
        
        Args:
            filename: File path to save to
        """
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save based on available library
        if hnsw_lib_name == "nmslib":
            # For nmslib, save the index and metadata separately
            self.index.saveIndex(filename)
            
            # Save metadata
            metadata = {
                'space': self.space,
                'M': self.M,
                'ef_construction': self.ef_construction,
                'dim': self.dim,
                'ids': self.ids
            }
            with open(f"{filename}.meta", 'wb') as f:
                pickle.dump(metadata, f)
            
        elif hnsw_lib_name == "hnswlib":
            # For hnswlib, save the index and metadata separately
            self.index.save_index(filename)
            
            # Save metadata
            metadata = {
                'space': self.space,
                'M': self.M,
                'ef_construction': self.ef_construction,
                'dim': self.dim,
                'ids': self.ids
            }
            with open(f"{filename}.meta", 'wb') as f:
                pickle.dump(metadata, f)
            
        elif hnsw_lib_name == "faiss":
            # For faiss, use built-in write_index
            hnsw_lib.write_index(self.index, filename)
            
            # Save metadata
            metadata = {
                'space': self.space,
                'M': self.M,
                'ef_construction': self.ef_construction,
                'dim': self.dim,
                'ids': self.ids
            }
            with open(f"{filename}.meta", 'wb') as f:
                pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, filename: str) -> 'HNSWIndex':
        """
        Load HNSW index from disk.
        
        Args:
            filename: File path to load from
            
        Returns:
            Loaded HNSW index
        """
        initialize_hnsw()
        
        # Load metadata
        with open(f"{filename}.meta", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create index
        index = cls(space=metadata['space'], M=metadata['M'], ef_construction=metadata['ef_construction'])
        index.dim = metadata['dim']
        index.ids = metadata['ids']
        
        # Load based on available library
        if hnsw_lib_name == "nmslib":
            index.index = hnsw_lib.init(method='hnsw', space=index.space)
            index.index.loadIndex(filename)
            
        elif hnsw_lib_name == "hnswlib":
            # Convert space name
            space_map = {'l2': 'l2', 'cosine': 'cosine', 'ip': 'ip'}
            hnswlib_space = space_map.get(index.space, 'l2')
            
            index.index = hnsw_lib.Index(space=hnswlib_space, dim=index.dim)
            index.index.load_index(filename, max_elements=len(index.ids))
            
        elif hnsw_lib_name == "faiss":
            index.index = hnsw_lib.read_index(filename)
        
        return index