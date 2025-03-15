"""
segmenters.py - Implementation of segmentation strategies for LANNS
"""
import numpy as np
from typing import Tuple, List, Dict, Any
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmenterBase:
    """Base class for all segmentation strategies"""
    
    def __init__(self, num_segments: int, spill: float = 0.15):
        """
        Initialize the segmenter.
        
        Args:
            num_segments: Number of segments to create
            spill: Spill parameter (0.15 means 30% queries go to both sides)
        """
        self.num_segments = num_segments
        self.spill = spill
        self.depth = int(np.ceil(np.log2(num_segments)))
        self.tree_params = []
        
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the segmenter to the data.
        
        Args:
            embeddings: Data points to segment
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign segment IDs to data points (without spill).
        
        Args:
            embeddings: Data points to assign to segments
            
        Returns:
            segment_ids: Segment ID for each data point
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def route_queries(self, queries: np.ndarray) -> List[List[int]]:
        """
        Route queries to segments, with spill.
        
        Args:
            queries: Query points to route
            
        Returns:
            List of lists, where each inner list contains
            segment IDs that the query should be routed to
        """
        raise NotImplementedError("Subclasses must implement route_queries()")
    
    def save(self, filename: str) -> None:
        """
        Save the segmenter to disk.
        
        Args:
            filename: File path to save to
        """
        np.save(filename, {
            'num_segments': self.num_segments,
            'spill': self.spill,
            'depth': self.depth,
            'tree_params': self.tree_params
        })
    
    @classmethod
    def load(cls, filename: str) -> 'SegmenterBase':
        """
        Load a segmenter from disk.
        
        Args:
            filename: File path to load from
            
        Returns:
            Loaded segmenter
        """
        data = np.load(filename, allow_pickle=True).item()
        segmenter = cls(data['num_segments'], data['spill'])
        segmenter.depth = data['depth']
        segmenter.tree_params = data['tree_params']
        return segmenter


class RandomSegmenter(SegmenterBase):
    """Random Segmenter (RS): Randomly assigns data points to segments"""
    
    def fit(self, embeddings: np.ndarray) -> None:
        """No fitting needed for random segmenter"""
        logger.info(f"Using random segmenter with {self.num_segments} segments")
        pass
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Randomly assign segment IDs"""
        return np.random.randint(0, self.num_segments, size=len(embeddings))
    
    def route_queries(self, queries: np.ndarray) -> List[List[int]]:
        """Route each query to all segments"""
        return [list(range(self.num_segments)) for _ in range(len(queries))]


class RandomHyperplaneSegmenter(SegmenterBase):
    """
    Random Hyperplane Segmenter (RH): 
    Uses random hyperplanes to split data.
    """
    
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the RH segmenter by building a tree of random hyperplanes.
        
        Args:
            embeddings: Data points to segment
        """
        logger.info(f"Fitting Random Hyperplane segmenter with {self.num_segments} segments...")
        start_time = time.time()
        
        n_samples, dim = embeddings.shape
        
        # Build the tree level by level
        current_nodes = {0: np.arange(n_samples)}
        self.tree_params = []
        
        for level in range(self.depth):
            level_params = []
            new_nodes = {}
            
            for node_id, node_points in current_nodes.items():
                if len(node_points) <= 1:
                    continue
                    
                # Generate random hyperplane
                hyperplane = np.random.randn(dim)
                hyperplane /= np.linalg.norm(hyperplane)
                
                # Project points onto hyperplane
                node_data = embeddings[node_points]
                projections = np.dot(node_data, hyperplane)
                
                # Find median for splitting
                median = np.median(projections)
                
                # Calculate spill boundaries
                n_points = len(projections)
                sorted_proj = np.sort(projections)
                left_boundary = sorted_proj[int((0.5 - self.spill) * n_points)]
                right_boundary = sorted_proj[int((0.5 + self.spill) * n_points)]
                
                # Store node parameters for query routing
                level_params.append({
                    'node_id': node_id,
                    'hyperplane': hyperplane,
                    'median': median,
                    'left_boundary': left_boundary,
                    'right_boundary': right_boundary
                })
                
                # Split based on median (no spill for data points)
                left_points = node_points[projections < median]
                right_points = node_points[projections >= median]
                
                # Assign to new nodes
                left_node_id = node_id * 2 + 1
                right_node_id = node_id * 2 + 2
                
                new_nodes[left_node_id] = left_points
                new_nodes[right_node_id] = right_points
            
            self.tree_params.append(level_params)
            current_nodes = new_nodes
        
        # Verify the tree is correctly formed
        logger.info(f"RH segmenter tree built with {len(self.tree_params)} levels")
        logger.info(f"RH segmenter fit in {time.time() - start_time:.2f}s")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign segment IDs to data points (without spill).
        
        Args:
            embeddings: Data points to assign to segments
            
        Returns:
            segment_ids: Segment ID for each data point
        """
        n_samples = len(embeddings)
        segment_ids = np.zeros(n_samples, dtype=int)
        
        for i, embedding in enumerate(embeddings):
            node_id = 0
            
            for level in range(self.depth):
                # Find this node's parameters
                node_params = None
                for params in self.tree_params[level]:
                    if params['node_id'] == node_id:
                        node_params = params
                        break
                
                if node_params is None:
                    # This should not happen if the tree is correctly formed
                    break
                
                # Project onto hyperplane
                projection = np.dot(embedding, node_params['hyperplane'])
                
                # Route based on median (no spill)
                if projection < node_params['median']:
                    node_id = node_id * 2 + 1  # Left child
                else:
                    node_id = node_id * 2 + 2  # Right child
            
            # Convert final node ID to segment ID
            segment_ids[i] = node_id % self.num_segments
        
        return segment_ids
    
    def route_queries(self, queries: np.ndarray) -> List[List[int]]:
        """
        Route queries to segments, with spill.
        
        Args:
            queries: Query points to route
            
        Returns:
            List of lists, where each inner list contains
            segment IDs that the query should be routed to
        """
        routed_segments = []
        
        for query in queries:
            # Start at root node
            current_nodes = [0]
            
            # Traverse the tree with spill
            for level in range(self.depth):
                next_nodes = []
                
                for node_id in current_nodes:
                    # Find this node's parameters
                    node_params = None
                    for params in self.tree_params[level]:
                        if params['node_id'] == node_id:
                            node_params = params
                            break
                    
                    if node_params is None:
                        continue
                    
                    # Project onto hyperplane
                    projection = np.dot(query, node_params['hyperplane'])
                    
                    # Route with spill
                    if projection < node_params['left_boundary']:
                        # Only route left
                        next_nodes.append(node_id * 2 + 1)
                    elif projection > node_params['right_boundary']:
                        # Only route right
                        next_nodes.append(node_id * 2 + 2)
                    else:
                        # Spill: route both directions
                        next_nodes.append(node_id * 2 + 1)
                        next_nodes.append(node_id * 2 + 2)
                
                current_nodes = next_nodes
            
            # Convert node IDs to segment IDs
            segment_ids = [node_id % self.num_segments for node_id in current_nodes]
            routed_segments.append(list(set(segment_ids)))  # Remove duplicates
        
        return routed_segments


class APDSegmenter(SegmenterBase):
    """
    Approximate Principal Direction Segmenter (APD): 
    Uses approximate eigenvectors for partitioning.
    """
    
    def __init__(self, num_segments: int, spill: float = 0.15, num_iterations: int = 10):
        """
        Initialize the APD segmenter.
        
        Args:
            num_segments: Number of segments to create
            spill: Spill parameter (0.15 means 30% queries go to both sides)
            num_iterations: Number of power iterations for eigenvector approximation
        """
        super().__init__(num_segments, spill)
        self.num_iterations = num_iterations
    
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the APD segmenter by building a tree of approximate principal directions.
        
        Args:
            embeddings: Data points to segment
        """
        logger.info(f"Fitting APD segmenter with {self.num_segments} segments...")
        start_time = time.time()
        
        n_samples, dim = embeddings.shape
        
        # Build the tree level by level
        current_nodes = {0: np.arange(n_samples)}
        self.tree_params = []
        
        for level in range(self.depth):
            level_params = []
            new_nodes = {}
            
            for node_id, node_points in current_nodes.items():
                if len(node_points) <= 1:
                    continue
                    
                # Get data for this node
                node_data = embeddings[node_points]
                
                # Approximate second eigenvector using power iteration
                v = np.random.randn(dim)
                v = v - np.mean(v)  # Ensure orthogonal to first eigenvector
                v /= np.linalg.norm(v)
                
                for _ in range(self.num_iterations):
                    # Approximate matrix-vector product with D*D^T*v
                    v = node_data.T @ (node_data @ v)
                    v = v - np.mean(v)  # Keep orthogonal to first eigenvector
                    v_norm = np.linalg.norm(v)
                    if v_norm > 1e-10:
                        v /= v_norm
                    else:
                        v = np.random.randn(dim)
                        v = v - np.mean(v)
                        v /= np.linalg.norm(v)
                
                # Project points onto eigenvector
                projections = np.dot(node_data, v)
                
                # Find median for splitting
                median = np.median(projections)
                
                # Calculate spill boundaries
                n_points = len(projections)
                sorted_proj = np.sort(projections)
                left_boundary = sorted_proj[int((0.5 - self.spill) * n_points)]
                right_boundary = sorted_proj[int((0.5 + self.spill) * n_points)]
                
                # Store node parameters for query routing
                level_params.append({
                    'node_id': node_id,
                    'hyperplane': v,  # Eigenvector as hyperplane
                    'median': median,
                    'left_boundary': left_boundary,
                    'right_boundary': right_boundary
                })
                
                # Split based on median (no spill for data points)
                left_points = node_points[projections < median]
                right_points = node_points[projections >= median]
                
                # Assign to new nodes
                left_node_id = node_id * 2 + 1
                right_node_id = node_id * 2 + 2
                
                new_nodes[left_node_id] = left_points
                new_nodes[right_node_id] = right_points
            
            self.tree_params.append(level_params)
            current_nodes = new_nodes
        
        logger.info(f"APD segmenter tree built with {len(self.tree_params)} levels")
        logger.info(f"APD segmenter fit in {time.time() - start_time:.2f}s")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Same as RH segmenter's predict method"""
        n_samples = len(embeddings)
        segment_ids = np.zeros(n_samples, dtype=int)
        
        for i, embedding in enumerate(embeddings):
            node_id = 0
            
            for level in range(self.depth):
                # Find this node's parameters
                node_params = None
                for params in self.tree_params[level]:
                    if params['node_id'] == node_id:
                        node_params = params
                        break
                
                if node_params is None:
                    break
                
                # Project onto hyperplane
                projection = np.dot(embedding, node_params['hyperplane'])
                
                # Route based on median (no spill)
                if projection < node_params['median']:
                    node_id = node_id * 2 + 1  # Left child
                else:
                    node_id = node_id * 2 + 2  # Right child
            
            # Convert final node ID to segment ID
            segment_ids[i] = node_id % self.num_segments
        
        return segment_ids
    
    def route_queries(self, queries: np.ndarray) -> List[List[int]]:
        """Same as RH segmenter's route_queries method"""
        routed_segments = []
        
        for query in queries:
            # Start at root node
            current_nodes = [0]
            
            # Traverse the tree with spill
            for level in range(self.depth):
                next_nodes = []
                
                for node_id in current_nodes:
                    # Find this node's parameters
                    node_params = None
                    for params in self.tree_params[level]:
                        if params['node_id'] == node_id:
                            node_params = params
                            break
                    
                    if node_params is None:
                        continue
                    
                    # Project onto hyperplane
                    projection = np.dot(query, node_params['hyperplane'])
                    
                    # Route with spill
                    if projection < node_params['left_boundary']:
                        # Only route left
                        next_nodes.append(node_id * 2 + 1)
                    elif projection > node_params['right_boundary']:
                        # Only route right
                        next_nodes.append(node_id * 2 + 2)
                    else:
                        # Spill: route both directions
                        next_nodes.append(node_id * 2 + 1)
                        next_nodes.append(node_id * 2 + 2)
                
                current_nodes = next_nodes
            
            # Convert node IDs to segment IDs
            segment_ids = [node_id % self.num_segments for node_id in current_nodes]
            routed_segments.append(list(set(segment_ids)))  # Remove duplicates
        
        return routed_segments