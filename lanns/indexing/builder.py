"""
builder.py - LANNS index builder implementation
"""
import os
import numpy as np
import json
import time
import pickle
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
from tqdm import tqdm

from lanns.core.segmenters import SegmenterBase, RandomSegmenter, RandomHyperplaneSegmenter, APDSegmenter
from lanns.core.sharding import Sharder
from lanns.core.hnsw_utils import HNSWIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LANNSIndexBuilder:
    """
    Builder for LANNS index.
    Implements the two-level partitioning strategy (sharding and segmentation).
    """
    
    def __init__(self, 
                 num_shards: int,
                 num_segments: int,
                 segmenter_type: str = 'apd',
                 spill: float = 0.15,
                 space: str = 'l2',
                 hnsw_m: int = 16,
                 hnsw_ef_construction: int = 200,
                 max_workers: int = None):
        """
        Initialize the LANNS index builder.
        
        Args:
            num_shards: Number of shards
            num_segments: Number of segments per shard
            segmenter_type: Segmentation strategy ('rs', 'rh', or 'apd')
            spill: Spill parameter for segmenters
            space: Distance metric ('l2', 'cosine', etc.)
            hnsw_m: HNSW parameter for max number of connections
            hnsw_ef_construction: HNSW parameter for index building
            max_workers: Maximum number of parallel workers (default: number of CPU cores)
        """
        self.num_shards = num_shards
        self.num_segments = num_segments
        self.segmenter_type = segmenter_type.lower()
        self.spill = spill
        self.space = space
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.max_workers = max_workers
        
        # Initialize sharder
        self.sharder = Sharder(num_shards)
        
        # Segmenters (one per shard)
        self.segmenters = {}
        
        # Initialize metadata
        self.metadata = {
            'num_shards': num_shards,
            'num_segments': num_segments,
            'segmenter_type': segmenter_type,
            'spill': spill,
            'space': space,
            'hnsw_m': hnsw_m,
            'hnsw_ef_construction': hnsw_ef_construction,
            'creation_time': None,
            'build_time': None,
            'total_indexed_points': 0,
            'points_per_shard': {},
            'points_per_segment': {}
        }
        
        logger.info(f"Initialized LANNS index builder with {num_shards} shards and {num_segments} segments per shard")
        logger.info(f"Using {segmenter_type} segmenter with spill={spill}")
    
    def _create_segmenter(self) -> SegmenterBase:
        """Create a segmenter based on the configured type"""
        if self.segmenter_type == 'rs':
            return RandomSegmenter(self.num_segments, self.spill)
        elif self.segmenter_type == 'rh':
            return RandomHyperplaneSegmenter(self.num_segments, self.spill)
        elif self.segmenter_type == 'apd':
            return APDSegmenter(self.num_segments, self.spill)
        else:
            raise ValueError(f"Unknown segmenter type: {self.segmenter_type}")
    
    def build(self, 
              embeddings: np.ndarray, 
              ids: List[Any],
              output_dir: str) -> None:
        """
        Build the LANNS index for the given embeddings.
        
        Args:
            embeddings: Data embeddings to index
            ids: ID for each data point
            output_dir: Directory to save the index
        """
        # Verify inputs
        if len(embeddings) != len(ids):
            raise ValueError(f"Length mismatch: {len(embeddings)} embeddings vs {len(ids)} IDs")
        
        start_time = time.time()
        n_samples, dim = embeddings.shape
        
        logger.info(f"Building LANNS index for {n_samples} points with dimension {dim}")
        logger.info(f"Index will be saved to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata['creation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata['total_indexed_points'] = n_samples
        self.metadata['dimension'] = dim
        
        # Step 1: Assign data points to shards
        logger.info("Step 1: Assigning data points to shards")
        shard_ids = self.sharder.assign_shards(ids)
        
        # Step 2: Process each shard
        logger.info("Step 2: Processing shards")
        
        # Group by shard
        shard_groups = {}
        for i in range(n_samples):
            shard_id = shard_ids[i]
            if shard_id not in shard_groups:
                shard_groups[shard_id] = []
            shard_groups[shard_id].append(i)
        
        # Create a common segmenter for all shards
        logger.info(f"Creating {self.segmenter_type} segmenter")
        
        # For APD and RH, learn on a sample
        if self.segmenter_type in ['apd', 'rh']:
            # Create a segmenter
            segmenter = self._create_segmenter()
            
            # Sample data for learning (max 100k points)
            sample_size = min(100000, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            # Fit segmenter
            logger.info(f"Fitting segmenter on {sample_size} sample points")
            segmenter.fit(sample_embeddings)
            
            # Save segmenter for each shard
            for shard_id in range(self.num_shards):
                self.segmenters[shard_id] = segmenter
                
            # Save the common segmenter
            segmenter_path = os.path.join(output_dir, 'common_segmenter.pkl')
            with open(segmenter_path, 'wb') as f:
                pickle.dump(segmenter, f)
        
        # Process each shard - potentially in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for shard_id in range(self.num_shards):
                # Get data for this shard
                if shard_id in shard_groups:
                    shard_indices = shard_groups[shard_id]
                    shard_embeddings = embeddings[shard_indices]
                    shard_ids_list = [ids[i] for i in shard_indices]
                    
                    # Log shard info
                    logger.info(f"Shard {shard_id}: {len(shard_indices)} points")
                    self.metadata['points_per_shard'][str(shard_id)] = len(shard_indices)
                    
                    # Create shard directory
                    shard_dir = os.path.join(output_dir, f'shard_{shard_id}')
                    os.makedirs(shard_dir, exist_ok=True)
                    
                    # Submit task to executor
                    future = executor.submit(
                        self._process_shard, 
                        shard_id, 
                        shard_embeddings, 
                        shard_ids_list,
                        shard_dir,
                        self.segmenter_type
                    )
                    futures[future] = shard_id
                else:
                    logger.warning(f"Shard {shard_id} has no data points")
                    self.metadata['points_per_shard'][str(shard_id)] = 0
            
            # Wait for all futures to complete
            with tqdm(total=len(futures), desc="Building shards") as pbar:
                for future in as_completed(futures):
                    shard_id = futures[future]
                    try:
                        # Get segment distribution for this shard
                        segment_distribution = future.result()
                        
                        # Update metadata
                        for segment_id, count in segment_distribution.items():
                            key = f"{shard_id}_{segment_id}"
                            self.metadata['points_per_segment'][key] = count
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing shard {shard_id}: {str(e)}")
                        raise
        
        # Save metadata
        self.metadata['build_time'] = time.time() - start_time
        logger.info(f"Index built in {self.metadata['build_time']:.2f}s")
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"LANNS index saved to {output_dir}")
        logger.info(f"Total indexed points: {n_samples}")
        logger.info(f"Average build time per point: {self.metadata['build_time'] / n_samples * 1000:.2f}ms")
    
    def _process_shard(self, 
                       shard_id: int, 
                       embeddings: np.ndarray, 
                       ids: List[Any],
                       shard_dir: str,
                       segmenter_type: str) -> Dict[str, int]:
        """
        Process a shard by segmenting it and building HNSW indices.
        
        Args:
            shard_id: ID of the shard
            embeddings: Embeddings for this shard
            ids: IDs for this shard
            shard_dir: Directory to save the shard
            segmenter_type: Type of segmenter to use
            
        Returns:
            segment_distribution: Dictionary mapping segment IDs to counts
        """
        try:
            n_samples = len(embeddings)
            
            # Step 2.1: Create or load segmenter
            if segmenter_type == 'rs':
                # For RS, create a new segmenter
                segmenter = RandomSegmenter(self.num_segments, self.spill)
                segmenter.fit(embeddings)
            else:
                # For APD and RH, use the common segmenter
                segmenter = self.segmenters.get(shard_id)
                if segmenter is None:
                    # If not available (e.g., in a subprocess), load it
                    segmenter_path = os.path.join(os.path.dirname(shard_dir), 'common_segmenter.pkl')
                    with open(segmenter_path, 'rb') as f:
                        segmenter = pickle.load(f)
            
            # Step 2.2: Assign data points to segments
            segment_ids = segmenter.predict(embeddings)
            
            # Track segment distribution
            segment_distribution = {}
            
            # Step 2.3: Build HNSW index for each segment
            segment_groups = {}
            for i in range(n_samples):
                segment_id = segment_ids[i]
                if segment_id not in segment_groups:
                    segment_groups[segment_id] = []
                segment_groups[segment_id].append(i)
            
            # Create segment directory
            os.makedirs(shard_dir, exist_ok=True)
            
            # Save segmenter
            segmenter_path = os.path.join(shard_dir, 'segmenter.pkl')
            with open(segmenter_path, 'wb') as f:
                pickle.dump(segmenter, f)
            
            # Process each segment
            for segment_id in range(self.num_segments):
                segment_path = os.path.join(shard_dir, f'segment_{segment_id}')
                os.makedirs(segment_path, exist_ok=True)
                
                if segment_id in segment_groups:
                    segment_indices = segment_groups[segment_id]
                    segment_embeddings = embeddings[segment_indices]
                    segment_ids_list = [ids[i] for i in segment_indices]
                    
                    # Track segment distribution
                    segment_distribution[str(segment_id)] = len(segment_indices)
                    
                    # Build HNSW index
                    hnsw_index = HNSWIndex(
                        space=self.space,
                        M=self.hnsw_m,
                        ef_construction=self.hnsw_ef_construction
                    )
                    hnsw_index.build(segment_embeddings, segment_ids_list)
                    
                    # Save index
                    hnsw_index.save(os.path.join(segment_path, 'hnsw_index'))
                else:
                    # Create empty segment
                    segment_distribution[str(segment_id)] = 0
            
            return segment_distribution
            
        except Exception as e:
            logger.error(f"Error in _process_shard for shard {shard_id}: {str(e)}")
            raise

    def build_from_files(self, 
                         embeddings_dir: str, 
                         ids_file: Optional[str] = None,
                         output_dir: str = 'lanns_index',
                         batch_size: int = 100000) -> None:
        """
        Build LANNS index from embedding files.
        
        This method supports reading embeddings from numpy files or an HDF5 file,
        and can process data in batches to handle large datasets.
        
        Args:
            embeddings_dir: Directory containing embedding files or path to an HDF5 file
            ids_file: Path to a JSON file containing IDs (optional)
            output_dir: Directory to save the index
            batch_size: Batch size for processing large datasets
        """
        # Check embeddings_dir type
        if embeddings_dir.endswith('.h5') or embeddings_dir.endswith('.hdf5'):
            # HDF5 file
            self._build_from_h5(embeddings_dir, ids_file, output_dir, batch_size)
        elif os.path.isdir(embeddings_dir):
            # Directory of npy files
            self._build_from_dir(embeddings_dir, ids_file, output_dir, batch_size)
        elif embeddings_dir.endswith('.npy'):
            # Single npy file
            self._build_from_npy(embeddings_dir, ids_file, output_dir, batch_size)
        else:
            raise ValueError(f"Unsupported embeddings source: {embeddings_dir}")
    
    def _build_from_npy(self, 
                       embeddings_file: str, 
                       ids_file: Optional[str], 
                       output_dir: str,
                       batch_size: int) -> None:
        """Build from a single numpy file"""
        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        
        # Load IDs
        if ids_file:
            logger.info(f"Loading IDs from {ids_file}")
            with open(ids_file, 'r') as f:
                ids = json.load(f)
        else:
            # Generate sequential IDs
            ids = list(range(len(embeddings)))
        
        # Build index
        self.build(embeddings, ids, output_dir)
    
    def _build_from_dir(self, 
                       embeddings_dir: str, 
                       ids_file: Optional[str], 
                       output_dir: str,
                       batch_size: int) -> None:
        """Build from directory of numpy files"""
        start_time = time.time()
        
        # List all numpy files
        npy_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy') and f.startswith('embeddings_')]
        npy_files.sort(key=lambda x: int(x.split('_')[1]))
        
        # Load IDs
        all_ids = []
        if ids_file:
            logger.info(f"Loading IDs from {ids_file}")
            with open(ids_file, 'r') as f:
                all_ids = json.load(f)
        
        # Check if we can process all at once
        total_size = sum(os.path.getsize(os.path.join(embeddings_dir, f)) for f in npy_files)
        if total_size < 1e9:  # 1 GB threshold
            # Small dataset - process all at once
            logger.info("Loading all embeddings at once")
            all_embeddings = []
            
            for npy_file in npy_files:
                file_path = os.path.join(embeddings_dir, npy_file)
                embeddings = np.load(file_path)
                all_embeddings.append(embeddings)
            
            # Combine all embeddings
            all_embeddings = np.vstack(all_embeddings)
            
            # If no IDs provided, generate them
            if not all_ids:
                all_ids = list(range(len(all_embeddings)))
            
            # Build index
            self.build(all_embeddings, all_ids, output_dir)
            
        else:
            # Large dataset - process in batches
            logger.info(f"Processing {len(npy_files)} embedding files in batches")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize metadata for batch processing
            dimension = None
            total_points = 0
            points_per_shard = {str(i): 0 for i in range(self.num_shards)}
            points_per_segment = {}
            
            # Create directory structure
            for shard_id in range(self.num_shards):
                shard_dir = os.path.join(output_dir, f'shard_{shard_id}')
                os.makedirs(shard_dir, exist_ok=True)
                
                for segment_id in range(self.num_segments):
                    segment_path = os.path.join(shard_dir, f'segment_{segment_id}')
                    os.makedirs(segment_path, exist_ok=True)
            
            # Process in batches
            start_idx = 0
            
            # Process first batch to get dimension
            first_file = npy_files[0]
            first_file_path = os.path.join(embeddings_dir, first_file)
            first_batch = np.load(first_file_path)
            dimension = first_batch.shape[1]
            
            # Common segmenter setup
            if self.segmenter_type in ['apd', 'rh']:
                # Create a segmenter
                segmenter = self._create_segmenter()
                
                # Fit segmenter on first batch
                logger.info(f"Fitting {self.segmenter_type} segmenter on first batch")
                segmenter.fit(first_batch)
                
                # Save the common segmenter
                segmenter_path = os.path.join(output_dir, 'common_segmenter.pkl')
                with open(segmenter_path, 'wb') as f:
                    pickle.dump(segmenter, f)
                
                # Save segmenter for each shard
                for shard_id in range(self.num_shards):
                    self.segmenters[shard_id] = segmenter
                    
                    # Also save it to the shard directory
                    shard_dir = os.path.join(output_dir, f'shard_{shard_id}')
                    segmenter_path = os.path.join(shard_dir, 'segmenter.pkl')
                    with open(segmenter_path, 'wb') as f:
                        pickle.dump(segmenter, f)
            
            for batch_idx, npy_file in enumerate(tqdm(npy_files, desc="Processing embedding files")):
                file_path = os.path.join(embeddings_dir, npy_file)
                batch_embeddings = np.load(file_path)
                current_batch_size = len(batch_embeddings)
                
                # Get IDs for this batch
                if ids_file:
                    batch_ids = all_ids[start_idx:start_idx + current_batch_size]
                else:
                    batch_ids = list(range(start_idx, start_idx + current_batch_size))
                
                # Assign to shards
                shard_ids = self.sharder.assign_shards(batch_ids)
                
                # Group by shard
                shard_groups = {}
                for i in range(current_batch_size):
                    shard_id = shard_ids[i]
                    if shard_id not in shard_groups:
                        shard_groups[shard_id] = []
                    shard_groups[shard_id].append(i)
                
                # Process each shard
                for shard_id in range(self.num_shards):
                    shard_dir = os.path.join(output_dir, f'shard_{shard_id}')
                    
                    # Check if we have data for this shard
                    if shard_id in shard_groups:
                        shard_indices = shard_groups[shard_id]
                        shard_embeddings = batch_embeddings[shard_indices]
                        shard_ids_list = [batch_ids[i] for i in shard_indices]
                        
                        # Update metadata
                        points_per_shard[str(shard_id)] += len(shard_indices)
                        
                        # For the first batch, create the segmenter (if not already created)
                        if batch_idx == 0 and self.segmenter_type == 'rs':
                            # For RS, create a new segmenter for each shard
                            segmenter = RandomSegmenter(self.num_segments, self.spill)
                            segmenter.fit(shard_embeddings)
                            
                            # Save segmenter
                            segmenter_path = os.path.join(shard_dir, 'segmenter.pkl')
                            with open(segmenter_path, 'wb') as f:
                                pickle.dump(segmenter, f)
                            
                            self.segmenters[shard_id] = segmenter
                        elif shard_id not in self.segmenters:
                            # Load segmenter if not in memory
                            segmenter_path = os.path.join(shard_dir, 'segmenter.pkl')
                            if os.path.exists(segmenter_path):
                                with open(segmenter_path, 'rb') as f:
                                    self.segmenters[shard_id] = pickle.load(f)
                            else:
                                # If segmenter not found, check common segmenter
                                common_segmenter_path = os.path.join(output_dir, 'common_segmenter.pkl')
                                if os.path.exists(common_segmenter_path):
                                    with open(common_segmenter_path, 'rb') as f:
                                        self.segmenters[shard_id] = pickle.load(f)
                        
                        segmenter = self.segmenters[shard_id]
                        
                        # Assign to segments
                        segment_ids = segmenter.predict(shard_embeddings)
                        
                        # Group by segment
                        segment_groups = {}
                        for i in range(len(shard_indices)):
                            segment_id = segment_ids[i]
                            if segment_id not in segment_groups:
                                segment_groups[segment_id] = []
                            segment_groups[segment_id].append(i)
                        
                        # Process each segment
                        for segment_id in range(self.num_segments):
                            segment_path = os.path.join(shard_dir, f'segment_{segment_id}')
                            
                            # Check if we have data for this segment
                            if segment_id in segment_groups:
                                segment_indices = segment_groups[segment_id]
                                segment_embeddings = shard_embeddings[segment_indices]
                                segment_ids_list = [shard_ids_list[i] for i in segment_indices]
                                
                                # Update metadata
                                segment_key = f"{shard_id}_{segment_id}"
                                if segment_key in points_per_segment:
                                    points_per_segment[segment_key] += len(segment_indices)
                                else:
                                    points_per_segment[segment_key] = len(segment_indices)
                                
                                # Update or create HNSW index
                                index_path = os.path.join(segment_path, 'hnsw_index')
                                
                                # If first batch or index doesn't exist, create new index
                                if batch_idx == 0 or not os.path.exists(f"{index_path}.meta"):
                                    hnsw_index = HNSWIndex(
                                        space=self.space,
                                        M=self.hnsw_m,
                                        ef_construction=self.hnsw_ef_construction
                                    )
                                    hnsw_index.build(segment_embeddings, segment_ids_list)
                                    hnsw_index.save(index_path)
                                else:
                                    # For now, we don't support incremental updates
                                    # This is a limitation of most HNSW libraries
                                    logger.warning("Incremental updates not supported. Skipping additional data points.")
                
                # Update counters
                start_idx += current_batch_size
                total_points += current_batch_size
            
            # Save metadata.json
            self.metadata['creation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.metadata['total_indexed_points'] = total_points
            self.metadata['dimension'] = dimension
            self.metadata['points_per_shard'] = points_per_shard
            self.metadata['points_per_segment'] = points_per_segment
            self.metadata['build_time'] = time.time() - start_time
            
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"LANNS index saved to {output_dir}")
            logger.info(f"Total indexed points: {total_points}")
            logger.info(f"Index built in {self.metadata['build_time']:.2f}s")
            logger.info(f"Average build time per point: {self.metadata['build_time'] / total_points * 1000:.2f}ms")


    def _build_from_h5(self, 
                     h5_file: str, 
                     ids_file: Optional[str], 
                     output_dir: str,
                     batch_size: int) -> None:
        """Build from HDF5 file"""
        # Open HDF5 file
        logger.info(f"Loading embeddings from {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            # Check if we have embeddings and IDs datasets
            if 'embeddings' not in f:
                raise ValueError(f"HDF5 file {h5_file} does not contain 'embeddings' dataset")
            
            embeddings_dataset = f['embeddings']
            total_points = embeddings_dataset.shape[0]
            
            # Get IDs
            if 'ids' in f:
                # IDs are in the HDF5 file
                ids_dataset = f['ids']
                ids_in_h5 = True
            elif ids_file:
                # IDs are in a separate file
                logger.info(f"Loading IDs from {ids_file}")
                with open(ids_file, 'r') as ids_f:
                    all_ids = json.load(ids_f)
                ids_in_h5 = False
            else:
                # Generate sequential IDs
                all_ids = list(range(total_points))
                ids_in_h5 = False
            
            # Check if we can process all at once
            if total_points <= batch_size:
                # Small dataset - process all at once
                logger.info(f"Loading all {total_points} embeddings at once")
                embeddings = embeddings_dataset[:]
                
                if ids_in_h5:
                    ids = ids_dataset[:]
                else:
                    ids = all_ids
                
                # Build index
                self.build(embeddings, ids, output_dir)
                
            else:
                # Large dataset - process in batches
                logger.info(f"Processing {total_points} embeddings in batches of {batch_size}")
                
                # Create empty index directories
                os.makedirs(output_dir, exist_ok=True)
                for shard_id in range(self.num_shards):
                    shard_dir = os.path.join(output_dir, f'shard_{shard_id}')
                    os.makedirs(shard_dir, exist_ok=True)
                    
                    for segment_id in range(self.num_segments):
                        segment_path = os.path.join(shard_dir, f'segment_{segment_id}')
                        os.makedirs(segment_path, exist_ok=True)
                
                # Process in batches
                num_batches = (total_points + batch_size - 1) // batch_size
                
                with tqdm(total=num_batches, desc=f"Processing {h5_file} in batches") as pbar:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, total_points)
                        current_batch_size = end_idx - start_idx
                        
                        # Load batch embeddings
                        batch_embeddings = embeddings_dataset[start_idx:end_idx]
                        
                        # Load batch IDs
                        if ids_in_h5:
                            batch_ids = ids_dataset[start_idx:end_idx]
                        else:
                            batch_ids = all_ids[start_idx:end_idx]
                        
                        # Process batch
                        # TODO: Implement batch processing
                        # This would be similar to the implementation in _build_from_dir
                        
                        pbar.update(1)