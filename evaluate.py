#!/usr/bin/env python
"""
evaluate.py - Script to evaluate LANNS performance

This script measures:
1. Build time
2. Query time / QPS
3. Recall compared to exact search

Usage:
    python evaluate.py --embeddings_file all_embeddings.npy --query_file query_embeddings.npy --output_dir ./eval_results
    python evaluate.py --embeddings_dir ./embeddings --query_file query_embeddings.npy --output_dir ./eval_results
"""
import os
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, Optional
import logging
from scipy.spatial.distance import cdist
from tqdm import tqdm
import shutil

from lanns.indexing.builder import LANNSIndexBuilder
from lanns.indexing.storage import LANNSIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LANNS performance')
    
    # Data - Add support for directories
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--embeddings_file', type=str, help='Path to embeddings file')
    data_group.add_argument('--embeddings_dir', type=str, help='Directory containing chunked embedding files')
    
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query_file', type=str, help='Path to query embeddings file')
    query_group.add_argument('--query_dir', type=str, help='Directory containing chunked query embedding files')
    
    # IDs (optional)
    parser.add_argument('--ids_file', type=str, help='Path to IDs file')
    parser.add_argument('--query_ids_file', type=str, help='Path to query IDs file')
    
    # Index options
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument('--existing_index_dir', type=str, help='Directory containing an existing LANNS index to evaluate')
    index_group.add_argument('--index_config_file', type=str, help='Path to a JSON file with index configurations to test')
    
    # LANNS parameters to evaluate (when building new indices)
    parser.add_argument('--num_shards', type=int, default=[1], nargs='+', help='Number of shards to test')
    parser.add_argument('--num_segments', type=int, default=[8], nargs='+', help='Number of segments per shard to test')
    parser.add_argument('--segmenters', type=str, default=['rs', 'rh', 'apd'], nargs='+', 
                        choices=['rs', 'rh', 'apd'], help='Segmentation strategies to test')
    parser.add_argument('--spill', type=float, default=0.15, help='Spill parameter')
    
    # HNSW parameters
    parser.add_argument('--hnsw_m', type=int, default=16, help='HNSW M parameter')
    parser.add_argument('--hnsw_ef_construction', type=int, default=200, help='HNSW efConstruction parameter')
    parser.add_argument('--hnsw_ef_search', type=int, default=[100], nargs='+', help='HNSW efSearch parameter values to test')
    parser.add_argument('--skip_build', action='store_true', help='Skip index building (use with --existing_index_dir)')
    
    # Evaluation parameters
    parser.add_argument('--k_values', type=int, default=[10, 50, 100], nargs='+', help='k values to evaluate')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers')
    parser.add_argument('--sample_size', type=int, default=10000, help='Number of query samples to use when creating query file')
    parser.add_argument('--create_sample_queries', action='store_true', help='Create a sample query file from the embeddings')
    parser.add_argument('--skip_exact_search', action='store_true', help='Skip exact search for ground truth')
    parser.add_argument('--ground_truth_file', type=str, help='Path to pre-computed ground truth file')
    
    return parser.parse_args()

def load_chunked_embeddings(dir_path: str, limit: Optional[int] = None) -> Tuple[Union[np.ndarray, List[str]], Union[List[Any], None]]:
    """
    Load embeddings from a directory of chunked numpy files.
    
    Args:
        dir_path: Path to directory containing embedding files
        limit: Optional limit on number of files to load
        
    Returns:
        embeddings: Combined numpy array of embeddings or list of file paths
        ids: List of IDs (if available) or None
    """
    logger.info(f"Loading chunked embeddings from {dir_path}")
    
    # Find all embedding files
    embedding_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy') and f.startswith('embeddings_')])
    
    if not embedding_files:
        raise ValueError(f"No embedding files found in {dir_path}")
    
    if limit and limit < len(embedding_files):
        logger.info(f"Limiting to first {limit} embedding files")
        embedding_files = embedding_files[:limit]
    
    # Load the first file to get dimensions
    first_file = os.path.join(dir_path, embedding_files[0])
    first_batch = np.load(first_file)
    dim = first_batch.shape[1]
    
    # Calculate total size
    total_entries = 0
    for file in embedding_files:
        file_path = os.path.join(dir_path, file)
        file_shape = np.load(file_path, mmap_mode='r').shape
        total_entries += file_shape[0]
    
    # If the total size is too large, return file paths instead for batch processing
    total_size_estimate = total_entries * dim * 4  # Rough estimate in bytes
    
    if total_size_estimate > 8 * 1024 * 1024 * 1024:  # If more than 8GB
        logger.info(f"Total estimated size ({total_size_estimate/(1024**3):.2f} GB) is too large to load at once.")
        logger.info(f"Will process in batches during evaluation.")
        return embedding_files, dir_path
    
    # Otherwise, load all embeddings
    all_embeddings = []
    all_ids = []
    
    for file in tqdm(embedding_files, desc="Loading embedding files"):
        file_path = os.path.join(dir_path, file)
        embeddings = np.load(file_path)
        all_embeddings.append(embeddings)
        
        # Try to load corresponding IDs file
        id_file = file.replace('embeddings_', 'ids_').replace('.npy', '.json')
        id_path = os.path.join(dir_path, id_file)
        if os.path.exists(id_path):
            with open(id_path, 'r') as f:
                batch_ids = json.load(f)
                all_ids.extend(batch_ids)
    
    # Combine all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    
    # If we didn't load any IDs, create sequential IDs
    if not all_ids:
        all_ids = list(range(len(combined_embeddings)))
    
    logger.info(f"Loaded {len(combined_embeddings)} embeddings with dimension {combined_embeddings.shape[1]}")
    
    return combined_embeddings, all_ids

def exact_search(query_embeddings: np.ndarray, index_embeddings: np.ndarray, k: int = 100, distance_metric: str = 'l2') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform exact nearest neighbor search (brute force).
    
    Args:
        query_embeddings: Query embeddings
        index_embeddings: Data embeddings to search in
        k: Number of nearest neighbors to return
        distance_metric: Distance metric ('l2', 'cosine', etc.)
        
    Returns:
        indices: Indices of nearest neighbors for each query
        distances: Distances to nearest neighbors for each query
    """
    logger.info(f"Performing exact search for {len(query_embeddings)} queries, k={k}")
    start_time = time.time()
    
    # Convert LANNS distance metric to scipy format
    scipy_metric = 'euclidean' if distance_metric == 'l2' else distance_metric
    
    results_indices = []
    results_distances = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = (len(query_embeddings) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Exact search"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(query_embeddings))
        batch_queries = query_embeddings[start_idx:end_idx]
        
        # Calculate distances
        dist_matrix = cdist(batch_queries, index_embeddings, metric=scipy_metric)
        
        # Get top-k indices and distances
        for i in range(len(batch_queries)):
            # Get indices of k smallest distances
            indices = np.argsort(dist_matrix[i])[:k]
            distances = dist_matrix[i][indices]
            
            results_indices.append(indices)
            results_distances.append(distances)
    
    logger.info(f"Exact search completed in {time.time() - start_time:.2f}s")
    
    return results_indices, results_distances

def exact_search_chunked(query_embeddings: np.ndarray, embedding_files: List[str], embeddings_dir: str, k: int = 100, distance_metric: str = 'l2') -> Tuple[List[List[int]], List[List[float]]]:
    """
    Perform exact nearest neighbor search (brute force) with chunked embeddings.
    
    Args:
        query_embeddings: Query embeddings
        embedding_files: List of embedding file names
        embeddings_dir: Directory containing embedding files
        k: Number of nearest neighbors to return
        distance_metric: Distance metric ('l2', 'cosine', etc.)
        
    Returns:
        indices: Indices of nearest neighbors for each query
        distances: Distances to nearest neighbors for each query
    """
    logger.info(f"Performing exact search with chunked data for {len(query_embeddings)} queries, k={k}")
    start_time = time.time()
    
    # Convert LANNS distance metric to scipy format
    scipy_metric = 'euclidean' if distance_metric == 'l2' else distance_metric
    
    # Initialize results with empty lists
    results_ids = [[] for _ in range(len(query_embeddings))]
    results_distances = [[] for _ in range(len(query_embeddings))]
    
    # Process embeddings in chunks
    global_offset = 0
    
    for file_name in tqdm(embedding_files, desc="Processing embedding chunks"):
        file_path = os.path.join(embeddings_dir, file_name)
        chunk_embeddings = np.load(file_path)
        chunk_size = len(chunk_embeddings)
        
        # Process queries in batches
        batch_size = 100
        num_batches = (len(query_embeddings) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(query_embeddings))
            batch_queries = query_embeddings[start_idx:end_idx]
            
            # Calculate distances
            dist_matrix = cdist(batch_queries, chunk_embeddings, metric=scipy_metric)
            
            # For each query in this batch
            for i, query_idx in enumerate(range(start_idx, end_idx)):
                # Get top-k indices and distances for this chunk
                sorted_indices = np.argsort(dist_matrix[i])
                top_indices = sorted_indices[:k]
                top_distances = dist_matrix[i][top_indices]
                
                # Convert to global indices
                global_indices = [idx + global_offset for idx in top_indices]
                
                # Extend current results
                results_ids[query_idx].extend(global_indices)
                results_distances[query_idx].extend(top_distances.tolist())
                
                # Keep only top-k overall
                if len(results_ids[query_idx]) > k:
                    combined = list(zip(results_distances[query_idx], results_ids[query_idx]))
                    combined.sort()  # Sort by distance
                    
                    # Extract only top-k
                    top_k_combined = combined[:k]
                    results_distances[query_idx] = [d for d, _ in top_k_combined]
                    results_ids[query_idx] = [idx for _, idx in top_k_combined]
        
        global_offset += chunk_size
    
    logger.info(f"Exact search completed in {time.time() - start_time:.2f}s")
    
    return results_ids, results_distances

def calculate_recall(lanns_results: List[List[Any]], exact_results: List[List[Any]], k: int) -> float:
    """
    Calculate recall@k between LANNS results and exact search results.
    
    Args:
        lanns_results: List of lists of IDs returned by LANNS
        exact_results: List of lists of indices returned by exact search
        k: k value for recall@k
        
    Returns:
        recall: Recall@k value (0-1)
    """
    if not lanns_results or not exact_results:
        return 0.0
    
    total_recall = 0.0
    count = 0
    
    for lanns_ids, exact_indices in zip(lanns_results, exact_results):
        # Convert to sets for intersection
        lanns_set = set(lanns_ids[:k])
        exact_set = set(exact_indices[:k])
        
        # Calculate recall for this query
        if exact_set:
            recall = len(lanns_set.intersection(exact_set)) / len(exact_set)
            total_recall += recall
            count += 1
    
    return total_recall / count if count > 0 else 0.0

def create_sample_queries(embeddings: Union[np.ndarray, List[str]], embeddings_dir: Optional[str], output_file: str, sample_size: int = 10000) -> np.ndarray:
    """
    Create a sample query set from embeddings.
    
    Args:
        embeddings: Embeddings array or list of embedding file paths
        embeddings_dir: Directory containing embedding files if embeddings is a list
        output_file: Path to save the sample query set
        sample_size: Number of queries to sample
        
    Returns:
        query_embeddings: Sample query embeddings
    """
    logger.info(f"Creating sample query set with {sample_size} queries")
    
    if isinstance(embeddings, np.ndarray):
        # Direct sampling from array
        indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
        query_embeddings = embeddings[indices]
    else:
        # Sample from chunked files
        all_sizes = []
        for file_name in embeddings:
            file_path = os.path.join(embeddings_dir, file_name)
            all_sizes.append(len(np.load(file_path, mmap_mode='r')))
        
        total_size = sum(all_sizes)
        cumulative_sizes = np.cumsum([0] + all_sizes)
        
        # Sample global indices
        global_indices = np.random.choice(total_size, min(sample_size, total_size), replace=False)
        global_indices.sort()  # Sort for more efficient loading
        
        # Map global indices to file and local indices
        query_embeddings = []
        
        current_file_idx = 0
        current_global_indices = []
        
        for global_idx in global_indices:
            # Find which file this index belongs to
            while global_idx >= cumulative_sizes[current_file_idx + 1]:
                # If we have accumulated indices for the current file, process them
                if current_global_indices:
                    file_path = os.path.join(embeddings_dir, embeddings[current_file_idx])
                    file_data = np.load(file_path)
                    
                    # Convert global indices to local indices for this file
                    local_indices = [idx - cumulative_sizes[current_file_idx] for idx in current_global_indices]
                    query_embeddings.append(file_data[local_indices])
                    
                    current_global_indices = []
                
                current_file_idx += 1
            
            current_global_indices.append(global_idx)
        
        # Process any remaining indices for the last file
        if current_global_indices:
            file_path = os.path.join(embeddings_dir, embeddings[current_file_idx])
            file_data = np.load(file_path)
            
            local_indices = [idx - cumulative_sizes[current_file_idx] for idx in current_global_indices]
            query_embeddings.append(file_data[local_indices])
        
        query_embeddings = np.vstack(query_embeddings)
    
    # Save sample queries
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    np.save(output_file, query_embeddings)
    logger.info(f"Saved {len(query_embeddings)} sample queries to {output_file}")
    
    return query_embeddings

def run_evaluation(args):
    """Main evaluation function"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    chunked_embeddings = False
    if args.embeddings_file:
        logger.info(f"Loading embeddings from {args.embeddings_file}")
        embeddings = np.load(args.embeddings_file)
        ids = list(range(len(embeddings)))
        embeddings_dir = None
    else:
        # Load chunked embeddings
        embeddings, ids = load_chunked_embeddings(args.embeddings_dir)
        chunked_embeddings = isinstance(embeddings, list)  # Check if we got file paths
        embeddings_dir = args.embeddings_dir
    
    # Load or create query embeddings
    if args.create_sample_queries:
        query_file = os.path.join(args.output_dir, "sample_queries.npy")
        query_embeddings = create_sample_queries(embeddings, embeddings_dir, query_file, args.sample_size)
    elif args.query_file:
        logger.info(f"Loading query embeddings from {args.query_file}")
        query_embeddings = np.load(args.query_file)
    else:
        # For queries, we'll always load them into memory
        query_embeddings, _ = load_chunked_embeddings(args.query_dir)
        if isinstance(query_embeddings, list):
            # If still too large, sample a subset
            query_file = os.path.join(args.output_dir, "query_sample.npy")
            query_embeddings = create_sample_queries(query_embeddings, args.query_dir, query_file, args.sample_size)
    
    # Load or compute ground truth
    exact_indices = {}
    exact_distances = {}
    
    if args.ground_truth_file and os.path.exists(args.ground_truth_file):
        logger.info(f"Loading ground truth from {args.ground_truth_file}")
        with open(args.ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
            
        for k_str, indices in ground_truth.items():
            k = int(k_str)
            exact_indices[k] = [[int(idx) for idx in query_indices] for query_indices in indices]
            
    elif not args.skip_exact_search:
        # Perform exact search for ground truth
        logger.info("Computing exact search results for ground truth...")
        
        for k in args.k_values:
            if chunked_embeddings:
                indices, distances = exact_search_chunked(query_embeddings, embeddings, embeddings_dir, k=k, distance_metric='l2')
            else:
                indices, distances = exact_search(query_embeddings, embeddings, k=k, distance_metric='l2')
            
            exact_indices[k] = indices
            exact_distances[k] = distances
            
        # Save ground truth for future use
        ground_truth_file = os.path.join(args.output_dir, 'ground_truth.json')
        with open(ground_truth_file, 'w') as f:
            # Convert numpy arrays to lists
            json_data = {k: [[int(idx) for idx in query_indices] for query_indices in indices] 
                       for k, indices in exact_indices.items()}
            json.dump(json_data, f)
            
        logger.info(f"Saved ground truth to {ground_truth_file}")
            
    else:
        logger.warning("Skipping exact search. Recall metrics will not be available.")
    
    # Prepare results dictionary
    results = {
        'configurations': [],
        'metrics': {},
        'parameters': vars(args)
    }
    
    # Determine configurations to evaluate
    configurations = []
    
    # Option 1: Evaluate an existing index directory
    if args.existing_index_dir:
        logger.info(f"Evaluating existing index at {args.existing_index_dir}")
        
        # Read the metadata to get configuration details
        metadata_file = os.path.join(args.existing_index_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Create a single configuration from the metadata
            configurations.append({
                'index_dir': args.existing_index_dir,
                'num_shards': metadata.get('num_shards', 1),
                'num_segments': metadata.get('num_segments', 8),
                'segmenter_type': metadata.get('segmenter_type', 'apd'),
                'ef_search': args.hnsw_ef_search[0],  # Use the first ef_search value
                'skip_build': True  # Skip building
            })
            
            # If multiple ef_search values specified, create additional configs
            if len(args.hnsw_ef_search) > 1:
                for ef in args.hnsw_ef_search[1:]:
                    configurations.append({
                        'index_dir': args.existing_index_dir,
                        'num_shards': metadata.get('num_shards', 1),
                        'num_segments': metadata.get('num_segments', 8),
                        'segmenter_type': metadata.get('segmenter_type', 'apd'),
                        'ef_search': ef,
                        'skip_build': True  # Skip building
                    })
        else:
            logger.warning(f"No metadata.json found in {args.existing_index_dir}. Using default values.")
            # Create a configuration with default values
            for ef in args.hnsw_ef_search:
                configurations.append({
                    'index_dir': args.existing_index_dir,
                    'num_shards': 1,
                    'num_segments': 8,
                    'segmenter_type': 'apd',
                    'ef_search': ef,
                    'skip_build': True  # Skip building
                })
    
    # Option 2: Read configurations from a JSON file
    elif args.index_config_file:
        logger.info(f"Loading index configurations from {args.index_config_file}")
        with open(args.index_config_file, 'r') as f:
            config_data = json.load(f)
            
        # Add each configuration from the file
        for config in config_data:
            configurations.append(config)
    
    # Option 3: Generate configurations from parameters
    else:
        total_configs = len(args.num_shards) * len(args.num_segments) * len(args.segmenters) * len(args.hnsw_ef_search)
        logger.info(f"Generating {total_configs} index configurations...")
        
        for num_shards in args.num_shards:
            for num_segments in args.num_segments:
                for segmenter_type in args.segmenters:
                    for ef_search in args.hnsw_ef_search:
                        config_name = f"s{num_shards}_g{num_segments}_{segmenter_type}_ef{ef_search}"
                        config = {
                            'num_shards': num_shards,
                            'num_segments': num_segments,
                            'segmenter_type': segmenter_type,
                            'ef_search': ef_search,
                            'index_dir': os.path.join(args.output_dir, f"index_{config_name}"),
                            'skip_build': args.skip_build
                        }
                        configurations.append(config)
    
    # Evaluate each configuration
    logger.info(f"Evaluating {len(configurations)} configurations...")
    
    for config_idx, config in enumerate(configurations):
        num_shards = config['num_shards']
        num_segments = config['num_segments']
        segmenter_type = config['segmenter_type']
        ef_search = config['ef_search']
        index_dir = config['index_dir']
        skip_build = config.get('skip_build', False)
        
        config_name = f"s{num_shards}_g{num_segments}_{segmenter_type}_ef{ef_search}"
        logger.info(f"Configuration {config_idx+1}/{len(configurations)}: {config_name}")
        
        # Create index directory if needed
        os.makedirs(index_dir, exist_ok=True)
        
        # Build index if not skipping build step
        build_time = 0
        if not skip_build:
            logger.info(f"Building index...")
            builder = LANNSIndexBuilder(
                num_shards=num_shards,
                num_segments=num_segments,
                segmenter_type=segmenter_type,
                spill=args.spill,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                max_workers=args.max_workers
            )
            
            start_time = time.time()
            if chunked_embeddings:
                # Use build_from_files instead
                builder.build_from_files(
                    embeddings_dir=args.embeddings_dir,
                    output_dir=index_dir,
                    batch_size=10000
                )
            else:
                builder.build(embeddings, ids, index_dir)
            build_time = time.time() - start_time
        else:
            logger.info(f"Skipping index build (using existing index)")
        
        # Load index for querying
        logger.info(f"Loading index for querying...")
        index = LANNSIndex(index_dir, ef_search=ef_search)
        
        # Query metrics
        query_metrics = {}
        
        for k in args.k_values:
            logger.info(f"Querying with k={k}...")
            
            # Perform queries
            start_time = time.time()
            results_ids, results_distances = index.batch_query(query_embeddings, k=k)
            query_time = time.time() - start_time
            
            # Calculate recall if ground truth is available
            recall = 0.0
            if k in exact_indices and exact_indices[k]:
                recall = calculate_recall(results_ids, exact_indices[k], k)
            
            # Calculate QPS
            qps = len(query_embeddings) / query_time
            
            query_metrics[f"k{k}"] = {
                "recall": recall,
                "query_time_ms": query_time * 1000 / len(query_embeddings),
                "qps": qps
            }
        
        # Store configuration results
        config_result = {
            "name": config_name,
            "num_shards": num_shards,
            "num_segments": num_segments,
            "segmenter_type": segmenter_type,
            "ef_search": ef_search,
            "index_dir": index_dir,
            "build_time_s": build_time,
            "build_time_per_point_ms": build_time * 1000 / (len(embeddings) if not chunked_embeddings else 1000000) if build_time > 0 else 0,
            "query_metrics": query_metrics
        }
        
        results['configurations'].append(config_result)
        
        # Save intermediate results
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Generate summary metrics
    for config in results['configurations']:
        for k_value in args.k_values:
            k_key = f"k{k_value}"
            if k_key not in results['metrics']:
                results['metrics'][k_key] = {
                    "best_recall": {
                        "value": 0.0,
                        "config": None
                    },
                    "best_qps": {
                        "value": 0.0,
                        "config": None
                    }
                }
            
            # Check if this config has better recall
            recall = config['query_metrics'][k_key]['recall']
            if recall > results['metrics'][k_key]['best_recall']['value']:
                results['metrics'][k_key]['best_recall']['value'] = recall
                results['metrics'][k_key]['best_recall']['config'] = config['name']
            
            # Check if this config has better QPS
            qps = config['query_metrics'][k_key]['qps']
            if qps > results['metrics'][k_key]['best_qps']['value']:
                results['metrics'][k_key]['best_qps']['value'] = qps
                results['metrics'][k_key]['best_qps']['config'] = config['name']
    
    # Save final results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}/results.json")
    
    # Generate plots if requested
    if args.plot:
        generate_plots(results, args.output_dir)
    
    return results

def generate_plots(results, output_dir):
    """Generate performance plots from evaluation results"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group configurations by segmenter type
    segmenter_groups = {}
    for config in results['configurations']:
        segmenter = config['segmenter_type']
        if segmenter not in segmenter_groups:
            segmenter_groups[segmenter] = []
        segmenter_groups[segmenter].append(config)
    
    # Extract k values
    k_values = []
    if results['configurations']:
        k_values = [int(k[1:]) for k in results['configurations'][0]['query_metrics'].keys()]
    
    # Plot recall vs QPS for each k value
    for k in k_values:
        plt.figure(figsize=(10, 6))
        
        for segmenter, configs in segmenter_groups.items():
            recalls = []
            qps_values = []
            labels = []
            
            for config in configs:
                if f'k{k}' in config['query_metrics']:
                    recalls.append(config['query_metrics'][f'k{k}']['recall'])
                    qps_values.append(config['query_metrics'][f'k{k}']['qps'])
                    labels.append(f"s{config['num_shards']}_g{config['num_segments']}_ef{config['ef_search']}")
            
            if recalls:  # Only plot if we have data
                plt.scatter(recalls, qps_values, label=segmenter, alpha=0.7)
                
                # Add labels for each point
                for i, label in enumerate(labels):
                    plt.annotate(label, (recalls[i], qps_values[i]), fontsize=8)
        
        plt.xlabel('Recall')
        plt.ylabel('QPS')
        plt.title(f'Recall vs QPS for k={k}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(plot_dir, f'recall_vs_qps_k{k}.png'), dpi=300)
        plt.close()
    
    # Plot build time vs num_segments for each segmenter and num_shards
    for num_shards in set(config['num_shards'] for config in results['configurations']):
        plt.figure(figsize=(10, 6))
        
        for segmenter, configs in segmenter_groups.items():
            # Filter configs by num_shards
            filtered_configs = [c for c in configs if c['num_shards'] == num_shards]
            
            if not filtered_configs:
                continue
            
            # Group by num_segments
            segment_groups = {}
            for config in filtered_configs:
                segments = config['num_segments']
                if segments not in segment_groups:
                    segment_groups[segments] = []
                segment_groups[segments].append(config)
            
            # Calculate average build time for each num_segments
            segments = []
            build_times = []
            
            for seg, configs in segment_groups.items():
                segments.append(seg)
                avg_time = sum(c['build_time_s'] for c in configs) / len(configs)
                build_times.append(avg_time)
            
            # Sort by segments
            if segments:
                sorted_indices = np.argsort(segments)
                segments = [segments[i] for i in sorted_indices]
                build_times = [build_times[i] for i in sorted_indices]
                
                plt.plot(segments, build_times, marker='o', label=segmenter)
        
        plt.xlabel('Number of Segments')
        plt.ylabel('Build Time (s)')
        plt.title(f'Build Time vs Number of Segments (Shards={num_shards})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(plot_dir, f'build_time_vs_segments_s{num_shards}.png'), dpi=300)
        plt.close()
    
    # Plot query time vs num_segments
    for num_shards in set(config['num_shards'] for config in results['configurations']):
        plt.figure(figsize=(10, 6))
        
        for k in k_values:
            plt.figure(figsize=(10, 6))
            
            for segmenter, configs in segmenter_groups.items():
                # Filter configs by num_shards
                filtered_configs = [c for c in configs if c['num_shards'] == num_shards]
                
                if not filtered_configs:
                    continue
                
                # Group by num_segments
                segment_groups = {}
                for config in filtered_configs:
                    segments = config['num_segments']
                    if segments not in segment_groups:
                        segment_groups[segments] = []
                    segment_groups[segments].append(config)
                
                # Calculate average query time for each num_segments
                segments = []
                query_times = []
                
                for seg, configs in segment_groups.items():
                    segments.append(seg)
                    avg_time = sum(c['query_metrics'][f'k{k}']['query_time_ms'] for c in configs if f'k{k}' in c['query_metrics']) / len(configs)
                    query_times.append(avg_time)
                
                # Sort by segments
                if segments:
                    sorted_indices = np.argsort(segments)
                    segments = [segments[i] for i in sorted_indices]
                    query_times = [query_times[i] for i in sorted_indices]
                    
                    plt.plot(segments, query_times, marker='o', label=f"{segmenter}")
            
            plt.xlabel('Number of Segments')
            plt.ylabel('Query Time (ms)')
            plt.title(f'Query Time vs Number of Segments (Shards={num_shards}, k={k})')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(plot_dir, f'query_time_vs_segments_s{num_shards}_k{k}.png'), dpi=300)
            plt.close()
    
    logger.info(f"Performance plots saved to {plot_dir}")

def main():
    args = parse_args()
    run_evaluation(args)

if __name__ == '__main__':
    main()