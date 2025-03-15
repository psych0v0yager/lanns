#!/usr/bin/env python
"""
evaluate.py - Script to evaluate LANNS performance

This script measures:
1. Build time
2. Query time / QPS
3. Recall compared to exact search

Usage:
    python evaluate.py --embeddings_file all_embeddings.npy --query_file query_embeddings.npy --output_dir ./eval_results
"""
import os
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging
from scipy.spatial.distance import cdist
from tqdm import tqdm

from lanns.indexing.builder import LANNSIndexBuilder
from lanns.indexing.storage import LANNSIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LANNS performance')
    
    # Data
    parser.add_argument('--embeddings_file', type=str, required=True, help='Path to embeddings file')
    parser.add_argument('--query_file', type=str, required=True, help='Path to query embeddings file')
    
    # IDs (optional)
    parser.add_argument('--ids_file', type=str, help='Path to IDs file')
    parser.add_argument('--query_ids_file', type=str, help='Path to query IDs file')
    
    # LANNS parameters to evaluate
    parser.add_argument('--num_shards', type=int, default=[1], nargs='+', help='Number of shards to test')
    parser.add_argument('--num_segments', type=int, default=[8], nargs='+', help='Number of segments per shard to test')
    parser.add_argument('--segmenters', type=str, default=['rs', 'rh', 'apd'], nargs='+', 
                        choices=['rs', 'rh', 'apd'], help='Segmentation strategies to test')
    parser.add_argument('--spill', type=float, default=0.15, help='Spill parameter')
    
    # HNSW parameters
    parser.add_argument('--hnsw_m', type=int, default=16, help='HNSW M parameter')
    parser.add_argument('--hnsw_ef_construction', type=int, default=200, help='HNSW efConstruction parameter')
    parser.add_argument('--hnsw_ef_search', type=int, default=[100], nargs='+', help='HNSW efSearch parameter values to test')
    
    # Evaluation parameters
    parser.add_argument('--k_values', type=int, default=[10, 50, 100], nargs='+', help='k values to evaluate')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers')
    
    return parser.parse_args()

def exact_search(query_embeddings, index_embeddings, k=100, distance_metric='l2'):
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
        dist_matrix = cdist(batch_queries, index_embeddings, metric=distance_metric)
        
        # Get top-k indices and distances
        for i in range(len(batch_queries)):
            # Get indices of k smallest distances
            indices = np.argsort(dist_matrix[i])[:k]
            distances = dist_matrix[i][indices]
            
            results_indices.append(indices)
            results_distances.append(distances)
    
    logger.info(f"Exact search completed in {time.time() - start_time:.2f}s")
    
    return results_indices, results_distances

def calculate_recall(lanns_results, exact_results, k):
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

def run_evaluation(args):
    """Main evaluation function"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading embeddings from {args.embeddings_file}")
    embeddings = np.load(args.embeddings_file)
    
    logger.info(f"Loading query embeddings from {args.query_file}")
    query_embeddings = np.load(args.query_file)
    
    # Load IDs if provided
    ids = None
    if args.ids_file:
        logger.info(f"Loading IDs from {args.ids_file}")
        with open(args.ids_file, 'r') as f:
            ids = json.load(f)
    else:
        ids = list(range(len(embeddings)))
    
    # Perform exact search for ground truth
    logger.info("Computing exact search results for ground truth...")
    exact_indices = {}
    exact_distances = {}
    
    for k in args.k_values:
        indices, distances = exact_search(query_embeddings, embeddings, k=k, distance_metric='l2')
        exact_indices[k] = indices
        exact_distances[k] = distances
    
    # Prepare results dictionary
    results = {
        'configurations': [],
        'metrics': {},
        'parameters': vars(args)
    }
    
    # Test different configurations
    total_configs = len(args.num_shards) * len(args.num_segments) * len(args.segmenters) * len(args.hnsw_ef_search)
    logger.info(f"Evaluating {total_configs} configurations...")
    
    config_idx = 0
    for num_shards in args.num_shards:
        for num_segments in args.num_segments:
            for segmenter_type in args.segmenters:
                for ef_search in args.hnsw_ef_search:
                    config_idx += 1
                    config_name = f"s{num_shards}_g{num_segments}_{segmenter_type}_ef{ef_search}"
                    logger.info(f"Configuration {config_idx}/{total_configs}: {config_name}")
                    
                    # Create index directory
                    index_dir = os.path.join(args.output_dir, f"index_{config_name}")
                    os.makedirs(index_dir, exist_ok=True)
                    
                    # Build index
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
                    builder.build(embeddings, ids, index_dir)
                    build_time = time.time() - start_time
                    
                    # Load index for querying
                    index = LANNSIndex(index_dir, ef_search=ef_search)
                    
                    # Query metrics
                    query_metrics = {}
                    
                    for k in args.k_values:
                        logger.info(f"Querying with k={k}...")
                        
                        # Perform queries
                        start_time = time.time()
                        results_ids, results_distances = index.batch_query(query_embeddings, k=k)
                        query_time = time.time() - start_time
                        
                        # Calculate recall
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
                        "build_time_s": build_time,
                        "build_time_per_point_ms": build_time * 1000 / len(embeddings),
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
    k_values = [int(k[1:]) for k in list(results['configurations'][0]['query_metrics'].keys())]
    
    # Plot recall vs QPS for each k value
    for k in k_values:
        plt.figure(figsize=(10, 6))
        
        for segmenter, configs in segmenter_groups.items():
            recalls = []
            qps_values = []
            labels = []
            
            for config in configs:
                recalls.append(config['query_metrics'][f'k{k}']['recall'])
                qps_values.append(config['query_metrics'][f'k{k}']['qps'])
                labels.append(f"s{config['num_shards']}_g{config['num_segments']}_ef{config['ef_search']}")
            
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
    
    logger.info(f"Performance plots saved to {plot_dir}")

def main():
    args = parse_args()
    run_evaluation(args)

if __name__ == '__main__':
    main()