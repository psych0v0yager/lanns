#!/usr/bin/env python
"""
query_index.py - Script to query a LANNS index

Usage:
    python query_index.py --index_dir ./lanns_index --query_file query_embeddings.npy --k 10
    python query_index.py --index_dir ./lanns_index --query_h5 queries.h5 --k 10
    python query_index.py --index_dir ./lanns_index --input_text "some text to embed" --k 10
"""
import os
import argparse
import json
import time
import numpy as np
import h5py
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

from lanns.indexing.storage import LANNSIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Query a LANNS index')
    
    # Index options
    parser.add_argument('--index_dir', type=str, required=True, help='Directory containing the LANNS index')
    
    # Query options
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query_file', type=str, help='Path to a numpy file containing query embeddings')
    query_group.add_argument('--query_h5', type=str, help='Path to an HDF5 file containing query embeddings')
    query_group.add_argument('--input_text', type=str, help='Text to embed and use as query')
    
    # Query parameters
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors to return')
    parser.add_argument('--ef_search', type=int, default=100, help='HNSW efSearch parameter')
    
    # Output options
    parser.add_argument('--output_file', type=str, help='Path to save query results (JSON)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing queries')
    
    # Model options for text input
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', 
                       help='Model to use for embedding input text')
    
    return parser.parse_args()

def embed_text(texts, model_name):
    """
    Embed text using a sentence transformer model.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the model to use
        
    Returns:
        embeddings: Array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts)
    except ImportError:
        logger.error("sentence-transformers is required for text embedding. Install with: pip install sentence-transformers")
        raise

def process_results(ids, distances, query_ids=None):
    """
    Process query results into a structured format.
    
    Args:
        ids: List of lists of IDs returned for each query
        distances: List of lists of distances for each query
        query_ids: Optional list of query IDs
        
    Returns:
        results: Structured results for each query
    """
    results = []
    
    for i, (query_neighbors, query_distances) in enumerate(zip(ids, distances)):
        query_id = query_ids[i] if query_ids else i
        
        # Create structured result
        neighbors = []
        for j, (neighbor_id, distance) in enumerate(zip(query_neighbors, query_distances)):
            neighbors.append({
                'id': neighbor_id,
                'distance': float(distance),
                'rank': j+1
            })
        
        results.append({
            'query_id': query_id,
            'neighbors': neighbors
        })
    
    return results

def main():
    """Main function to query a LANNS index"""
    args = parse_args()
    
    # Load the index
    logger.info(f"Loading LANNS index from {args.index_dir}")
    start_time = time.time()
    index = LANNSIndex(args.index_dir, ef_search=args.ef_search)
    logger.info(f"Index loaded in {time.time() - start_time:.2f}s")
    
    # Load query embeddings
    query_embeddings = None
    query_ids = None
    
    if args.query_file:
        logger.info(f"Loading query embeddings from {args.query_file}")
        query_embeddings = np.load(args.query_file)
        
        # Check if we have query IDs
        ids_file = os.path.splitext(args.query_file)[0] + '_ids.json'
        if os.path.exists(ids_file):
            with open(ids_file, 'r') as f:
                query_ids = json.load(f)
    
    elif args.query_h5:
        logger.info(f"Loading query embeddings from {args.query_h5}")
        with h5py.File(args.query_h5, 'r') as f:
            if 'embeddings' not in f:
                raise ValueError(f"HDF5 file {args.query_h5} does not contain 'embeddings' dataset")
            
            query_embeddings = f['embeddings'][:]
            
            if 'ids' in f:
                query_ids = f['ids'][:]
    
    elif args.input_text:
        logger.info(f"Embedding input text using {args.model_name}")
        query_embeddings = embed_text([args.input_text], args.model_name)
    
    # Perform queries
    logger.info(f"Querying index with {len(query_embeddings)} queries, k={args.k}")
    query_start = time.time()
    
    if len(query_embeddings) <= args.batch_size:
        # Small number of queries - process all at once
        results_ids, results_distances = index.batch_query(
            query_embeddings,
            k=args.k,
            ef_search=args.ef_search
        )
    else:
        # Large number of queries - process in batches
        logger.info(f"Processing queries in batches of {args.batch_size}")
        results_ids = []
        results_distances = []
        
        num_batches = (len(query_embeddings) + args.batch_size - 1) // args.batch_size
        
        with tqdm(total=num_batches, desc="Processing query batches") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(query_embeddings))
                
                batch_embeddings = query_embeddings[start_idx:end_idx]
                
                batch_ids, batch_distances = index.batch_query(
                    batch_embeddings,
                    k=args.k,
                    ef_search=args.ef_search
                )
                
                results_ids.extend(batch_ids)
                results_distances.extend(batch_distances)
                
                pbar.update(1)
    
    query_time = time.time() - query_start
    logger.info(f"Queries completed in {query_time:.2f}s")
    logger.info(f"Average query time: {query_time / len(query_embeddings) * 1000:.2f}ms per query")
    logger.info(f"QPS: {len(query_embeddings) / query_time:.2f}")
    
    # Process results
    results = process_results(results_ids, results_distances, query_ids)
    
    # Save results if output file specified
    if args.output_file:
        logger.info(f"Saving results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Print a few results
        print("\nSample results:")
        for i, result in enumerate(results[:3]):
            print(f"Query {result['query_id']}:")
            for neighbor in result['neighbors'][:5]:
                print(f"  Rank {neighbor['rank']}: ID {neighbor['id']}, Distance {neighbor['distance']:.6f}")
            print()
        
        if len(results) > 3:
            print(f"... and {len(results) - 3} more queries")

if __name__ == '__main__':
    main()