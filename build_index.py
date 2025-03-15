#!/usr/bin/env python
"""
build_index.py - Script to build a LANNS index from embeddings

Usage:
    python build_index.py --embeddings_dir ./embeddings --output_dir ./lanns_index
    python build_index.py --embeddings_file all_embeddings.npy --ids_file all_ids.json --output_dir ./lanns_index
    python build_index.py --embeddings_h5 embeddings.h5 --output_dir ./lanns_index
"""
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path
import h5py
import logging

from lanns.indexing.builder import LANNSIndexBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Build a LANNS index from embeddings')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--embeddings_dir', type=str, help='Directory containing embedding files')
    input_group.add_argument('--embeddings_file', type=str, help='Path to a single embeddings numpy file')
    input_group.add_argument('--embeddings_h5', type=str, help='Path to an HDF5 file containing embeddings')
    
    # ID mapping
    parser.add_argument('--ids_file', type=str, help='Path to a JSON file containing IDs')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='lanns_index', help='Directory to save the index')
    
    # LANNS parameters
    parser.add_argument('--num_shards', type=int, default=1, help='Number of shards')
    parser.add_argument('--num_segments', type=int, default=8, help='Number of segments per shard')
    parser.add_argument('--segmenter', type=str, default='apd', choices=['rs', 'rh', 'apd'], help='Segmentation strategy')
    parser.add_argument('--spill', type=float, default=0.15, help='Spill parameter')
    parser.add_argument('--space', type=str, default='l2', choices=['l2', 'cosine', 'ip'], help='Distance metric')
    
    # HNSW parameters
    parser.add_argument('--hnsw_m', type=int, default=16, help='HNSW M parameter')
    parser.add_argument('--hnsw_ef_construction', type=int, default=200, help='HNSW efConstruction parameter')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size for processing')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers')
    
    return parser.parse_args()

def main():
    """Main function to build a LANNS index"""
    args = parse_args()
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create index builder
    builder = LANNSIndexBuilder(
        num_shards=args.num_shards,
        num_segments=args.num_segments,
        segmenter_type=args.segmenter,
        spill=args.spill,
        space=args.space,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        max_workers=args.max_workers
    )
    
    # Build index based on input type
    if args.embeddings_dir:
        # Build from directory
        builder.build_from_files(
            embeddings_dir=args.embeddings_dir,
            ids_file=args.ids_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    elif args.embeddings_file:
        # Build from single file
        # Load embeddings
        logger.info(f"Loading embeddings from {args.embeddings_file}")
        embeddings = np.load(args.embeddings_file)
        
        # Load IDs if provided
        if args.ids_file:
            logger.info(f"Loading IDs from {args.ids_file}")
            with open(args.ids_file, 'r') as f:
                ids = json.load(f)
        else:
            # Generate sequential IDs
            ids = list(range(len(embeddings)))
        
        # Build index
        builder.build(embeddings, ids, args.output_dir)
    elif args.embeddings_h5:
        # Build from HDF5 file
        builder.build_from_files(
            embeddings_dir=args.embeddings_h5,
            ids_file=args.ids_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    
    total_time = time.time() - start_time
    logger.info(f"Total build time: {total_time:.2f}s")

if __name__ == '__main__':
    main()