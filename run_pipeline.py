#!/usr/bin/env python
"""
run_pipeline.py - End-to-end pipeline for LANNS

This script orchestrates the full LANNS workflow:
1. Generate embeddings from data
2. (Optional) Enhance embeddings using vLLM
3. Build a LANNS index
4. Optionally create sample queries

Usage:
    python run_pipeline.py --data_file input.ndjson --output_dir ./lanns_project
"""
import os
import argparse
import subprocess
import time
import json
import random
import numpy as np
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Run the complete LANNS pipeline')
    
    # Input/output options
    parser.add_argument('--data_file', type=str, required=True, help='Path to the raw data file')
    parser.add_argument('--output_dir', type=str, default='./lanns_project', help='Base output directory')
    
    # Embedding options
    parser.add_argument('--model_name', type=str, default='Snowflake/snowflake-arctic-embed-l-v2.0', 
                       help='Embedding model to use')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for embeddings')
    parser.add_argument('--use_fp16', action='store_true', help='Use half precision (FP16)')
    parser.add_argument('--use_mrl', action='store_true', help='Use MRL to compress embeddings')
    parser.add_argument('--mrl_dimensions', type=int, default=256, help='Dimensions for MRL')
    
    # Enhanced embeddings options
    parser.add_argument('--use_enhanced_embeddings', action='store_true', 
                       help='Enable vLLM-enhanced embeddings')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                       help='vLLM model for enhanced profiles')
    
    # LANNS options
    parser.add_argument('--num_shards', type=int, default=2, help='Number of shards for LANNS')
    parser.add_argument('--num_segments', type=int, default=16, help='Number of segments per shard')
    parser.add_argument('--segmenter', type=str, default='apd', choices=['rs', 'rh', 'apd'], 
                       help='Segmentation strategy')
    parser.add_argument('--spill', type=float, default=0.15, help='Spill parameter')
    
    # Query options
    parser.add_argument('--create_query_samples', action='store_true', 
                       help='Create sample query embeddings for testing')
    parser.add_argument('--query_sample_size', type=int, default=10,
                       help='Number of sample queries to generate')
    
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command with helpful output"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        exit(result.returncode)
        
    duration = time.time() - start_time
    print(f"Completed in {duration:.2f} seconds")
    
    return duration

def main():
    args = parse_args()
    start_time = time.time()
    
    # Create directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    index_dir = os.path.join(args.output_dir, "lanns_index")
    query_dir = os.path.join(args.output_dir, "query_samples")
    
    # Determine which embeddings directory to use for indexing
    if args.use_enhanced_embeddings:
        embeddings_for_index = os.path.join(embeddings_dir, "combined_embeddings")
    else:
        embeddings_for_index = embeddings_dir
    
    print("\n=== LANNS Pipeline ===")
    print(f"Input data: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    
    if args.use_enhanced_embeddings:
        print(f"Enhanced embeddings: Enabled (model: {args.llm_model})")
    
    print(f"LANNS config: {args.num_shards} shards, {args.num_segments} segments, {args.segmenter} segmenter")
    
    # Step 1: Generate Embeddings
    embedding_cmd = [
        "python", "generate_embeddings.py",
        "--data_path", args.data_file,
        "--output_dir", embeddings_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size)
    ]
    
    if args.use_fp16:
        embedding_cmd.append("--use_fp16")
        
    if args.use_mrl:
        embedding_cmd.append("--use_mrl")
        embedding_cmd.extend(["--mrl_dimensions", str(args.mrl_dimensions)])
        
    if args.use_enhanced_embeddings:
        embedding_cmd.append("--use_enhanced_embeddings")
        embedding_cmd.extend(["--llm_model", args.llm_model])
    
    run_command(embedding_cmd, "Step 1: Generating Embeddings")
    
    # Step 2: Build LANNS Index
    index_cmd = [
        "python", "build_index.py",
        "--embeddings_dir", embeddings_for_index,
        "--output_dir", index_dir,
        "--num_shards", str(args.num_shards),
        "--num_segments", str(args.num_segments),
        "--segmenter", args.segmenter,
        "--spill", str(args.spill)
    ]
    
    run_command(index_cmd, "Step 2: Building LANNS Index")
    
    # Step 3: Create sample queries if requested
    if args.create_query_samples:
        os.makedirs(query_dir, exist_ok=True)
        
        # Get a list of embedding files
        embedding_files = [f for f in os.listdir(embeddings_for_index) 
                         if f.endswith('.npy') and f.startswith('embeddings_')]
        
        if embedding_files:
            print("\n=== Step 3: Creating Sample Queries ===")
            
            # Select a random file
            random_file = random.choice(embedding_files)
            embeddings = np.load(os.path.join(embeddings_for_index, random_file))
            
            # Select random samples
            if len(embeddings) > args.query_sample_size:
                indices = random.sample(range(len(embeddings)), args.query_sample_size)
                query_embeddings = embeddings[indices]
            else:
                query_embeddings = embeddings[:args.query_sample_size]
                
            # Save as query file
            query_file = os.path.join(query_dir, f"queries_{args.query_sample_size}.npy")
            np.save(query_file, query_embeddings)
            print(f"Created {args.query_sample_size} sample queries in {query_file}")
            
            # Run a test query
            test_cmd = [
                "python", "query_index.py",
                "--index_dir", index_dir,
                "--query_file", query_file,
                "--k", "10"
            ]
            
            run_command(test_cmd, "Running test query")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n=== Pipeline Complete ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Embeddings: {embeddings_dir}")
    print(f"LANNS index: {index_dir}")
    
    if args.create_query_samples:
        print(f"Query samples: {query_dir}")
    
    print("\nYou can now query the index with:")
    print(f"  python query_index.py --index_dir {index_dir} --query_file /path/to/query_embeddings.npy --k 10")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {str(e)}")