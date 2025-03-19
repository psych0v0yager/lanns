#!/usr/bin/env python
"""
generate_embeddings.py - Generate embeddings for LANNS

This script provides a command-line interface to generate embeddings
using the LANNS embedding modules. It supports both raw embeddings
and optionally enhanced embeddings using vLLM.

Usage:
    python generate_embeddings.py --data_path input.ndjson --output_dir ./embeddings
"""
import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import gc
from tqdm import tqdm

from lanns.embeddings import setup_logging, get_processor, EmbeddingGenerator
from lanns.embeddings import vllm_available

# Only import enhancer if vLLM is available
if vllm_available:
    from lanns.embeddings import VLLMEnhancer, combine_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings for LANNS')
    
    # Input/output options
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file or directory')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Directory to save embeddings')
    
    # Data processing options
    parser.add_argument('--format', type=str, choices=['json', 'csv', 'ndjson', 'text'], 
                       help='Format of the input data (auto-detected if not specified)')
    parser.add_argument('--text_field', type=str, default=None, 
                       help='For JSON/NDJSON files, the field containing the text to embed')
    parser.add_argument('--id_field', type=str, default='ROW_ID',
                       help='Field to use as ID in the output')
    parser.add_argument('--fields_order', type=str, default=None,
                       help='Comma-separated list of fields to prioritize (e.g. "FIRST_NAME,LAST_NAME,EMAIL")')
    parser.add_argument('--prompt_prefix', type=str, default=None,
                       help='Optional prefix to add to each text before embedding')
    
    # Embedding options
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    parser.add_argument('--model_name', type=str, default='Snowflake/snowflake-arctic-embed-l-v2.0', 
                       help='Sentence transformer model name')
    parser.add_argument('--use_fp16', action='store_true', help='Use half precision (FP16)')
    parser.add_argument('--use_mrl', action='store_true', help='Use MRL to compress embeddings')
    parser.add_argument('--mrl_dimensions', type=int, default=256, help='Dimensions to keep when using MRL')
    
    # Processing options
    parser.add_argument('--checkpoint_frequency', type=int, default=100, 
                       help='Save checkpoint after this many batches')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                       help='Resume from a checkpoint file')
    
    # Enhanced embeddings options (vLLM)
    parser.add_argument('--use_enhanced_embeddings', action='store_true', 
                       help='Enable vLLM-enhanced embeddings (requires vLLM)')
    parser.add_argument('--llm_model', type=str, default='mistralai/Mistral-Small-3.1-24B-Instruct-2503', 
                       help='vLLM model for enhanced profiles')
    parser.add_argument('--fusion_method', type=str, default='concatenate', 
                       choices=['concatenate', 'weighted_average'],
                       help='Method to combine raw and enhanced embeddings')
    parser.add_argument('--raw_weight', type=float, default=0.5,
                       help='Weight for raw embeddings when using weighted_average (0-1)')
    
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    
    # Setup logging
    setup_logging(os.path.join(args.output_dir, "embedding_generation.log"))
    logger = logging.getLogger(__name__)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    raw_embeddings_dir = args.output_dir
    if args.use_enhanced_embeddings:
        if not vllm_available:
            logger.error("Enhanced embeddings requested but vLLM is not available. Install with: pip install vllm")
            return
            
        # Create separate directories for raw, enhanced, and combined embeddings
        raw_embeddings_dir = os.path.join(args.output_dir, "raw_embeddings")
        enhanced_embeddings_dir = os.path.join(args.output_dir, "enhanced_embeddings")
        profiles_dir = os.path.join(args.output_dir, "profiles")
        combined_embeddings_dir = os.path.join(args.output_dir, "combined_embeddings")
        
        os.makedirs(raw_embeddings_dir, exist_ok=True)
        os.makedirs(enhanced_embeddings_dir, exist_ok=True)
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(combined_embeddings_dir, exist_ok=True)
    
    # Parse fields_order if provided
    fields_order = args.fields_order.split(',') if args.fields_order else None
    
    # Log configuration
    logger.info("\n=== LANNS Embedding Generation ===")
    logger.info(f"Input data: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Embedding model: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"MRL compression: {'Enabled - ' + str(args.mrl_dimensions) + ' dimensions' if args.use_mrl else 'Disabled'}")
    logger.info(f"FP16 precision: {'Enabled' if args.use_fp16 else 'Disabled'}")
    
    if args.text_field:
        logger.info(f"Text field: '{args.text_field}'")
    else:
        logger.info(f"Using formatted record (no specific text field)")
        
    if fields_order:
        logger.info(f"Fields order: {args.fields_order}")
        
    if args.prompt_prefix:
        logger.info(f"Prompt prefix: '{args.prompt_prefix}'")
    
    if args.use_enhanced_embeddings:
        logger.info(f"Enhanced embeddings: Enabled")
        logger.info(f"LLM model: {args.llm_model}")
        logger.info(f"Fusion method: {args.fusion_method} (raw weight: {args.raw_weight})")
    
    # Initialize data processor
    processor = get_processor(
        args.data_path,
        format=args.format,
        batch_size=args.batch_size,
        text_field=args.text_field,
        id_field=args.id_field,
        fields_order=fields_order,
        prompt_prefix=args.prompt_prefix
    )
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_dir=raw_embeddings_dir,
        use_fp16=args.use_fp16,
        use_mrl=args.use_mrl,
        mrl_dimensions=args.mrl_dimensions
    )
    
    # Initialize enhancer if requested
    enhancer = None
    if args.use_enhanced_embeddings:
        enhancer = VLLMEnhancer(
            model_name=args.llm_model,
            cache_dir=os.path.join(profiles_dir, "cache"),
            output_dir=profiles_dir,
            use_fp16=args.use_fp16
        )
    
    # Checkpoint file
    checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
    if args.resume_from_checkpoint:
        checkpoint_file = args.resume_from_checkpoint
    
    # Process data and generate embeddings
    logger.info("\nGenerating embeddings...")
    
    # Store raw data for potential enhancement
    all_data = []
    all_ids = []
    
    # Process data in batches
    total_processed = 0
    batch_num = 0
    
    # Resume from checkpoint if specified
    start_batch_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_batch_idx = checkpoint.get('next_batch_start_idx', 0)
            total_processed = checkpoint.get('total_processed', 0)
            logger.info(f"Resuming from checkpoint at batch index {start_batch_idx}")
    
    for batch_data, batch_texts, batch_ids, batch_idx, is_last_batch in processor.process():
        # Skip batches before resume point
        if batch_idx < start_batch_idx:
            continue
            
        batch_size = len(batch_texts)
        
        # Store data for potential enhancement
        if args.use_enhanced_embeddings:
            all_data.extend(batch_data)
            all_ids.extend(batch_ids)
        
        # Generate raw embeddings
        embedding_generator.generate(batch_texts, batch_ids, batch_idx=batch_idx)
        
        # Update counters
        total_processed += batch_size
        batch_num += 1
        
        # Save checkpoint
        if batch_num % args.checkpoint_frequency == 0 or is_last_batch:
            checkpoint = {
                "next_batch_start_idx": batch_idx + batch_size,
                "total_processed": total_processed,
                "timestamp": str(time.strftime("%Y-%m-%d %H:%M:%S")),
                "enhanced_processed": False
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            logger.info(f"Saved checkpoint after {total_processed:,} examples")
    
    # Generate enhanced embeddings if requested
    if args.use_enhanced_embeddings and all_data:
        logger.info("\nGenerating enhanced embeddings...")
        
        # Generate enhanced profiles
        profiles = enhancer.generate_enhanced_profiles(
            all_data,
            ids=all_ids,
            batch_size=min(args.batch_size, 16),
            use_cache=True
        )
        
        # Generate embeddings for profiles
        profile_embedding_generator = EmbeddingGenerator(
            model_name=args.model_name,
            batch_size=args.batch_size,
            output_dir=enhanced_embeddings_dir,
            use_fp16=args.use_fp16,
            use_mrl=args.use_mrl,
            mrl_dimensions=args.mrl_dimensions
        )
        
        # Process profiles in batches
        for i in range(0, len(profiles), args.batch_size):
            batch_end = min(i + args.batch_size, len(profiles))
            batch_profiles = profiles[i:batch_end]
            batch_ids = all_ids[i:batch_end]
            
            profile_embedding_generator.generate(
                batch_profiles,
                batch_ids,
                batch_idx=i
            )
        
        # Combine raw and enhanced embeddings
        logger.info("\nCombining raw and enhanced embeddings...")
        
        # Get list of embedding files
        raw_files = sorted([f for f in os.listdir(raw_embeddings_dir) if f.endswith('.npy')])
        enhanced_files = sorted([f for f in os.listdir(enhanced_embeddings_dir) if f.endswith('.npy')])
        
        for raw_file, enhanced_file in zip(raw_files, enhanced_files):
            # Extract indices
            parts = raw_file.replace('embeddings_', '').replace('.npy', '').split('_')
            start_idx, end_idx = int(parts[0]), int(parts[1])
            
            # Load embeddings
            raw_emb = np.load(os.path.join(raw_embeddings_dir, raw_file))
            enhanced_emb = np.load(os.path.join(enhanced_embeddings_dir, enhanced_file))
            
            # Combine
            combined = combine_embeddings(
                raw_emb,
                enhanced_emb,
                method=args.fusion_method,
                raw_weight=args.raw_weight
            )
            
            # Save combined embeddings
            combined_file = os.path.join(combined_embeddings_dir, f"embeddings_{start_idx}_{end_idx}.npy")
            np.save(combined_file, combined)
            
            # Copy IDs file
            ids_file = os.path.join(raw_embeddings_dir, f"ids_{start_idx}_{end_idx}.json")
            if os.path.exists(ids_file):
                import shutil
                shutil.copy(ids_file, os.path.join(combined_embeddings_dir, f"ids_{start_idx}_{end_idx}.json"))
        
        # Update checkpoint
        checkpoint["enhanced_processed"] = True
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    # Final stats
    total_time = time.time() - start_time
    logger.info("\nEmbedding generation complete!")
    logger.info(f"Total examples processed: {total_processed:,}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average speed: {total_processed/total_time:.2f} examples/second")
    logger.info(f"Embeddings saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress up to the last checkpoint has been saved.")
        print("You can resume by using --resume_from_checkpoint option.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        print("\nProgress up to the last checkpoint has been saved.")
        print("You can resume by using --resume_from_checkpoint option.")