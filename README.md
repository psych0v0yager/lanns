# LANNS: Library for Large Scale Approximate Nearest Neighbor Search

This is an implementation of the LANNS system described in the paper "LANNS: A Web-Scale Approximate Nearest Neighbor Lookup System".

## Overview

LANNS is a scalable system for approximate nearest neighbor search that can efficiently handle web-scale datasets (100M+ records in high dimensions). It uses a two-level partitioning strategy to overcome the limitations of the HNSW algorithm for large datasets:

1. **Sharding (Level 1)**: Horizontal partitioning of data across multiple machines
2. **Segmentation (Level 2)**: Further partitioning within each shard using one of three strategies:
   - Random Segmenter (RS)
   - Random Hyperplane Segmenter (RH)
   - Approximate Principal Direction Segmenter (APD)

## Features

- Scalable to web-scale datasets (100M+ records)
- High throughput (~2.5k QPS on a single node)
- Low latency (few milliseconds per query)
- High recall (95%+ for most configurations)
- Flexible segmentation strategies
- Built on the state-of-the-art HNSW algorithm
- **NEW**: Integrated embedding generation from raw data
- **NEW**: Optional smart embeddings enhancement (requires vLLM)
- **NEW**: End-to-end pipeline from raw data to queryable index

## Installation

```bash
# Clone the repository
git clone https://github.com/psych0v0yager/lanns.git
cd lanns

# Basic installation
pip install -r requirements.txt
pip install -e .

# For enhanced embeddings support (optional)
pip install "vllm>=0.2.0" "torch>=2.0.0"
# Or simply:
pip install -e ".[enhanced]"

# To install everything
pip install -e ".[all]"
```

## Dependencies

- numpy
- scipy
- h5py
- tqdm
- matplotlib (for evaluation plots)
- sentence-transformers (for embedding generation)
- One of the following HNSW libraries:
  - nmslib (recommended)
  - hnswlib
  - faiss
- vLLM (optional, for enhanced embeddings)

## Usage

### Complete Pipeline

For an end-to-end workflow from raw data to queryable index:

```bash
python run_pipeline.py --data_file /path/to/your/data.ndjson \
    --output_dir ./lanns_project \
    --num_shards 2 \
    --num_segments 16 \
    --segmenter apd \
    --create_query_samples
```

With enhanced embeddings (requires vLLM):

```bash
python run_pipeline.py --data_file /path/to/your/data.ndjson \
    --output_dir ./lanns_project \
    --use_enhanced_embeddings \
    --llm_model meta-llama/Llama-2-7b-chat-hf \
    --num_shards 2 \
    --num_segments 16
```

### Generating Embeddings

Generate embeddings from raw data:

```bash
# From NDJSON data
python generate_embeddings.py --data_path /path/to/data.ndjson \
    --output_dir ./embeddings \
    --model_name Snowflake/snowflake-arctic-embed-l-v2.0 \
    --batch_size 1024

# With MRL dimensionality reduction
python generate_embeddings.py --data_path /path/to/data.ndjson \
    --output_dir ./embeddings \
    --use_mrl --mrl_dimensions 256

# With enhanced embeddings (requires vLLM)
python generate_embeddings.py --data_path /path/to/data.ndjson \
    --output_dir ./embeddings \
    --use_enhanced_embeddings \
    --llm_model meta-llama/Llama-2-7b-chat-hf
```

### Building an Index

```bash
# From a directory of embedding files
python build_index.py --embeddings_dir /path/to/embeddings --output_dir ./lanns_index --num_shards 4 --num_segments 8 --segmenter apd

# From a single numpy file
python build_index.py --embeddings_file all_embeddings.npy --ids_file all_ids.json --output_dir ./lanns_index --num_shards 4 --num_segments 8 --segmenter apd

# From an HDF5 file
python build_index.py --embeddings_h5 embeddings.h5 --output_dir ./lanns_index --num_shards 4 --num_segments 8 --segmenter apd
```

### Querying an Index

```bash
# From a numpy file
python query_index.py --index_dir ./lanns_index --query_file query_embeddings.npy --k 100

# From an HDF5 file
python query_index.py --index_dir ./lanns_index --query_h5 queries.h5 --k 100

# From input text (requires sentence-transformers)
python query_index.py --index_dir ./lanns_index --input_text "query text" --k 100
```

### Evaluating Performance

```bash
python evaluate.py --embeddings_file all_embeddings.npy --query_file query_embeddings.npy --output_dir ./eval_results --plot
```

## Parameter Tuning

### Key Parameters

- **num_shards**: Number of shards (level 1 partitioning)
- **num_segments**: Number of segments per shard (level 2 partitioning)
- **segmenter**: Segmentation strategy ('rs', 'rh', or 'apd')
- **spill**: Spill parameter (0.15 means 30% of queries go to both sides)
- **hnsw_m**: HNSW parameter for max number of connections
- **hnsw_ef_construction**: HNSW parameter for index building
- **hnsw_ef_search**: HNSW parameter for search

#### Embedding Generation Parameters

- **model_name**: Name of the sentence-transformer model to use
- **batch_size**: Batch size for embedding generation
- **use_mrl**: Apply dimensionality reduction
- **mrl_dimensions**: Number of dimensions to keep when using MRL
- **use_enhanced_embeddings**: Enable enhanced embeddings using vLLM
- **llm_model**: vLLM model to use for profile enhancement

### Recommended Configurations

- For maximum recall (95%+):
  - segmenter: 'apd'
  - spill: 0.15
  - hnsw_ef_construction: 200
  - hnsw_ef_search: 100

- For maximum throughput:
  - segmenter: 'rh'
  - spill: 0.10
  - hnsw_ef_construction: 100
  - hnsw_ef_search: 50

- For best embedding quality:
  - model_name: 'Snowflake/snowflake-arctic-embed-l-v2.0'
  - use_enhanced_embeddings: True
  - llm_model: 'meta-llama/Llama-2-7b-chat-hf'

## Python API

### Generating Embeddings

```python
from lanns.embeddings.generator import EmbeddingGenerator
from lanns.embeddings.processor import get_processor

# Initialize processor
processor = get_processor(
    file_path='/path/to/data.ndjson',
    batch_size=1024,
    text_field='description'  # optional
)

# Initialize embedding generator
generator = EmbeddingGenerator(
    model_name='Snowflake/snowflake-arctic-embed-l-v2.0',
    batch_size=1024,
    output_dir='./embeddings',
    use_mrl=True,
    mrl_dimensions=256
)

# Process with checkpointing
generator.process_with_checkpointing(
    processor,
    checkpoint_file='./checkpoint.json',
    checkpoint_frequency=100
)
```

### Enhanced Embeddings (Optional)

```python
from lanns.embeddings import VLLMEnhancer, combine_embeddings
import numpy as np

# Initialize enhancer
enhancer = VLLMEnhancer(
    model_name='meta-llama/Llama-2-7b-chat-hf',
    cache_dir='./profiles_cache',
    output_dir='./enhanced_profiles'
)

# Generate enhanced profiles
profiles = enhancer.generate_enhanced_profiles(data_list, ids=ids)

# Generate embeddings for profiles
profile_embeddings = generator.generate(profiles)

# Combine with raw embeddings
raw_embeddings = np.load('raw_embeddings.npy')
combined = combine_embeddings(
    raw_embeddings, 
    profile_embeddings,
    method='weighted_average',
    raw_weight=0.7
)
```

### Building an Index

```python
from lanns.indexing.builder import LANNSIndexBuilder

# Create builder
builder = LANNSIndexBuilder(
    num_shards=4,
    num_segments=8,
    segmenter_type='apd',
    spill=0.15,
    space='l2',
    hnsw_m=16,
    hnsw_ef_construction=200
)

# Build index
builder.build(embeddings, ids, 'lanns_index')
```

### Querying an Index

```python
from lanns.indexing.storage import LANNSIndex

# Load index
index = LANNSIndex('lanns_index', ef_search=100)

# Single query
ids, distances = index.query(query_embedding, k=10)

# Batch query
ids_list, distances_list = index.batch_query(query_embeddings, k=10)
```

## Implementation Notes

1. The current implementation supports three backends for HNSW:
   - nmslib (recommended for best performance)
   - hnswlib
   - faiss

2. The system automatically uses the first available HNSW library in the order listed above.

3. For very large datasets, the system can process data in batches to manage memory usage.

4. Enhanced embeddings require vLLM, which needs a CUDA-capable GPU. If vLLM is not available, the system will fall back to standard embeddings.

5. For checkpointing support, the embedding generator saves progress periodically. You can resume processing from a checkpoint if interrupted.

## Citation

```
@article{doshi2022lanns,
  title={LANNS: A Web-Scale Approximate Nearest Neighbor Lookup System},
  author={Doshi, Ishita and Das, Dhritiman and Bhutani, Ashish and Kumar, Rajeev and Bhatt, Rushi and Balasubramanian, Niranjan},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={4},
  pages={850--858},
  year={2022},
  publisher={VLDB Endowment}
}
```

## License

This implementation is provided under the MIT License.