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

## Installation

```bash
# Clone the repository
git clone https://github.com/psych0v0yager/lanns.git
cd LANNS_Implementation

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Dependencies

- numpy
- scipy
- h5py
- tqdm
- matplotlib (for evaluation plots)
- One of the following HNSW libraries:
  - nmslib (recommended)
  - hnswlib
  - faiss

## Usage

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

## Python API

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