# Project Structure
lanns/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── segmenters.py      # Segmentation strategies
│   ├── sharding.py        # Sharding logic
│   └── hnsw_utils.py      # HNSW integration
├── indexing/
│   ├── __init__.py
│   ├── builder.py         # LANNS index builder
│   └── storage.py         # Index storage utilities
├── query/
│   ├── __init__.py
│   ├── router.py          # Query routing logic
│   └── merger.py          # Result merging logic
├── build_index.py         # Script to build LANNS index from embeddings
├── query_index.py         # Script to query the built index
└── evaluate.py            # Simple evaluation script