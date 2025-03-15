from setuptools import setup, find_packages

setup(
    name="lanns",
    version="0.1.0",
    description="Library for Large Scale Approximate Nearest Neighbor Search",
    author="Ian Matejka",
    author_email="ian.matejka@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "h5py>=3.1.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "nmslib": ["nmslib>=2.1.0"],
        "hnswlib": ["hnswlib>=0.6.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
        "text": ["sentence-transformers>=2.2.0"],
        "all": [
            "nmslib>=2.1.0",
            "hnswlib>=0.6.0", 
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.2.0"
        ]
    },
    python_requires=">=3.7",
)