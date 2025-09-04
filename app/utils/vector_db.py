"""FAISS‑based vector database utilities.

This module wraps the FAISS library to provide a simple API for building
and querying a vector similarity index.  The index stores dense vector
embeddings and allows nearest‑neighbor search to find the most similar
vectors to a given query.  Metadata (e.g. the original text and source
filename) are kept in memory alongside the index so that search results
can be mapped back to human‑readable content.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .helpers import get_embedding_model
import time
from functools import lru_cache

# Global variables for caching the database and its last load time
cache = {}
last_load_time = 0
cache_lifetime = 24 * 60 * 60  # 24 hours in seconds
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class VectorDatabase:
    """A simple wrapper around a FAISS index and associated metadata."""

    def __init__(self, index: faiss.Index, metadata: List[Dict[str, Any]]):
        self.index = index
        self.metadata = metadata

    @classmethod
    def build(
        cls, embeddings: np.ndarray, metadata: List[Dict[str, Any]], use_gpu: bool = False
    ) -> "VectorDatabase":
        """Build a FAISS index from embeddings and return a `VectorDatabase`.

        Args:
            embeddings: NumPy array of shape (n_vectors, dim).
            metadata: List of metadata dicts, one per embedding.
            use_gpu: If True and CUDA is available, move the index to GPU.

        Returns:
            A VectorDatabase instance containing the index and metadata.
        """
        # assert len(embeddings) == len(metadata), "Embeddings and metadata length mismatch"
        dim = embeddings.shape[1]
        logging.info(f"Building FAISS index with {len(embeddings)} vectors of dimension {dim}")
        index = faiss.IndexFlatL2(dim)  # exact search using L2 distance
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logging.info("Using GPU for FAISS index")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU resources: {e}; falling back to CPU")
        index.add(embeddings.astype(np.float32))
        logging.info("Index built and vectors added")
        return cls(index, metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for the nearest neighbors of a query embedding.

        Args:
            query_embedding: NumPy array of shape (dim,) representing the query.
            top_k: Number of nearest neighbors to return.

        Returns:
            A list of (index, distance) tuples for the top_k matches.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, top_k)
        results: List[Tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            results.append((int(idx), float(dist)))
        return results

    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve the original text chunks for the given vector indices."""
        texts: List[str] = []
        for idx in indices:
            if 0 <= idx < len(self.metadata):
                texts.append(self.metadata[idx]["text"])
        return texts

    def save(self, index_path: str, metadata_path: str = None) -> None:
        """Persist the FAISS index and metadata to disk.

        The index is saved using FAISS's native serialization, while
        metadata is pickled.
        """
        # If the index is on GPU, move it back to CPU before saving
        if isinstance(self.index, faiss.IndexPreTransform):
            # Composite indexes may wrap other types; handle simply
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            except Exception:
                cpu_index = self.index
        logging.info(f"Saving index to {index_path}")
        faiss.write_index(cpu_index, index_path)
        if metadata_path:
            metadata_path = Path(metadata_path)
            logging.info(f"Saving metadata to {metadata_path}")
            with metadata_path.open("wb") as f:
                pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> VectorDatabase:
        """Load a FAISS index and metadata from disk."""
        logging.info(f"Loading index from {index_path}")
        index = faiss.read_index(index_path)
        if metadata_path:
            logging.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            # metadata = pickle.load(Path(metadata_path).open("rb"))
            print(f"Loaded metadata with {len(metadata)} entries")
            return cls(index, metadata)
        else:
            logging.warning("No metadata path provided; metadata will be empty")
            return cls(index, [])


def run_build(embeddings_path: str, index_out: str, metadata_out: str = None) -> None:
    """CLI subcommand: build a vector database from embeddings and metadata."""
    embeddings = np.load(embeddings_path)
    logging.info(f"Loaded embeddings array with shape {embeddings.shape}")
    # with open(args.metadata_path, "rb") as f:
    #     metadata = pickle.load(f)
    db = VectorDatabase.build(embeddings, metadata=None, use_gpu=False)
    db.save(index_out, metadata_out)
    # logging.info(
    #     f"Finished building vector database. Index saved to {args.index_out}; metadata saved to {args.metadata_out}"
    # )

def load_db(index_path, metadata_path: str = None):
    global last_load_time, cache
    
    current_time = time.time()  # Get the current time in seconds
    
    # Check if the database is already cached and if it's still valid (not expired)
    if index_path in cache and (current_time - last_load_time) < cache_lifetime:
        print(f"Using cached database for {index_path}")
        return cache[index_path]
    
    # If the cache has expired or the database is not cached, load the DB again
    print(f"Loading database from {index_path}...")
    db = VectorDatabase.load(index_path, metadata_path)  # Replace with actual DB loading logic
    
    # Update the cache and the timestamp of the last load
    cache[index_path] = db
    last_load_time = current_time    
    return db


def run_vector_search(query, db, model, top_k=3) -> None:
    """CLI subcommand: search the vector database for a query string."""
    
    # Encode the query
    query_embedding = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    # Search
    results = db.search(query_embedding, top_k=top_k)
    if not results:
        print("No results found.")
        return None
    vector_text = "\n".join("")
    for idx, dist in results:
        meta = db.metadata[idx]
        text = meta['text']
        vector_text += text + "\n"
    return vector_text


def run_vector_search_v2(query, dense_index):

    # Search the dense index
    results = dense_index.search(
        namespace="example-namespace",
        query={
            "top_k": 10,
            "inputs": {
                'text': query
            }
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 3,
            "rank_fields": ["chunk_text"]
        }  
    )
    response = ""
    # Print the results
    for hit in results['result']['hits']:
        response += hit['fields']['chunk_text'] + "\n"
    return response
