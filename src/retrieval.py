"""
retrieval.py
------------
Exact k-NN retrieval using scikit-learn, cosine similarity on
L2-normalized embeddings. This is the Phase 1 baseline search, and is
also used in Phase 2 as the "exact" reference point when benchmarking
FAISS/ANNOY approximate search.
"""

import time
from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config


def build_knn_index(gallery_embeddings: np.ndarray, k_max: int = config.K_MAX) -> Tuple[NearestNeighbors, float]:
    """Fits an exact cosine-similarity k-NN index on the gallery (train) embeddings."""
    start = time.time()
    index = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine", algorithm="brute")
    index.fit(gallery_embeddings)
    build_time = time.time() - start
    print(f"Exact k-NN index built on {gallery_embeddings.shape[0]} items in {build_time:.3f}s")
    return index, build_time


def retrieve_top_k(
    query_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    k: int,
    index: NearestNeighbors,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves the Top-K nearest gallery neighbors for each query, excluding
    an exact product-ID self-match if present in the gallery.

    Returns: neighbor_ids [N,k] (object array, None-padded), neighbor_sims [N,k],
    per-query latency array [N] (seconds).
    """
    n_queries = query_embeddings.shape[0]
    neighbor_ids = np.empty((n_queries, k), dtype=object)
    neighbor_sims = np.empty((n_queries, k), dtype=np.float32)
    latencies = np.empty(n_queries, dtype=np.float64)

    for i in range(n_queries):
        q = query_embeddings[i:i + 1]
        t0 = time.time()
        distances, indices = index.kneighbors(q, n_neighbors=k + 1)
        latencies[i] = time.time() - t0

        distances, indices = distances[0], indices[0]
        cand_ids = gallery_ids[indices]

        keep_mask = cand_ids != query_ids[i]
        cand_ids = cand_ids[keep_mask][:k]
        cand_sims = (1.0 - distances[keep_mask][:k])

        if len(cand_ids) < k:
            pad = k - len(cand_ids)
            cand_ids = np.concatenate([cand_ids, np.array([None] * pad, dtype=object)])
            cand_sims = np.concatenate([cand_sims, np.full(pad, np.nan, dtype=np.float32)])

        neighbor_ids[i] = cand_ids
        neighbor_sims[i] = cand_sims

    return neighbor_ids, neighbor_sims, latencies
