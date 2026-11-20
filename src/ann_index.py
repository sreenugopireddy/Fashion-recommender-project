"""
ann_index.py
------------
Phase 2: approximate nearest-neighbor indexing with FAISS or ANNOY, and
a benchmarking function that compares retrieval quality + latency against
the exact k-NN baseline in retrieval.py, on the same queries and protocol.
"""

import time
from typing import Tuple

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------
def build_faiss_index(gallery_embeddings: np.ndarray) -> Tuple["faiss.Index", float]:
    import faiss

    dim = gallery_embeddings.shape[1]
    gallery = np.ascontiguousarray(gallery_embeddings.astype(np.float32))

    start = time.time()
    if config.FAISS_INDEX_TYPE == "IndexFlatIP":
        # Exact inner-product search via FAISS — useful as an "exact, but
        # FAISS-backed" reference point distinct from sklearn's brute-force.
        index = faiss.IndexFlatIP(dim)
    elif config.FAISS_INDEX_TYPE == "IndexIVFFlat":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, config.FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
        index.train(gallery)
        index.nprobe = config.FAISS_NPROBE
    else:
        raise ValueError(f"Unsupported FAISS_INDEX_TYPE: {config.FAISS_INDEX_TYPE!r}")

    index.add(gallery)
    build_time = time.time() - start
    print(f"FAISS ({config.FAISS_INDEX_TYPE}) index built on {gallery.shape[0]} items in {build_time:.3f}s")
    return index, build_time


def search_faiss_index(
    index: "faiss.Index",
    query_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_queries = query_embeddings.shape[0]
    neighbor_ids = np.empty((n_queries, k), dtype=object)
    neighbor_sims = np.empty((n_queries, k), dtype=np.float32)
    latencies = np.empty(n_queries, dtype=np.float64)

    query = np.ascontiguousarray(query_embeddings.astype(np.float32))

    for i in range(n_queries):
        q = query[i:i + 1]
        t0 = time.time()
        sims, indices = index.search(q, k + 1)  # +1 in case of self-match
        latencies[i] = time.time() - t0

        sims, indices = sims[0], indices[0]
        valid = indices >= 0
        cand_ids = gallery_ids[indices[valid]]
        cand_sims = sims[valid]

        keep_mask = cand_ids != query_ids[i]
        cand_ids = cand_ids[keep_mask][:k]
        cand_sims = cand_sims[keep_mask][:k]

        if len(cand_ids) < k:
            pad = k - len(cand_ids)
            cand_ids = np.concatenate([cand_ids, np.array([None] * pad, dtype=object)])
            cand_sims = np.concatenate([cand_sims, np.full(pad, np.nan, dtype=np.float32)])

        neighbor_ids[i] = cand_ids
        neighbor_sims[i] = cand_sims

    return neighbor_ids, neighbor_sims, latencies


# ---------------------------------------------------------------------------
# ANNOY backend
# ---------------------------------------------------------------------------
def build_annoy_index(gallery_embeddings: np.ndarray, n_trees: int = config.ANNOY_N_TREES):
    from annoy import AnnoyIndex

    dim = gallery_embeddings.shape[1]
    index = AnnoyIndex(dim, metric="angular")  # angular ~ cosine for normalized vectors

    start = time.time()
    for i, vec in enumerate(gallery_embeddings):
        index.add_item(i, vec.tolist())
    index.build(n_trees)
    build_time = time.time() - start
    print(f"ANNOY index built on {gallery_embeddings.shape[0]} items ({n_trees} trees) in {build_time:.3f}s")
    return index, build_time


def search_annoy_index(
    index,
    query_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_queries = query_embeddings.shape[0]
    neighbor_ids = np.empty((n_queries, k), dtype=object)
    neighbor_sims = np.empty((n_queries, k), dtype=np.float32)
    latencies = np.empty(n_queries, dtype=np.float64)

    for i in range(n_queries):
        t0 = time.time()
        indices, distances = index.get_nns_by_vector(
            query_embeddings[i].tolist(), k + 1, include_distances=True
        )
        latencies[i] = time.time() - t0

        cand_ids = gallery_ids[indices]
        # ANNOY angular distance -> approximate cosine similarity
        cand_sims = 1.0 - (np.array(distances) ** 2) / 2.0

        keep_mask = cand_ids != query_ids[i]
        cand_ids = cand_ids[keep_mask][:k]
        cand_sims = cand_sims[keep_mask][:k]

        if len(cand_ids) < k:
            pad = k - len(cand_ids)
            cand_ids = np.concatenate([cand_ids, np.array([None] * pad, dtype=object)])
            cand_sims = np.concatenate([cand_sims, np.full(pad, np.nan, dtype=np.float32)])

        neighbor_ids[i] = cand_ids
        neighbor_sims[i] = cand_sims

    return neighbor_ids, neighbor_sims, latencies


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------
def build_ann_index(gallery_embeddings: np.ndarray, backend: str = config.ANN_BACKEND):
    if backend == "faiss":
        return build_faiss_index(gallery_embeddings)
    elif backend == "annoy":
        return build_annoy_index(gallery_embeddings)
    raise ValueError(f"Unknown ANN_BACKEND: {backend!r} (expected 'faiss' or 'annoy')")


def search_ann_index(
    index, query_embeddings: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray, k: int,
    backend: str = config.ANN_BACKEND,
):
    if backend == "faiss":
        return search_faiss_index(index, query_embeddings, query_ids, gallery_ids, k)
    elif backend == "annoy":
        return search_annoy_index(index, query_embeddings, query_ids, gallery_ids, k)
    raise ValueError(f"Unknown ANN_BACKEND: {backend!r} (expected 'faiss' or 'annoy')")
