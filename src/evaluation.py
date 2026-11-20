"""
evaluation.py
-------------
Relevance Protocol A (category-level: articleType match) and the
retrieval metrics used to evaluate every model in this project —
pretrained baseline, fine-tuned, fused, exact, and approximate search —
under the *same* protocol so comparisons are fair.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def is_relevant(query_category: str, neighbor_category) -> bool:
    """Protocol A: relevant iff articleType matches exactly."""
    return neighbor_category == query_category


def build_category_lookup(ids_array: np.ndarray, categories_array: np.ndarray) -> Dict:
    return dict(zip(ids_array, categories_array))


def precision_recall_at_k(
    neighbor_ids: np.ndarray,
    gallery_categories_lookup: Dict,
    gallery_categories: np.ndarray,
    query_categories: np.ndarray,
    k_values: List[int],
) -> Dict[int, Dict[str, float]]:
    """neighbor_ids: [N, K_MAX] retrieved ids per query, ordered by rank."""
    n_queries = neighbor_ids.shape[0]
    results = {k: {"precision": [], "recall": []} for k in k_values}

    for i in range(n_queries):
        q_cat = query_categories[i]
        total_relevant = int(np.sum(gallery_categories == q_cat))
        if total_relevant == 0:
            continue

        for k in k_values:
            topk_ids = neighbor_ids[i, :k]
            topk_cats = [gallery_categories_lookup.get(pid, None) for pid in topk_ids if pid is not None]
            n_relevant_retrieved = sum(is_relevant(q_cat, c) for c in topk_cats)

            results[k]["precision"].append(n_relevant_retrieved / k)
            results[k]["recall"].append(n_relevant_retrieved / total_relevant)

    return {k: {"precision": float(np.mean(v["precision"])) if v["precision"] else float("nan"),
                "recall": float(np.mean(v["recall"])) if v["recall"] else float("nan")}
            for k, v in results.items()}


def average_precision_at_k(topk_ids: np.ndarray, gallery_categories_lookup: Dict, query_category: str, k: int) -> float:
    topk_ids = topk_ids[:k]
    hits, sum_precisions = 0, 0.0
    for rank, pid in enumerate(topk_ids, start=1):
        if pid is None:
            continue
        cat = gallery_categories_lookup.get(pid, None)
        if is_relevant(query_category, cat):
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / hits if hits > 0 else 0.0


def mean_average_precision_at_k(
    neighbor_ids: np.ndarray, gallery_categories_lookup: Dict, query_categories: np.ndarray, k: int
) -> float:
    aps = [
        average_precision_at_k(neighbor_ids[i], gallery_categories_lookup, query_categories[i], k)
        for i in range(neighbor_ids.shape[0])
    ]
    return float(np.mean(aps))


def category_consistency_at_k(
    neighbor_ids: np.ndarray, gallery_categories_lookup: Dict, query_categories: np.ndarray, k: int
) -> float:
    n_queries = neighbor_ids.shape[0]
    scores = []
    for i in range(n_queries):
        topk_ids = neighbor_ids[i, :k]
        topk_cats = [gallery_categories_lookup.get(pid, None) for pid in topk_ids if pid is not None]
        if not topk_cats:
            continue
        match = sum(c == query_categories[i] for c in topk_cats)
        scores.append(match / len(topk_cats))
    return float(np.mean(scores)) if scores else float("nan")


def recall_overlap_at_k(
    neighbor_ids_a: np.ndarray, neighbor_ids_b: np.ndarray, k: int
) -> float:
    """
    Phase 2 utility: fraction of neighbor_ids_b's Top-K (e.g. ANN results)
    that also appear in neighbor_ids_a's Top-K (e.g. exact k-NN results),
    averaged over queries. Used to quantify ANN recall loss vs. exact search.
    """
    n_queries = neighbor_ids_a.shape[0]
    overlaps = []
    for i in range(n_queries):
        set_a = {pid for pid in neighbor_ids_a[i, :k] if pid is not None}
        set_b = {pid for pid in neighbor_ids_b[i, :k] if pid is not None}
        if not set_a:
            continue
        overlaps.append(len(set_a & set_b) / len(set_a))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def metrics_to_dataframe(
    pr_results: Dict[int, Dict[str, float]],
    map_results: Dict[int, float],
    category_consistency_results: Dict[int, float],
    extra_rows: Dict[str, float] = None,
) -> pd.DataFrame:
    rows = []
    for k, v in pr_results.items():
        rows.append({"metric": f"Precision@{k}", "value": v["precision"]})
        rows.append({"metric": f"Recall@{k}", "value": v["recall"]})
    for k, v in map_results.items():
        rows.append({"metric": f"mAP@{k}", "value": v})
    for k, v in category_consistency_results.items():
        rows.append({"metric": f"CategoryConsistency@{k}", "value": v})
    if extra_rows:
        for name, value in extra_rows.items():
            rows.append({"metric": name, "value": value})
    return pd.DataFrame(rows)
