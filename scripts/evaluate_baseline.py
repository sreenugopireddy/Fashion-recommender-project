"""
scripts/evaluate_baseline.py
------------------------------
Builds an exact k-NN index on the train embeddings for the given --tag
(defaults to "baseline"), retrieves Top-K for every test query, computes
Precision@K/Recall@K/mAP@10/mAP@20/Category Consistency@K under Relevance
Protocol A, logs latency, and saves metrics/plots.

Run this with --tag baseline (Phase 1) and again with --tag finetuned
(Phase 2, after scripts/train_triplet.py + extract_embeddings --tag finetuned)
to get directly comparable metrics files — same test set, same protocol.

Usage:
    python scripts/evaluate_baseline.py                    # Phase 1 baseline
    python scripts/evaluate_baseline.py --tag finetuned    # Phase 2 fine-tuned
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, embeddings, evaluation, retrieval, visualization  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline", choices=["baseline", "finetuned"],
                         help="Which saved embedding set to evaluate (default: baseline).")
    args = parser.parse_args()

    config.ensure_directories()

    emb_train, ids_train, cat_train = embeddings.load_embeddings("train", tag=args.tag)
    emb_test, ids_test, cat_test = embeddings.load_embeddings("test", tag=args.tag)

    knn_index, build_time = retrieval.build_knn_index(emb_train, k_max=config.K_MAX)

    print(f"Running exact k-NN retrieval ({args.tag}) for all {len(ids_test)} test queries...")
    neighbor_ids, neighbor_sims, query_latencies = retrieval.retrieve_top_k(
        emb_test, ids_test, ids_train, k=config.K_MAX, index=knn_index
    )

    gallery_lookup = evaluation.build_category_lookup(ids_train, cat_train)

    pr_results = evaluation.precision_recall_at_k(
        neighbor_ids, gallery_lookup, cat_train, cat_test, config.K_VALUES
    )
    map_results = {
        k: evaluation.mean_average_precision_at_k(neighbor_ids, gallery_lookup, cat_test, k)
        for k in config.MAP_K_VALUES
    }
    cat_consistency = {
        k: evaluation.category_consistency_at_k(neighbor_ids, gallery_lookup, cat_test, k)
        for k in config.CATEGORY_CONSISTENCY_K_VALUES
    }

    mean_query_latency_ms = float(np.mean(query_latencies) * 1000)
    median_query_latency_ms = float(np.median(query_latencies) * 1000)
    p95_query_latency_ms = float(np.percentile(query_latencies, 95) * 1000)

    extra_rows = {
        "num_test_queries": len(ids_test),
        "num_train_gallery_items": len(ids_train),
        "knn_index_build_time_s": build_time,
        "query_latency_mean_ms": mean_query_latency_ms,
        "query_latency_median_ms": median_query_latency_ms,
        "query_latency_p95_ms": p95_query_latency_ms,
    }

    metrics_df = evaluation.metrics_to_dataframe(pr_results, map_results, cat_consistency, extra_rows)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.METRICS_DIR / f"metrics_{args.tag}_exact_knn.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    print(metrics_df.to_string(index=False))

    latency_log = pd.DataFrame([
        {"stage": "knn_index_build_s", "value": build_time},
        {"stage": "query_latency_mean_ms", "value": mean_query_latency_ms},
        {"stage": "query_latency_median_ms", "value": median_query_latency_ms},
        {"stage": "query_latency_p95_ms", "value": p95_query_latency_ms},
    ])
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    latency_log_path = config.LOGS_DIR / f"latency_log_{args.tag}.csv"
    latency_log.to_csv(latency_log_path, index=False)
    print(f"Saved latency log to {latency_log_path}")

    visualization.plot_precision_recall_and_consistency(
        config.K_VALUES, pr_results, cat_consistency,
        title_suffix=f"({args.tag}, exact k-NN)",
        save_path=config.FIGURES_DIR / f"precision_recall_category_consistency_{args.tag}.png",
    )


if __name__ == "__main__":
    main()