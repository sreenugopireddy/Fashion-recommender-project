"""
scripts/build_ann_index.py
-----------------------------
Phase 2: builds an approximate nearest-neighbor index (FAISS or ANNOY,
per config.ANN_BACKEND) on the gallery embeddings, benchmarks it against
the exact k-NN baseline on the same test queries, and saves a comparison
report + plot.

By default this operates on the fine-tuned ("finetuned") image embeddings.
Pass --tag baseline to benchmark ANN vs. exact on the Phase 1 embeddings
instead, or --fused to use the fused image+text embeddings (run
scripts/evaluate_phase2.py's fusion step first to produce those).

Usage:
    python scripts/build_ann_index.py --tag finetuned
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import ann_index, config, embeddings, evaluation, retrieval, visualization  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="finetuned", choices=["baseline", "finetuned", "fused"])
    parser.add_argument("--k", type=int, default=10, help="K used for the quality-vs-latency comparison plot.")
    args = parser.parse_args()

    config.ensure_directories()

    emb_train, ids_train, cat_train = embeddings.load_embeddings("train", tag=args.tag)
    emb_test, ids_test, cat_test = embeddings.load_embeddings("test", tag=args.tag)
    gallery_lookup = evaluation.build_category_lookup(ids_train, cat_train)

    # --- Exact k-NN reference ----------------------------------------------
    knn_index, exact_build_time = retrieval.build_knn_index(emb_train, k_max=config.K_MAX)
    exact_ids, exact_sims, exact_latencies = retrieval.retrieve_top_k(
        emb_test, ids_test, ids_train, k=config.K_MAX, index=knn_index
    )
    exact_pr = evaluation.precision_recall_at_k(exact_ids, gallery_lookup, cat_train, cat_test, [args.k])

    # --- Approximate search --------------------------------------------------
    ann, ann_build_time = ann_index.build_ann_index(emb_train, backend=config.ANN_BACKEND)
    ann_ids, ann_sims, ann_latencies = ann_index.search_ann_index(
        ann, emb_test, ids_test, ids_train, k=config.K_MAX, backend=config.ANN_BACKEND
    )
    ann_pr = evaluation.precision_recall_at_k(ann_ids, gallery_lookup, cat_train, cat_test, [args.k])

    overlap_at_k = evaluation.recall_overlap_at_k(exact_ids, ann_ids, k=args.k)

    # --- Report ---------------------------------------------------------------
    print(f"\n=== Exact k-NN vs {config.ANN_BACKEND.upper()} (tag={args.tag}) ===")
    print(f"Index build time — exact: {exact_build_time:.3f}s | approx: {ann_build_time:.3f}s")
    print(f"Mean query latency  — exact: {np.mean(exact_latencies)*1000:.3f}ms "
          f"| approx: {np.mean(ann_latencies)*1000:.3f}ms")
    print(f"Precision@{args.k}       — exact: {exact_pr[args.k]['precision']:.4f} "
          f"| approx: {ann_pr[args.k]['precision']:.4f}")
    print(f"Top-{args.k} overlap (approx vs exact): {overlap_at_k:.4f}")

    comparison_path = config.METRICS_DIR / f"exact_vs_{config.ANN_BACKEND}_{args.tag}.csv"
    import pandas as pd
    pd.DataFrame([
        {"backend": "exact_knn", "index_build_time_s": exact_build_time,
         "mean_query_latency_ms": float(np.mean(exact_latencies) * 1000),
         f"precision_at_{args.k}": exact_pr[args.k]["precision"]},
        {"backend": config.ANN_BACKEND, "index_build_time_s": ann_build_time,
         "mean_query_latency_ms": float(np.mean(ann_latencies) * 1000),
         f"precision_at_{args.k}": ann_pr[args.k]["precision"],
         f"top_{args.k}_overlap_with_exact": overlap_at_k},
    ]).to_csv(comparison_path, index=False)
    print(f"Saved comparison to {comparison_path}")

    visualization.plot_exact_vs_ann_tradeoff(
        labels=["Exact k-NN", config.ANN_BACKEND.upper()],
        latencies_ms=[float(np.mean(exact_latencies) * 1000), float(np.mean(ann_latencies) * 1000)],
        precision_at_k=[exact_pr[args.k]["precision"], ann_pr[args.k]["precision"]],
        k=args.k,
        save_path=config.FIGURES_DIR / f"exact_vs_{config.ANN_BACKEND}_{args.tag}.png",
    )


if __name__ == "__main__":
    main()
