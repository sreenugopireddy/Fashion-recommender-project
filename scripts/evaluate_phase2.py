"""
scripts/evaluate_phase2.py
-----------------------------
Phase 2: if the dataset has a product-title text column
(config.HAS_TITLE_COLUMN, set via configs/phase1_config.yaml ->
dataset_has_title_column), builds fused image+text embeddings from the
fine-tuned image embeddings and titles, then compares image-only
(finetuned) vs. fused retrieval on the same test queries and Protocol A —
this answers RQ3 (does fusion reduce cross-category errors?).

If the dataset has NO title/text field (e.g. clothing-dataset-full, which
only has a category "label" and a "kids" boolean — encoding the category
itself as "text" would leak the answer into the fusion embedding and
invalidate the comparison), this step is skipped entirely and only the
image-only (fine-tuned) metrics are reported/saved. See README.md's
"Known scope limits" for the documented reasoning.

Prerequisite: run, in order —
    python scripts/train_triplet.py
    python scripts/extract_embeddings.py --split all --tag finetuned --checkpoint results/checkpoints/resnet50_triplet_finetuned.pt

Usage:
    python scripts/evaluate_phase2.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, embeddings, evaluation, preprocess, retrieval, text_fusion, visualization  # noqa: E402


def evaluate_embeddings(emb_train, ids_train, cat_train, emb_test, ids_test, cat_test, label: str):
    knn_index, _ = retrieval.build_knn_index(emb_train, k_max=config.K_MAX)
    neighbor_ids, _, _ = retrieval.retrieve_top_k(emb_test, ids_test, ids_train, k=config.K_MAX, index=knn_index)
    gallery_lookup = evaluation.build_category_lookup(ids_train, cat_train)

    pr_results = evaluation.precision_recall_at_k(neighbor_ids, gallery_lookup, cat_train, cat_test, config.K_VALUES)
    map_results = {k: evaluation.mean_average_precision_at_k(neighbor_ids, gallery_lookup, cat_test, k)
                   for k in config.MAP_K_VALUES}
    cat_consistency = {k: evaluation.category_consistency_at_k(neighbor_ids, gallery_lookup, cat_test, k)
                        for k in config.CATEGORY_CONSISTENCY_K_VALUES}

    print(f"\n--- {label} ---")
    for k in config.K_VALUES:
        print(f"Precision@{k}: {pr_results[k]['precision']:.4f} | Recall@{k}: {pr_results[k]['recall']:.4f}")
    for k, v in cat_consistency.items():
        print(f"Category Consistency@{k}: {v:.4f}")

    return evaluation.metrics_to_dataframe(pr_results, map_results, cat_consistency)


def main():
    config.ensure_directories()
    train_df, _, test_df = preprocess.load_splits()

    emb_train, ids_train, cat_train = embeddings.load_embeddings("train", tag="finetuned")
    emb_test, ids_test, cat_test = embeddings.load_embeddings("test", tag="finetuned")

    # --- Image-only (fine-tuned) baseline for this comparison ---------------
    image_only_metrics = evaluate_embeddings(
        emb_train, ids_train, cat_train, emb_test, ids_test, cat_test, "Image-only (fine-tuned ResNet50)"
    )
    image_only_metrics["model"] = "image_only_finetuned"

    if not config.HAS_TITLE_COLUMN:
        print(
            "\nSkipping image+text fusion: this dataset has no product-title "
            "text column (config.HAS_TITLE_COLUMN is False). Fusing the "
            "category label itself as 'text' would leak the evaluation "
            "target into the embedding, so RQ3 is not evaluated for this "
            "dataset — see README.md's 'Known scope limits'."
        )
        out_path = config.METRICS_DIR / "phase2_image_only.csv"
        image_only_metrics.to_csv(out_path, index=False)
        print(f"Saved image-only metrics to {out_path}")
        return

    # --- Fused image + text ---------------------------------------------------
    id_to_title_train = dict(zip(train_df["id"].astype(str), train_df["productDisplayName"]))
    id_to_title_test = dict(zip(test_df["id"].astype(str), test_df["productDisplayName"]))
    titles_train = pd.Series([id_to_title_train.get(pid, "") for pid in ids_train])
    titles_test = pd.Series([id_to_title_test.get(pid, "") for pid in ids_test])

    fused_train, _ = text_fusion.build_fused_embeddings(emb_train, titles_train)
    fused_test, _ = text_fusion.build_fused_embeddings(emb_test, titles_test)

    fused_metrics = evaluate_embeddings(
        fused_train, ids_train, cat_train, fused_test, ids_test, cat_test, "Fused image + text"
    )
    fused_metrics["model"] = "fused_image_text"

    combined = pd.concat([image_only_metrics, fused_metrics], ignore_index=True)
    out_path = config.METRICS_DIR / "phase2_image_only_vs_fused.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
