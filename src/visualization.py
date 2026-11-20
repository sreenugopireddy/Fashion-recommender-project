"""
visualization.py
-----------------
Plotting utilities: category distribution, Precision/Recall curves,
Category Consistency bars, exact-vs-ANN latency/quality comparison,
and qualitative query -> Top-K retrieval grids.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config


def plot_category_distribution(df: pd.DataFrame, top_n: int = 20, save_path: Optional[Path] = None):
    category_counts = df["articleType"].value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Number of products")
    plt.title(f"Top {top_n} article types by frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_precision_recall_and_consistency(
    k_values: List[int],
    pr_results: Dict[int, Dict[str, float]],
    category_consistency: Dict[int, float],
    title_suffix: str = "",
    save_path: Optional[Path] = None,
):
    precisions = [pr_results[k]["precision"] for k in k_values]
    recalls = [pr_results[k]["recall"] for k in k_values]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(k_values, precisions, marker="o", label="Precision@K")
    ax[0].plot(k_values, recalls, marker="s", label="Recall@K")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("Score")
    ax[0].set_title(f"Precision@K / Recall@K {title_suffix}")
    ax[0].set_xticks(k_values)
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    cc_k = list(category_consistency.keys())
    cc_v = list(category_consistency.values())
    ax[1].bar([str(k) for k in cc_k], cc_v, color="steelblue")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("Category Consistency")
    ax[1].set_title(f"Category Consistency@K {title_suffix}")
    ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_exact_vs_ann_tradeoff(
    labels: List[str],
    latencies_ms: List[float],
    precision_at_k: List[float],
    k: int,
    save_path: Optional[Path] = None,
):
    """Phase 2: side-by-side latency and Precision@K bars for exact vs. ANN backends."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].bar(labels, latencies_ms, color="indianred")
    ax[0].set_ylabel("Mean query latency (ms)")
    ax[0].set_title("Query latency: exact vs. approximate")
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].bar(labels, precision_at_k, color="seagreen")
    ax[1].set_ylabel(f"Precision@{k}")
    ax[1].set_title(f"Retrieval quality: exact vs. approximate")
    ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_query(
    query_id,
    query_category: str,
    query_image_path: str,
    topk_ids: np.ndarray,
    topk_sims: np.ndarray,
    gallery_df: pd.DataFrame,
    k: int = 5,
    save_path: Optional[Path] = None,
):
    """Shows a query image alongside its Top-K retrieved products."""
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.5))

    axes[0].imshow(Image.open(query_image_path).convert("RGB"))
    axes[0].set_title(f"QUERY\n{query_category}", fontsize=9)
    axes[0].axis("off")

    for rank, (pid, sim) in enumerate(zip(topk_ids[:k], topk_sims[:k]), start=1):
        ax = axes[rank]
        if pid is None:
            ax.axis("off")
            continue
        row = gallery_df[gallery_df["id"] == pid].iloc[0]
        ax.imshow(Image.open(row["image_path"]).convert("RGB"))
        ax.set_title(f"#{rank} sim={sim:.3f}\n{row['articleType']}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
