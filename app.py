"""
app.py
------
Streamlit demo for the visual retrieval project, covering both phases:
  - Model selector: Phase 1 pretrained baseline vs. Phase 2 fine-tuned.
  - Search backend selector: exact k-NN vs. FAISS approximate search.
Pick a test-set image or upload your own, and see the Top-K most visually
similar products, with live query latency so the Phase 2 speed/quality
findings (RQ2, RQ3) are directly visible, not just numbers in a report.

Run from the project root:
    streamlit run app.py

Prerequisites:
    python scripts/extract_embeddings.py --split all                     # baseline
    python scripts/train_triplet.py                                       # optional, for fine-tuned
    python scripts/extract_embeddings.py --split all --tag finetuned \\
        --checkpoint results/checkpoints/resnet50_triplet_finetuned.pt   # optional, for fine-tuned
The fine-tuned model option is skipped gracefully (with an explanation) if
its files aren't present — the baseline is required.
"""

import sys
import time
import urllib.parse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

from src import ann_index, config, data_loader, embeddings, evaluation, preprocess, retrieval

st.set_page_config(page_title="Visual Product Retrieval Demo", layout="wide")

MODEL_OPTIONS = {
    "Baseline (pretrained ResNet50)": "baseline",
    "Fine-tuned (triplet loss, Phase 2)": "finetuned",
}
BACKEND_OPTIONS = ["Exact k-NN", "FAISS (approximate)"]


def ecommerce_search_links(category: str) -> str:
    """
    Builds category-based search links (NOT direct links to the exact
    retrieved product — this dataset's images aren't tied to real product
    listings, so this searches each platform for the category name instead).
    """
    q = urllib.parse.quote_plus(category)
    amazon_url = f"https://www.amazon.in/s?k={q}"
    flipkart_url = f"https://www.flipkart.com/search?q={q}"
    google_shopping_url = f"https://www.google.com/search?tbm=shop&q={q}"
    return f"[Amazon]({amazon_url}) · [Flipkart]({flipkart_url}) · [Google Shopping]({google_shopping_url})"


@st.cache_resource
def load_model(tag: str):
    checkpoint_path = config.CHECKPOINT_PATH if tag == "finetuned" else None
    return embeddings.build_resnet50_backbone(checkpoint_path=checkpoint_path, freeze=True)


@st.cache_resource
def load_train_gallery(tag: str):
    return embeddings.load_embeddings("train", tag=tag)


@st.cache_resource
def load_test_gallery(tag: str):
    return embeddings.load_embeddings("test", tag=tag)


@st.cache_resource
def build_search_index(tag: str, backend: str):
    """backend: 'exact' or 'faiss'. Returns (index, build_time_s)."""
    emb_train, _, _ = load_train_gallery(tag)
    if backend == "exact":
        return retrieval.build_knn_index(emb_train, k_max=config.K_MAX)
    return ann_index.build_ann_index(emb_train, backend="faiss")


@st.cache_data
def load_splits():
    train_df, _, test_df = preprocess.load_splits()
    return train_df, test_df


def tag_is_available(tag: str) -> bool:
    try:
        embeddings.load_embeddings("train", tag=tag)
        embeddings.load_embeddings("test", tag=tag)
        if tag == "finetuned" and not config.CHECKPOINT_PATH.exists():
            return False
        return True
    except FileNotFoundError:
        return False


def embed_uploaded_image(model, pil_image: Image.Image) -> np.ndarray:
    """Runs the same preprocessing + backbone used for the gallery on a single uploaded image."""
    transform = data_loader.build_transform()
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        feat = model(tensor)
        feat = nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()


def retrieve_and_show(query_embedding, query_id, query_category, query_image, k, train_df, tag, backend):
    emb_train, ids_train, cat_train = load_train_gallery(tag)
    backend_key = "exact" if backend == "Exact k-NN" else "faiss"
    index, _ = build_search_index(tag, backend_key)

    t0 = time.time()
    if backend_key == "exact":
        neighbor_ids, neighbor_sims, _ = retrieval.retrieve_top_k(
            query_embedding, np.array([query_id]), ids_train, k=k, index=index
        )
    else:
        neighbor_ids, neighbor_sims, _ = ann_index.search_ann_index(
            index, query_embedding, np.array([query_id]), ids_train, k=k, backend="faiss"
        )
    latency_ms = (time.time() - t0) * 1000
    st.caption(f"Query latency: {latency_ms:.2f} ms  ·  backend: {backend}  ·  model: "
               f"{'fine-tuned' if tag == 'finetuned' else 'pretrained baseline'}")

    gallery_lookup = evaluation.build_category_lookup(ids_train, cat_train)

    cols = st.columns(k + 1)
    with cols[0]:
        st.image(query_image, caption=f"QUERY\n{query_category or '(uploaded)'}", width='stretch')

    for rank, (pid, sim) in enumerate(zip(neighbor_ids[0], neighbor_sims[0]), start=1):
        with cols[rank]:
            if pid is None:
                st.write("—")
                continue
            match_row = train_df[train_df["id"].astype(str) == str(pid)]
            if len(match_row) == 0:
                st.write(f"#{rank}\n(metadata not found)")
                continue
            row = match_row.iloc[0]
            cat = gallery_lookup.get(pid, "?")
            marker = " ✅" if query_category and cat == query_category else ""
            st.image(row["image_path"], caption=f"#{rank}  sim={sim:.3f}{marker}\n{cat}", width='stretch')
            st.caption(ecommerce_search_links(cat))


def main():
    st.title("Visual Product Retrieval Demo")
    st.caption(
        "Shopping links under each result search that platform for the retrieved "
        "item's *category* — not direct links to the exact pictured item, since "
        "this dataset's images aren't tied to real product listings."
    )

    try:
        train_df, test_df = load_splits()
    except FileNotFoundError as e:
        st.error(
            "Couldn't find saved splits. From the project root, run:\n\n"
            "```\npython scripts/extract_embeddings.py --split all\n```\n\n"
            f"Details: {e}"
        )
        return

    baseline_available = tag_is_available("baseline")
    finetuned_available = tag_is_available("finetuned")

    if not baseline_available:
        st.error(
            "Couldn't find baseline embeddings. From the project root, run:\n\n"
            "```\npython scripts/extract_embeddings.py --split all\n```"
        )
        return

    available_model_labels = [label for label, tag in MODEL_OPTIONS.items()
                               if (tag != "finetuned" or finetuned_available)]
    model_label = st.selectbox("Model", available_model_labels)
    tag = MODEL_OPTIONS[model_label]

    if not finetuned_available:
        st.caption(
            "Fine-tuned model not found — showing baseline only. Run `scripts/train_triplet.py` "
            "then `scripts/extract_embeddings.py --split all --tag finetuned --checkpoint ...` to enable it."
        )
    elif tag == "finetuned":
        st.warning(
            "Note (Phase 2 finding): on this project's test set, the fine-tuned model scored "
            "*lower* than the pretrained baseline across every metric (see phase2_writeup.md) — "
            "included here for side-by-side comparison, not as the recommended model.",
            icon="⚠️",
        )

    backend = st.selectbox("Search backend", BACKEND_OPTIONS)
    if backend == "FAISS (approximate)":
        st.caption(
            f"Using FAISS `{config.FAISS_INDEX_TYPE}` — if this is an approximate index type "
            "(e.g. IndexIVFFlat), results may occasionally differ slightly from exact k-NN "
            "(see phase2_writeup.md RQ3 for the measured quality/speed trade-off)."
        )

    try:
        model = load_model(tag)
        build_search_index(tag, "exact" if backend == "Exact k-NN" else "faiss")
        emb_test, ids_test, cat_test = load_test_gallery(tag)
    except FileNotFoundError as e:
        st.error(f"Couldn't load embeddings for this selection. Details: {e}")
        return

    k = st.slider("Top-K", min_value=1, max_value=20, value=5)
    mode = st.radio("Query source", ["Pick a test-set image", "Upload my own image"], horizontal=True)

    if mode == "Pick a test-set image":
        state_key = f"query_idx_{tag}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        if st.button("🎲 Random test query"):
            st.session_state[state_key] = int(np.random.randint(0, len(test_df)))

        idx = st.session_state[state_key]
        row = test_df.iloc[idx]
        query_image = Image.open(row["image_path"])
        # test_df and emb_test were both produced in the same row order (no
        # shuffling during extraction), so index alignment holds directly.
        query_embedding = emb_test[idx:idx + 1]
        query_id = ids_test[idx]
        query_category = cat_test[idx]

        retrieve_and_show(query_embedding, query_id, query_category, query_image, k, train_df, tag, backend)

    else:
        uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            query_image = Image.open(uploaded)
            query_embedding = embed_uploaded_image(model, query_image)
            retrieve_and_show(query_embedding, "uploaded", None, query_image, k, train_df, tag, backend)
        else:
            st.info("Upload an image to see similar products.")


if __name__ == "__main__":
    main()