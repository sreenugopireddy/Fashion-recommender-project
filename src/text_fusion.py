"""
text_fusion.py
--------------
Phase 2: encodes product titles (productDisplayName) as text embeddings
and fuses them with image embeddings to test whether fusion improves
category consistency / reduces cross-category retrieval errors.

Two text-encoding backends (config.TEXT_EMBEDDING_METHOD):
  - "tfidf": TF-IDF + TruncatedSVD down to a fixed dimension. No extra
             model download needed — safest default for Colab Free.
  - "sbert": sentence-transformers (e.g. all-MiniLM-L6-v2). Requires the
             `sentence-transformers` package (see requirements.txt) and a
             one-time model download.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config


def encode_titles_tfidf(
    titles: pd.Series,
    max_features: int = config.TFIDF_MAX_FEATURES,
    seed: int = config.SEED,
) -> np.ndarray:
    """TF-IDF -> TruncatedSVD to a fixed dimension, L2-normalized."""
    titles_filled = titles.fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features=max_features * 4, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(titles_filled)

    n_components = min(max_features, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(tfidf_matrix)

    return normalize(reduced, norm="l2", axis=1).astype(np.float32)


def encode_titles_sbert(titles: pd.Series, model_name: str = config.SBERT_MODEL_NAME) -> np.ndarray:
    """Sentence-BERT embeddings, L2-normalized. Requires sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    titles_filled = titles.fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(titles_filled, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def encode_titles(titles: pd.Series, method: str = config.TEXT_EMBEDDING_METHOD) -> np.ndarray:
    if method == "tfidf":
        return encode_titles_tfidf(titles)
    elif method == "sbert":
        return encode_titles_sbert(titles)
    raise ValueError(f"Unknown TEXT_EMBEDDING_METHOD: {method!r} (expected 'tfidf' or 'sbert')")


def fuse_image_text_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_weight: float = config.FUSION_IMAGE_WEIGHT,
    text_weight: float = config.FUSION_TEXT_WEIGHT,
) -> np.ndarray:
    """
    Weighted concatenation fusion: scales each modality by its weight, then
    concatenates and L2-normalizes the result. This keeps both modalities'
    original (already L2-normalized) directions intact while controlling
    their relative influence on the fused cosine similarity.

    NOTE: image_weight/text_weight should be chosen on the *validation*
    split only (see README's Research Validity notes) — never tuned on test.
    """
    assert image_embeddings.shape[0] == text_embeddings.shape[0], \
        "image and text embeddings must have the same number of rows"

    weighted_image = image_embeddings * image_weight
    weighted_text = text_embeddings * text_weight
    fused = np.concatenate([weighted_image, weighted_text], axis=1)
    return normalize(fused, norm="l2", axis=1).astype(np.float32)


def build_fused_embeddings(
    image_embeddings: np.ndarray,
    titles: pd.Series,
    method: str = config.TEXT_EMBEDDING_METHOD,
    image_weight: float = config.FUSION_IMAGE_WEIGHT,
    text_weight: float = config.FUSION_TEXT_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: returns (fused_embeddings, text_embeddings)."""
    text_embeddings = encode_titles(titles, method=method)
    fused = fuse_image_text_embeddings(image_embeddings, text_embeddings, image_weight, text_weight)
    return fused, text_embeddings
