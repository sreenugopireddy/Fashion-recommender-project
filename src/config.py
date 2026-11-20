"""
config.py
---------
Central configuration for both phases. All paths resolve relative to
PROJECT_ROOT so the project runs unmodified after being uploaded to
Colab or run locally, as long as `data/images/` and `data/styles.csv`
exist. Values can be overridden via configs/phase1_config.yaml and
configs/phase2_config.yaml without touching code.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# PROJECT_ROOT works whether config.py sits in the project root or in a
# subfolder such as src/.
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

PHASE1_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_config.yaml"
PHASE2_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase2_config.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


_p1 = _load_yaml(PHASE1_CONFIG_PATH)
_p2 = _load_yaml(PHASE2_CONFIG_PATH)


def _get(d: dict, key: str, default):
    return d.get(key, default)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = int(_get(_p1, "seed", 42))


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset layout
#   data/images/<id>.jpg
#   data/styles.csv
# In Colab you can keep the dataset on fast local disk by setting an
# environment variable before running, e.g. in a notebook cell:
#   import os; os.environ["ASTHRA_DATA_DIR"] = "/content/data"
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("ASTHRA_DATA_DIR", PROJECT_ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
STYLES_CSV_PATH = DATA_DIR / "styles.csv"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

METADATA_CLEAN_PATH = METADATA_DIR / "metadata_clean.csv"
TRAIN_SPLIT_PATH = SPLITS_DIR / "train_ids.csv"
VAL_SPLIT_PATH = SPLITS_DIR / "val_ids.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test_ids.csv"

# ---------------------------------------------------------------------------
# Raw column-name mapping
# Fashion Product Images (Small): id, articleType            -> defaults below
# Clothing dataset (full, agrigorev): image, label            -> set in
#   configs/phase1_config.yaml: dataset_id_col / dataset_category_col
# ---------------------------------------------------------------------------
RAW_ID_COL = _get(_p1, "dataset_id_col", "id")
RAW_CATEGORY_COL = _get(_p1, "dataset_category_col", "articleType")
HAS_TITLE_COLUMN = bool(_get(_p1, "dataset_has_title_column", True))  # False for clothing-dataset-full (no text field)

MIN_SAMPLES_PER_CATEGORY = int(_get(_p1, "min_samples_per_category", 10))
TRAIN_FRACTION = float(_get(_p1, "train_fraction", 0.70))
VAL_FRACTION = float(_get(_p1, "val_fraction", 0.15))
TEST_FRACTION = float(_get(_p1, "test_fraction", 0.15))
assert abs((TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION) - 1.0) < 1e-6

# ---------------------------------------------------------------------------
# Results / outputs
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
INDEXES_DIR = RESULTS_DIR / "indexes"
QUALITATIVE_DIR = RESULTS_DIR / "qualitative_examples"

ALL_DIRS = [
    DATA_DIR, IMAGES_DIR, METADATA_DIR, SPLITS_DIR, PROCESSED_DATA_DIR,
    RESULTS_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR, CHECKPOINTS_DIR, INDEXES_DIR, QUALITATIVE_DIR,
]


def ensure_directories() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image preprocessing / embedding (shared, Phase 1 + Phase 2)
# ---------------------------------------------------------------------------
IMAGE_SIZE = int(_get(_p1, "image_size", 224))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EMBEDDING_DIM = 2048

BATCH_SIZE = int(_get(_p1, "batch_size", 128))
NUM_WORKERS = int(_get(_p1, "num_workers", 2))

# ---------------------------------------------------------------------------
# Retrieval / evaluation (shared, Phase 1 + Phase 2)
# ---------------------------------------------------------------------------
K_VALUES = list(_get(_p1, "k_values", [1, 5, 10]))
K_MAX = int(_get(_p1, "k_max", 20))
MAP_K_VALUES = list(_get(_p1, "map_k_values", [10, 20]))
CATEGORY_CONSISTENCY_K_VALUES = list(_get(_p1, "category_consistency_k_values", [5, 10]))
RELEVANCE_PROTOCOL = _get(_p1, "relevance_protocol", "A")  # "A" = articleType match


def embeddings_paths(split: str, tag: str = "baseline") -> dict:
    """tag='baseline' -> Phase 1 pretrained embeddings; tag='finetuned' -> Phase 2."""
    prefix = PROCESSED_DATA_DIR / f"{tag}_{split}"
    return {
        "embeddings": Path(f"{prefix}_embeddings.npy"),
        "ids": Path(f"{prefix}_ids.npy"),
        "categories": Path(f"{prefix}_categories.npy"),
    }


METRICS_CSV_PATH = METRICS_DIR / "metrics_baseline_exact_knn.csv"
LATENCY_LOG_PATH = LOGS_DIR / "latency_log.csv"

# ---------------------------------------------------------------------------
# Phase 2 — triplet fine-tuning
# ---------------------------------------------------------------------------
TRIPLET_MARGIN = float(_get(_p2, "triplet_margin", 0.3))
TRIPLET_MINING = _get(_p2, "triplet_mining", "random")  # "random" or "semi_hard"
FINETUNE_EPOCHS = int(_get(_p2, "finetune_epochs", 4))   # changed from 5 to 4
FINETUNE_LR = float(_get(_p2, "finetune_lr", 1e-5))
FINETUNE_BATCH_SIZE = int(_get(_p2, "finetune_batch_size", 32))
TRIPLETS_PER_EPOCH = int(_get(_p2, "triplets_per_epoch", 5000))
UNFREEZE_LAST_N_BLOCKS = int(_get(_p2, "unfreeze_last_n_blocks", 1))  # fine-tune only layer4 by default
CHECKPOINT_PATH = CHECKPOINTS_DIR / "resnet50_triplet_finetuned.pt"          # best-epoch checkpoint (used downstream)
LAST_EPOCH_CHECKPOINT_PATH = CHECKPOINTS_DIR / "resnet50_triplet_last_epoch.pt"  # most recent epoch, for debugging/resuming
TRAINING_LOG_PATH = LOGS_DIR / "phase2_training_log.csv"

VAL_PROXY_GALLERY_SIZE = int(_get(_p2, "val_proxy_gallery_size", 500))
VAL_PROXY_QUERY_SIZE = int(_get(_p2, "val_proxy_query_size", 100))
VAL_PROXY_K = int(_get(_p2, "val_proxy_k", 5))

# ---------------------------------------------------------------------------
# Phase 2 — text fusion
# ---------------------------------------------------------------------------
TEXT_EMBEDDING_METHOD = _get(_p2, "text_embedding_method", "tfidf")  # "tfidf" or "sbert"
SBERT_MODEL_NAME = _get(_p2, "sbert_model_name", "all-MiniLM-L6-v2")
TFIDF_MAX_FEATURES = int(_get(_p2, "tfidf_max_features", 512))
FUSION_IMAGE_WEIGHT = float(_get(_p2, "fusion_image_weight", 0.7))
FUSION_TEXT_WEIGHT = float(_get(_p2, "fusion_text_weight", 0.3))

# ---------------------------------------------------------------------------
# Phase 2 — approximate nearest neighbor
# ---------------------------------------------------------------------------
ANN_BACKEND = _get(_p2, "ann_backend", "faiss")  # "faiss" or "annoy"
FAISS_INDEX_TYPE = _get(_p2, "faiss_index_type", "IndexFlatIP")  # exact-equivalent baseline for ANN comparison
FAISS_NLIST = int(_get(_p2, "faiss_nlist", 100))       # used only for IVF index types
FAISS_NPROBE = int(_get(_p2, "faiss_nprobe", 10))      # used only for IVF index types
ANNOY_N_TREES = int(_get(_p2, "annoy_n_trees", 50))
ANN_INDEX_PATH = INDEXES_DIR / "ann_index"