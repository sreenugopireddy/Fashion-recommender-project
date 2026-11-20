"""
embeddings.py
-------------
ResNet50 feature extractor.

Phase 1: pretrained ImageNet weights, fully frozen, eval() mode, no
fine-tuning — this is the exact-retrieval baseline.

Phase 2: the same architecture is reloaded with fine-tuned weights
(see finetune.py) to produce a second, comparable set of embeddings.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from . import config


def build_resnet50_backbone(
    device: torch.device = config.DEVICE,
    checkpoint_path: Optional[Path] = None,
    freeze: bool = True,
) -> nn.Module:
    """
    Loads ResNet50 with the classification head removed (-> 2048-d pooled
    features). If `checkpoint_path` is given, loads fine-tuned weights on
    top of the pretrained backbone (Phase 2). If `freeze=True`, disables
    gradients on all parameters (use freeze=False only when about to train).
    """
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.fc = nn.Identity()

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned weights from {checkpoint_path}")

    model.to(device)
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device = config.DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Runs the (frozen, eval-mode) backbone over a DataLoader.

    Returns: embeddings [N,2048] L2-normalized, ids [N], categories [N],
    elapsed_seconds, n_images.
    """
    all_embeddings, all_ids, all_categories = [], [], []
    n_images = 0
    start = time.time()

    for image_tensor, product_id, category, valid in loader:
        image_tensor = image_tensor.to(device, non_blocking=True)
        features = model(image_tensor)
        features = nn.functional.normalize(features, p=2, dim=1)
        all_embeddings.append(features.cpu().numpy())
        all_ids.extend(list(product_id))
        all_categories.extend(list(category))
        n_images += image_tensor.size(0)

    elapsed = time.time() - start
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    ids = np.array(all_ids)
    categories = np.array(all_categories)

    print(
        f"[{split_name}] extracted {embeddings.shape[0]} embeddings of dim "
        f"{embeddings.shape[1]} in {elapsed:.1f}s "
        f"({elapsed / max(n_images, 1) * 1000:.2f} ms/image)"
    )
    return embeddings, ids, categories, elapsed, n_images


def save_embeddings(split_name: str, embeddings: np.ndarray, ids: np.ndarray, categories: np.ndarray,
                     tag: str = "baseline") -> None:
    paths = config.embeddings_paths(split_name, tag=tag)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(paths["embeddings"], embeddings)
    np.save(paths["ids"], ids)
    np.save(paths["categories"], categories)


def load_embeddings(split_name: str, tag: str = "baseline") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = config.embeddings_paths(split_name, tag=tag)
    embeddings = np.load(paths["embeddings"], allow_pickle=True)
    ids = np.load(paths["ids"], allow_pickle=True)
    categories = np.load(paths["categories"], allow_pickle=True)
    return embeddings, ids, categories
