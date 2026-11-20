"""
triplet_sampling.py
--------------------
Phase 2: anchor/positive/negative triplet sampling based on articleType,
and the PyTorch Dataset that serves triplets for TripletMarginLoss
fine-tuning.

Two mining strategies are supported (config.TRIPLET_MINING):
  - "random":     positive = random other item with same articleType;
                  negative = random item from a different articleType.
  - "semi_hard":  negative is chosen, among a random candidate pool from
                  other categories, as the one whose *current* embedding
                  distance to the anchor is smallest (hardest-negative-in-pool)
                  while still being farther than the positive — requires the
                  current model to score candidates, so it's slower per step.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config


def build_category_index(df: pd.DataFrame) -> dict:
    """Maps articleType -> list of row indices, for fast anchor/positive/negative sampling."""
    return {cat: group.index.tolist() for cat, group in df.groupby("articleType")}


class TripletDataset(Dataset):
    """
    Yields (anchor_tensor, positive_tensor, negative_tensor) triplets sampled
    fresh each time __getitem__ is called (i.e. an "epoch" here means
    `config.TRIPLETS_PER_EPOCH` sampled triplets, not one pass over the data).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        n_triplets: int = config.TRIPLETS_PER_EPOCH,
        mining: str = config.TRIPLET_MINING,
        seed: int = config.SEED,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.n_triplets = n_triplets
        self.mining = mining
        self.rng = np.random.RandomState(seed)
        self.category_index = build_category_index(self.df)
        self.categories = list(self.category_index.keys())

        # articleTypes with fewer than 2 items can't provide a distinct positive
        self.valid_categories = [c for c in self.categories if len(self.category_index[c]) >= 2]
        if not self.valid_categories:
            raise ValueError("No articleType has >= 2 items; cannot form triplets.")

    def __len__(self) -> int:
        return self.n_triplets

    def _load_image(self, row_idx: int) -> torch.Tensor:
        row = self.df.iloc[row_idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image)

    def _sample_random_triplet(self) -> Tuple[int, int, int]:
        anchor_cat = self.rng.choice(self.valid_categories)
        anchor_idx, positive_idx = self.rng.choice(self.category_index[anchor_cat], size=2, replace=False)

        negative_cat = self.rng.choice([c for c in self.categories if c != anchor_cat])
        negative_idx = self.rng.choice(self.category_index[negative_cat])

        return int(anchor_idx), int(positive_idx), int(negative_idx)

    def __getitem__(self, _):
        # NOTE: the default training loop (finetune.py) always trains on
        # randomly-sampled triplets. `self.mining == "semi_hard"` is recorded
        # here for config completeness, but batched semi-hard mining is not
        # wired into the default loop — see finetune.select_semi_hard_negative()
        # for a starting point if you want to extend it.
        anchor_idx, positive_idx, negative_idx = self._sample_random_triplet()

        anchor = self._load_image(anchor_idx)
        positive = self._load_image(positive_idx)
        negative = self._load_image(negative_idx)

        return anchor, positive, negative
