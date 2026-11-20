"""
preprocess.py
-------------
Metadata cleaning and reproducible stratified train/val/test splitting.
Shared by Phase 1 and Phase 2 (the split is created once and reused).
"""

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from . import config


def clean_metadata(df_raw: pd.DataFrame, images_dir: Path = config.IMAGES_DIR) -> Tuple[pd.DataFrame, dict]:
    """
    Cleans raw styles.csv metadata:
      - keeps rows with non-empty id / articleType
      - resolves image_path and drops rows whose image file is missing
      - drops duplicate product IDs
      - drops rows whose image file cannot be opened (corrupt)

    Returns (cleaned_df, report) where report is a dict of counts for logging.
    """
    df = df_raw.copy()
    n_original = len(df)

    for c in ("id", "articleType"):
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in styles.csv. "
                              f"Available columns: {list(df.columns)}")

    # NOTE: pandas' Series.astype(str) does NOT reliably stringify actual
    # NaN cells across pandas versions (in pandas 3.x it can leave a real
    # NaN float in place, which then breaks .str.strip()/.str.lower()).
    # fillna("") first so every cell is guaranteed to be a real string
    # before any .str accessor is used.
    df["id"] = df["id"].fillna("").astype(str).str.strip()
    df["articleType"] = df["articleType"].fillna("").astype(str).str.strip()
    df = df[(df["id"] != "") & (df["id"].str.lower() != "nan")]
    df = df[(df["articleType"] != "") & (df["articleType"].str.lower() != "nan")]

    df["image_path"] = df["id"].apply(lambda pid: str(Path(images_dir) / f"{pid}.jpg"))
    image_exists = df["image_path"].apply(os.path.exists).astype(bool)  # explicit bool dtype —
    # pandas' Series.sum() on an EMPTY object-dtype Series can return '' instead of 0,
    # which then crashes int('') downstream with a confusing error. Casting to bool
    # up front avoids that regardless of how many rows survive.
    n_missing_image = int((~image_exists).sum())

    if image_exists.sum() == 0:
        sample_paths = df["image_path"].head(5).tolist()
        raise FileNotFoundError(
            "None of the constructed image paths exist on disk — every row would be "
            "dropped. This almost always means the images are nested one folder "
            "deeper than expected (e.g. data/images/images_compressed/<id>.jpg "
            "instead of data/images/<id>.jpg flat), or the CSV's ID column doesn't "
            "match actual filenames. First 5 paths checked:\n  "
            + "\n  ".join(sample_paths)
            + f"\nVerify these against the actual contents of {images_dir}."
        )

    df = df[image_exists].reset_index(drop=True)

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    n_duplicates = n_before_dedup - len(df)

    def is_readable(path: str) -> bool:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    readable_mask = df["image_path"].apply(is_readable).astype(bool)
    n_corrupt = int((~readable_mask).sum())
    df = df[readable_mask].reset_index(drop=True)

    report = {
        "original_records": n_original,
        "removed_missing_image": n_missing_image,
        "removed_duplicate_id": n_duplicates,
        "removed_corrupt_image": n_corrupt,
        "final_cleaned_records": len(df),
        "retained_categories": int(df["articleType"].nunique()),
    }
    return df, report


def drop_small_categories(df: pd.DataFrame, min_samples: int = config.MIN_SAMPLES_PER_CATEGORY) -> Tuple[pd.DataFrame, dict]:
    """Drops rows whose articleType has fewer than `min_samples` examples (can't be stratified)."""
    category_counts = df["articleType"].value_counts()
    small_categories = category_counts[category_counts < min_samples].index.tolist()
    n_dropped = int(df["articleType"].isin(small_categories).sum())
    filtered = df[~df["articleType"].isin(small_categories)].reset_index(drop=True)

    report = {
        "min_samples_per_category": min_samples,
        "num_small_categories_dropped": len(small_categories),
        "small_categories": small_categories,
        "rows_dropped_for_small_category": n_dropped,
        "final_size": len(filtered),
        "final_category_count": int(filtered["articleType"].nunique()),
    }
    return filtered, report


def stratified_split(
    df: pd.DataFrame,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits df into train/val/test, stratified by articleType, seeded, verified disjoint."""
    val_test_fraction = val_fraction + test_fraction
    train_df, temp_df = train_test_split(
        df, test_size=val_test_fraction, stratify=df["articleType"], random_state=seed,
    )
    relative_test_fraction = test_fraction / val_test_fraction
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_fraction, stratify=temp_df["articleType"], random_state=seed,
    )

    train_ids, val_ids, test_ids = set(train_df["id"]), set(val_df["id"]), set(test_df["id"])
    assert train_ids.isdisjoint(val_ids), "train/val split overlap detected"
    assert train_ids.isdisjoint(test_ids), "train/test split overlap detected"
    assert val_ids.isdisjoint(test_ids), "val/test split overlap detected"

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(config.VAL_SPLIT_PATH, index=False)
    test_df.to_csv(config.TEST_SPLIT_PATH, index=False)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(config.VAL_SPLIT_PATH)
    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    return train_df, val_df, test_df