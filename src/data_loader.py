"""
data_loader.py
--------------
Assumes the dataset has already been placed by you at:
    data/images/<id>.jpg
    data/styles.csv

Loads raw metadata and defines the PyTorch Dataset/DataLoader used for
embedding extraction in both Phase 1 and Phase 2.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from . import config


def check_dataset_present() -> None:
    """Raises a clear error if the dataset hasn't been added yet."""
    if not config.STYLES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{config.STYLES_CSV_PATH} not found. Download your dataset from Kaggle "
            f"and place its metadata CSV at this path (renaming it to styles.csv), "
            f"and its image files directly under {config.IMAGES_DIR}/. "
            f"See README.md for details, and configs/phase1_config.yaml for mapping "
            f"a different column schema (dataset_id_col / dataset_category_col)."
        )
    if not config.IMAGES_DIR.exists() or not any(config.IMAGES_DIR.iterdir()):
        raise FileNotFoundError(
            f"{config.IMAGES_DIR} is missing or empty. Place the dataset's image "
            "files (e.g. 1001.jpg, 1002.jpg, ...) directly inside this folder."
        )


def load_raw_metadata(styles_csv_path: Path = config.STYLES_CSV_PATH) -> pd.DataFrame:
    """
    Loads the metadata CSV defensively (some source CSVs, e.g. Fashion
    Product Images' styles.csv, have a handful of malformed rows with extra
    commas that would otherwise break a strict parse), then renames the raw
    ID/category columns to the canonical "id"/"articleType" names every other
    module expects — see config.RAW_ID_COL / config.RAW_CATEGORY_COL.
    """
    df = pd.read_csv(styles_csv_path, on_bad_lines="skip")

    rename_map = {}
    if config.RAW_ID_COL != "id":
        if config.RAW_ID_COL not in df.columns:
            raise ValueError(
                f"config.RAW_ID_COL={config.RAW_ID_COL!r} not found in {styles_csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        rename_map[config.RAW_ID_COL] = "id"
    if config.RAW_CATEGORY_COL != "articleType":
        if config.RAW_CATEGORY_COL not in df.columns:
            raise ValueError(
                f"config.RAW_CATEGORY_COL={config.RAW_CATEGORY_COL!r} not found in {styles_csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        rename_map[config.RAW_CATEGORY_COL] = "articleType"

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"Renamed raw columns to canonical schema: {rename_map}")

    return df


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------
def build_transform(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


class ProductImageDataset(Dataset):
    """Returns (image_tensor, product_id, category, valid) for each row."""

    def __init__(self, dataframe: pd.DataFrame, transform: T.Compose):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        product_id = str(row["id"])
        category = str(row["articleType"])
        image_path = row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            valid = True
        except (UnidentifiedImageError, OSError):
            # Defensive fallback; preprocess.clean_metadata() should already
            # have removed unreadable images before this point.
            image_tensor = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            valid = False
        return image_tensor, product_id, category, valid


def build_dataloader(
    dataframe: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    shuffle: bool = False,
) -> DataLoader:
    dataset = ProductImageDataset(dataframe, build_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
