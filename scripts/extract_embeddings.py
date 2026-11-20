"""
scripts/extract_embeddings.py
------------------------------
CLI entry point for Phase 1 (and re-usable for Phase 2's fine-tuned model):
cleans metadata (first run only), creates/reuses the train/val/test split,
then extracts and saves embeddings for the requested split(s).

Usage:
    python scripts/extract_embeddings.py --split train
    python scripts/extract_embeddings.py --split all
    python scripts/extract_embeddings.py --split all --tag finetuned --checkpoint results/checkpoints/resnet50_triplet_finetuned.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, data_loader, embeddings, preprocess  # noqa: E402


def get_or_create_splits():
    if config.TRAIN_SPLIT_PATH.exists() and config.VAL_SPLIT_PATH.exists() and config.TEST_SPLIT_PATH.exists():
        print("Using existing splits under data/splits/.")
        return preprocess.load_splits()

    print("No existing splits found — cleaning metadata and creating a new stratified split.")
    data_loader.check_dataset_present()
    df_raw = data_loader.load_raw_metadata()
    df_clean, clean_report = preprocess.clean_metadata(df_raw)
    print("Cleaning report:", clean_report)

    df_final, small_cat_report = preprocess.drop_small_categories(df_clean)
    print("Small-category report:", small_cat_report)

    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(config.METADATA_CLEAN_PATH, index=False)
    print(f"Saved cleaned metadata to {config.METADATA_CLEAN_PATH}")

    train_df, val_df, test_df = preprocess.stratified_split(df_final)
    preprocess.save_splits(train_df, val_df, test_df)
    print(f"Saved splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet50 embeddings for one or more splits.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--tag", default="baseline",
                         help="Embedding set name, e.g. 'baseline' (Phase 1) or 'finetuned' (Phase 2).")
    parser.add_argument("--checkpoint", default=None,
                         help="Optional fine-tuned checkpoint path (Phase 2). Omit for the pretrained baseline.")
    args = parser.parse_args()

    config.ensure_directories()
    config.set_global_seed()

    train_df, val_df, test_df = get_or_create_splits()
    splits = {"train": train_df, "val": val_df, "test": test_df}
    target_splits = list(splits.keys()) if args.split == "all" else [args.split]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    model = embeddings.build_resnet50_backbone(checkpoint_path=checkpoint_path, freeze=True)

    for split_name in target_splits:
        df = splits[split_name]
        loader = data_loader.build_dataloader(df, shuffle=False)
        emb, ids, cats, elapsed, n_images = embeddings.extract_embeddings(model, loader, split_name)
        embeddings.save_embeddings(split_name, emb, ids, cats, tag=args.tag)

    print("Done.")


if __name__ == "__main__":
    main()
