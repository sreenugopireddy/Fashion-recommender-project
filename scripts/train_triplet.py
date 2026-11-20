"""
scripts/train_triplet.py
--------------------------
Phase 2: fine-tunes ResNet50 with triplet loss on the train split.
Requires the same data/splits/*.csv produced by
scripts/extract_embeddings.py (run that first, or scripts/evaluate_baseline.py's
prerequisite splitting, to create them).

Usage:
    python scripts/train_triplet.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, finetune, preprocess  # noqa: E402


def main():
    config.ensure_directories()

    if not config.TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_SPLIT_PATH} not found. Run "
            "`python scripts/extract_embeddings.py --split train` first to create the splits."
        )

    train_df, val_df, _ = preprocess.load_splits()
    print(f"Fine-tuning on {len(train_df)} training items "
          f"({config.FINETUNE_EPOCHS} epochs x {config.TRIPLETS_PER_EPOCH} triplets/epoch)")

    _, log_df = finetune.finetune_with_triplet_loss(train_df, val_df=val_df)
    print(log_df.to_string(index=False))
    print(f"Best checkpoint: {config.CHECKPOINT_PATH}")
    print(f"Last-epoch checkpoint (for comparison/debugging): {config.LAST_EPOCH_CHECKPOINT_PATH}")
    print(f"Training log: {config.TRAINING_LOG_PATH}")
    print("\nNext step: python scripts/extract_embeddings.py --split all "
          f"--tag finetuned --checkpoint {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()