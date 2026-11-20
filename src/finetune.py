"""
finetune.py
-----------
Phase 2: fine-tunes the ResNet50 backbone with a triplet margin loss to
learn embeddings that pull same-articleType items together and push
different-articleType items apart, improving on the frozen Phase 1
baseline.

Only the last `config.UNFREEZE_LAST_N_BLOCKS` residual blocks (layer4,
then layer3, ...) are unfrozen by default — fine-tuning the whole
network is unnecessary and far slower/more overfit-prone on a dataset
this size, and won't fit Colab Free's time/memory budget well.
"""

import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .data_loader import build_transform
from .triplet_sampling import TripletDataset

RESNET_BLOCK_ORDER = ["layer1", "layer2", "layer3", "layer4"]


def unfreeze_last_n_blocks(model: nn.Module, n: int = config.UNFREEZE_LAST_N_BLOCKS) -> None:
    """Freezes everything, then unfreezes the last `n` residual blocks (layer4 first)."""
    for p in model.parameters():
        p.requires_grad = False

    blocks_to_unfreeze = RESNET_BLOCK_ORDER[-n:] if n > 0 else []
    for block_name in blocks_to_unfreeze:
        block = getattr(model, block_name, None)
        if block is not None:
            for p in block.parameters():
                p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Unfroze blocks: {blocks_to_unfreeze} -> {n_trainable:,}/{n_total:,} trainable params")


def select_semi_hard_negative(
    model: nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative_pool: List[torch.Tensor],
    margin: float,
    device: torch.device,
) -> torch.Tensor:
    """
    EXTENSION POINT — not wired into the default training loop below.

    The current `finetune_with_triplet_loss()` loop uses config.TRIPLET_MINING
    only to pick the sampler (see triplet_sampling.py), and always trains on
    randomly-sampled triplets — true batched semi-hard mining (re-scoring a
    negative candidate pool per-anchor, every step) is meaningfully slower
    and is left out of Phase 1's scope. This function shows how to do it for
    a single anchor/positive/candidate-pool if you want to extend the loop:
    it returns the negative among `negative_pool` whose distance to the
    anchor is smallest while the triplet still violates the margin (falls
    back to the first candidate if none violate it).
    """
    with torch.no_grad():
        a_emb = nn.functional.normalize(model(anchor.unsqueeze(0).to(device)), p=2, dim=1)
        p_emb = nn.functional.normalize(model(positive.unsqueeze(0).to(device)), p=2, dim=1)
        d_ap = torch.norm(a_emb - p_emb, p=2).item()

        best_idx, best_dist = 0, float("inf")
        for i, neg in enumerate(negative_pool):
            n_emb = nn.functional.normalize(model(neg.unsqueeze(0).to(device)), p=2, dim=1)
            d_an = torch.norm(a_emb - n_emb, p=2).item()
            if d_ap < d_an < d_ap + margin and d_an < best_dist:
                best_idx, best_dist = i, d_an

    return negative_pool[best_idx]


def compute_validation_proxy_score(
    model: nn.Module,
    gallery_df: pd.DataFrame,
    query_df: pd.DataFrame,
    k: int = config.VAL_PROXY_K,
) -> float:
    """
    Lightweight validation signal computed after each epoch: extracts
    embeddings for a small, FIXED gallery/query subsample (same rows every
    epoch — only the model weights differ) and returns Category
    Consistency@k under Relevance Protocol A. This is what
    finetune_with_triplet_loss() uses to pick the BEST epoch's checkpoint
    instead of blindly keeping whichever epoch happened to run last.

    This is a proxy, not the real evaluation — the full Phase 2 comparison
    still runs separately via `scripts/evaluate_baseline.py --tag finetuned`
    on the complete, untouched test set.
    """
    from . import embeddings as embeddings_mod
    from . import evaluation as evaluation_mod
    from . import retrieval as retrieval_mod
    from .data_loader import build_dataloader

    was_training = model.training
    model.eval()

    gallery_loader = build_dataloader(gallery_df, shuffle=False, batch_size=config.FINETUNE_BATCH_SIZE)
    query_loader = build_dataloader(query_df, shuffle=False, batch_size=config.FINETUNE_BATCH_SIZE)

    emb_gallery, ids_gallery, cat_gallery, _, _ = embeddings_mod.extract_embeddings(
        model, gallery_loader, "val-proxy-gallery"
    )
    emb_query, ids_query, cat_query, _, _ = embeddings_mod.extract_embeddings(
        model, query_loader, "val-proxy-query"
    )

    knn_index, _ = retrieval_mod.build_knn_index(emb_gallery, k_max=k)
    neighbor_ids, _, _ = retrieval_mod.retrieve_top_k(emb_query, ids_query, ids_gallery, k=k, index=knn_index)
    gallery_lookup = evaluation_mod.build_category_lookup(ids_gallery, cat_gallery)
    score = evaluation_mod.category_consistency_at_k(neighbor_ids, gallery_lookup, cat_query, k)

    if was_training:
        model.train()
    return score


def finetune_with_triplet_loss(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    device: torch.device = config.DEVICE,
) -> Tuple[nn.Module, pd.DataFrame]:
    """
    Runs Phase 2 triplet-loss fine-tuning. Returns (fine_tuned_model, training_log_df).

    If `val_df` is provided, a lightweight validation proxy score (Category
    Consistency@k on a small fixed subsample — see
    compute_validation_proxy_score) is computed after every epoch, and
    config.CHECKPOINT_PATH always holds the BEST epoch by that score, not
    just the last one. This matters: triplet-loss training loss going down
    does NOT guarantee retrieval quality goes up (a noisy random-triplet
    sample can make embeddings overfit to those specific triplets while
    drifting away from a more generally useful geometry) — without this,
    you could silently ship a fine-tuned model that's worse than the
    pretrained baseline. The most recent epoch's weights are always also
    kept separately at config.LAST_EPOCH_CHECKPOINT_PATH for comparison/debugging.

    If `val_df` is omitted, behavior matches the original version: the last
    epoch's weights are saved as the checkpoint (no best-epoch selection).

    Either way, a checkpoint is saved after EVERY epoch (not just at the
    end), and live per-batch progress is printed — a full run can take a
    long time on CPU, and losing a completed epoch to an interruption
    (Ctrl+C, laptop sleep, crash) wastes real compute.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    config.set_global_seed()

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    unfreeze_last_n_blocks(model, config.UNFREEZE_LAST_N_BLOCKS)

    criterion = nn.TripletMarginLoss(margin=config.TRIPLET_MARGIN, p=2)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.FINETUNE_LR)

    transform = build_transform()
    dataset = TripletDataset(train_df, transform, n_triplets=config.TRIPLETS_PER_EPOCH,
                              mining=config.TRIPLET_MINING)
    loader = DataLoader(dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True,
                         num_workers=config.NUM_WORKERS)
    n_batches_total = len(loader)

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed subsample for the validation proxy — same rows every epoch, so
    # score changes reflect the model, not sampling noise.
    use_val_proxy = val_df is not None and len(val_df) > 0
    if use_val_proxy:
        rng = np.random.RandomState(config.SEED)
        gallery_size = min(config.VAL_PROXY_GALLERY_SIZE, len(train_df))
        query_size = min(config.VAL_PROXY_QUERY_SIZE, len(val_df))
        proxy_gallery_df = train_df.sample(n=gallery_size, random_state=rng).reset_index(drop=True)
        proxy_query_df = val_df.sample(n=query_size, random_state=rng).reset_index(drop=True)
        print(f"Validation proxy: {gallery_size} gallery items, {query_size} queries, "
              f"CategoryConsistency@{config.VAL_PROXY_K} (fixed subsample every epoch)")

    def save_model_to(path, was_training_before):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        torch.save(model.state_dict(), path)
        if was_training_before:
            model.train()
            for block_name in RESNET_BLOCK_ORDER[-config.UNFREEZE_LAST_N_BLOCKS:]:
                block = getattr(model, block_name, None)
                if block is not None:
                    for p in block.parameters():
                        p.requires_grad_(True)

    log_rows = []
    best_score = -1.0
    best_epoch = None
    try:
        for epoch in range(1, config.FINETUNE_EPOCHS + 1):
            model.train()
            epoch_start = time.time()
            running_loss, n_batches = 0.0, 0

            for batch_idx, (anchor, positive, negative) in enumerate(loader, start=1):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                optimizer.zero_grad()
                a_emb = nn.functional.normalize(model(anchor), p=2, dim=1)
                p_emb = nn.functional.normalize(model(positive), p=2, dim=1)
                n_emb = nn.functional.normalize(model(negative), p=2, dim=1)

                loss = criterion(a_emb, p_emb, n_emb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

                elapsed = time.time() - epoch_start
                avg_batch_time = elapsed / batch_idx
                eta_s = avg_batch_time * (n_batches_total - batch_idx)
                print(
                    f"\rEpoch {epoch}/{config.FINETUNE_EPOCHS}  "
                    f"batch {batch_idx}/{n_batches_total}  "
                    f"loss={running_loss / n_batches:.4f}  "
                    f"elapsed={elapsed:.0f}s  eta={eta_s:.0f}s",
                    end="", flush=True,
                )

            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / max(n_batches, 1)
            print(f"\nEpoch {epoch}/{config.FINETUNE_EPOCHS} complete — loss: {avg_loss:.4f} — {epoch_time:.1f}s")

            log_row = {"epoch": epoch, "avg_triplet_loss": avg_loss, "epoch_time_s": epoch_time}

            # Always keep the most recent epoch's weights, regardless of score.
            save_model_to(config.LAST_EPOCH_CHECKPOINT_PATH, was_training_before=True)

            if use_val_proxy:
                val_score = compute_validation_proxy_score(model, proxy_gallery_df, proxy_query_df)
                log_row["val_proxy_category_consistency"] = val_score
                is_best = val_score > best_score
                if is_best and best_score >= 0:
                    status = f"(best so far, was {best_score:.4f})"
                elif is_best:
                    status = "(first epoch, sets initial best)"
                elif val_score == best_score:
                    status = f"(tied with best epoch {best_epoch}'s {best_score:.4f} — keeping epoch {best_epoch} as best)"
                else:
                    status = f"(WORSE than best epoch {best_epoch}'s {best_score:.4f} — not saving as best)"
                print(f"Validation proxy CategoryConsistency@{config.VAL_PROXY_K}: {val_score:.4f}  {status}")
                if is_best:
                    best_score, best_epoch = val_score, epoch
                    save_model_to(config.CHECKPOINT_PATH, was_training_before=True)
                    print(f"New best checkpoint (epoch {epoch}) saved -> {config.CHECKPOINT_PATH}")
            else:
                # No validation set given — preserve the original behavior:
                # the checkpoint is just whatever the latest epoch produced.
                save_model_to(config.CHECKPOINT_PATH, was_training_before=True)
                print(f"Checkpoint saved after epoch {epoch} -> {config.CHECKPOINT_PATH}")

            log_rows.append(log_row)
            pd.DataFrame(log_rows).to_csv(config.TRAINING_LOG_PATH, index=False)

    except KeyboardInterrupt:
        print("\n\nInterrupted — the checkpoint(s) and log from the last COMPLETED epoch "
              "(if any) are already saved on disk. No partial/mid-epoch progress is kept.")
        if not log_rows:
            print("No epoch finished before the interrupt, so no checkpoint exists yet.")
        raise

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log_df = pd.DataFrame(log_rows)
    if use_val_proxy:
        print(f"\nBest epoch: {best_epoch} (val proxy CategoryConsistency@{config.VAL_PROXY_K} = {best_score:.4f})")
        print(f"Best checkpoint: {config.CHECKPOINT_PATH}")
        print(f"Last epoch's checkpoint (may differ from best): {config.LAST_EPOCH_CHECKPOINT_PATH}")
    else:
        print(f"Saved fine-tuned checkpoint to {config.CHECKPOINT_PATH}")
    print(f"Saved training log to {config.TRAINING_LOG_PATH}")

    return model, log_df