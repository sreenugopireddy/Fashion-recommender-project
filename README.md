# Deep Learning-Based Image Product Recommendation System Using ResNet50 and Visual Similarity

Content-based visual product retrieval on fashion e-commerce images. Given a
query product image, retrieve visually and categorically similar catalog
items. Two-phase research project:

- **Phase 1** — pretrained ResNet50 embeddings + exact k-NN baseline.
- **Phase 2** — triplet-loss fine-tuning, image+text fusion, and FAISS/ANNOY
  approximate search benchmarked against exact k-NN.

## Research questions

1. How well does a pretrained ResNet50 perform for visual product
   retrieval (Precision@K, Recall@K, mAP)?
2. Does triplet-loss fine-tuning improve retrieval metrics over the
   pretrained baseline, on the same test set and protocol?
3. Does fusing image and product-title text features improve category
   consistency and reduce cross-category retrieval errors? **(dataset-
   dependent — see below)**
4. How does FAISS/ANNOY approximate search trade retrieval quality for
   query latency compared with exact k-NN?

> **If your dataset has no product-title/description text** (e.g. the
> Clothing dataset — full, high resolution, by agrigorev — which only has
> a category `label` and a `kids` boolean, no title field), RQ3 cannot be
> answered honestly: encoding the category label itself as "text" and
> fusing it would leak the evaluation target into the embedding and
> artificially inflate category-consistency numbers. In that case, set
> `dataset_has_title_column: false` in `configs/phase1_config.yaml` and
> `scripts/evaluate_phase2.py` will skip fusion automatically, saving only
> image-only (fine-tuned) metrics. Document this as a scope limitation
> rather than working around it — see "Known scope limits" below.

No metric, accuracy, or improvement claim in this repo is hardcoded —
every number in `results/` is produced by actually running the code.

## Using a different dataset (column mapping)

The codebase's internal modules (`preprocess.py`, `evaluation.py`,
`retrieval.py`, `triplet_sampling.py`, etc.) all use two canonical column
names: `id` and `articleType`. Rather than editing every module for a new
dataset's actual column names, `data_loader.load_raw_metadata()` renames
your raw CSV's columns to these canonical names right after loading, based
on `configs/phase1_config.yaml`:

```yaml
dataset_id_col: "image"              # your CSV's ID column
dataset_category_col: "label"        # your CSV's category column
dataset_has_title_column: false      # true only if a title/description field exists
```

Regardless of the source dataset's own file/folder naming, place its files
at the fixed paths this project expects — rename/copy on the way in:

```
data/images/<id>.jpg    # copy the dataset's image files here (flat, no subfolders)
data/styles.csv         # copy/rename the dataset's metadata CSV to this path
```

| Dataset | Real columns | Config values |
|---|---|---|
| Fashion Product Images (Small) | `id`, `articleType`, `productDisplayName`, ... | `dataset_id_col: "id"`, `dataset_category_col: "articleType"`, `dataset_has_title_column: true` |
| Clothing dataset (full, high resolution) — agrigorev | `image`, `sender_id`, `label`, `kids` | `dataset_id_col: "image"`, `dataset_category_col: "label"`, `dataset_has_title_column: false` |

Verify your actual downloaded filenames/columns before relying on the
table above — dataset packaging can change between Kaggle re-uploads.
`data_loader.load_raw_metadata()` raises a clear error naming the
available columns if `dataset_id_col`/`dataset_category_col` don't match.

## Directory structure

```
fashion_recommender_project/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── phase1_config.yaml       # seed, split fractions, batch size, K values
│   └── phase2_config.yaml       # triplet/fusion/ANN hyperparameters
├── data/
│   ├── images/                  # YOU add: <id>.jpg files from Kaggle
│   ├── styles.csv                # YOU add: Kaggle metadata file
│   ├── metadata/                 # generated: cleaned metadata CSV
│   ├── splits/                   # generated: train/val/test ID CSVs
│   └── processed/                # generated: saved embeddings (.npy)
├── notebooks/
│   ├── phase1_baseline.ipynb     # thin wrapper — runs Phase 1 end-to-end
│   └── phase2_improvements.ipynb # thin wrapper — runs Phase 2 end-to-end
├── scripts/                      # CLI entry points (calls into src/)
│   ├── extract_embeddings.py
│   ├── evaluate_baseline.py
│   ├── train_triplet.py
│   ├── build_ann_index.py
│   └── evaluate_phase2.py
├── src/                          # all real logic lives here
│   ├── config.py                 # paths + hyperparameters (reads configs/*.yaml)
│   ├── data_loader.py             # dataset checks, PyTorch Dataset/DataLoader
│   ├── preprocess.py              # metadata cleaning, stratified split
│   ├── embeddings.py              # ResNet50 feature extraction
│   ├── retrieval.py               # exact k-NN (sklearn, cosine)
│   ├── evaluation.py              # Protocol A relevance + all metrics
│   ├── visualization.py           # plots + qualitative query grids
│   ├── triplet_sampling.py        # Phase 2: anchor/positive/negative sampling
│   ├── finetune.py                # Phase 2: triplet-loss fine-tuning loop
│   ├── text_fusion.py             # Phase 2: TF-IDF/SBERT + image+text fusion
│   └── ann_index.py               # Phase 2: FAISS/ANNOY build + search
└── results/                       # gitignored — all generated outputs
    ├── metrics/                   # *.csv metric tables
    ├── figures/                   # *.png plots
    ├── logs/                      # latency logs, training logs
    ├── checkpoints/               # fine-tuned model weights (.pt)
    ├── indexes/                   # saved ANN index files (if you persist them)
    └── qualitative_examples/      # query -> Top-K visualization grids
```

## Setup

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Google Colab, skip the venv step and just run `!pip install -q -r requirements.txt`
in a cell (already included in both notebooks).

> **faiss on Colab:** `faiss-cpu` (in requirements.txt) works fine for this
> dataset's scale. If you want GPU-accelerated FAISS, swap it for the
> appropriate `faiss-gpu-cu*` build matching Colab's CUDA version.

### 2. Dataset

Download **Fashion Product Images (Small)** from Kaggle:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

After downloading and unzipping, arrange the files so that:

```
data/
├── images/
│   ├── 1000.jpg
│   ├── 1001.jpg
│   └── ...
└── styles.csv
```

`styles.csv` must contain at least the columns `id`, `articleType`, and
`productDisplayName` (the dataset's default columns — `subCategory`,
`baseColour`, `season`, `usage`, etc. — are also used where relevant).

`src/data_loader.check_dataset_present()` will raise a clear error at the
start of either notebook/script if this isn't set up correctly yet.

## Running the project

### Option A — Notebooks (Colab)

1. Upload this whole folder to Google Drive (or directly into a Colab
   session), with the dataset already placed as above.
2. Open `notebooks/phase1_baseline.ipynb` in Colab, update `PROJECT_DIR` in
   the first code cell to your actual path, and run all cells.
3. Inspect `results/metrics/metrics_baseline_exact_knn.csv` and the plots in
   `results/figures/`. **Confirm you're satisfied with Phase 1 before
   continuing** — Phase 2 fine-tuning is only meaningful as a controlled
   comparison against this exact baseline.
4. Open `notebooks/phase2_improvements.ipynb`, update `PROJECT_DIR` the same
   way, and run all cells.

### Option B — CLI scripts (local or Colab terminal)

```bash
# Phase 1
python scripts/extract_embeddings.py --split all           # cleans metadata, splits, extracts baseline embeddings
python scripts/evaluate_baseline.py                          # exact k-NN + Phase 1 metrics/plots

# Phase 2
python scripts/train_triplet.py                              # triplet-loss fine-tuning -> checkpoint
python scripts/extract_embeddings.py --split all \\
    --tag finetuned \\
    --checkpoint results/checkpoints/resnet50_triplet_finetuned.pt  # fine-tuned embeddings
python scripts/evaluate_phase2.py                             # image-only vs. fused (RQ3)
python scripts/build_ann_index.py --tag finetuned --k 10       # exact vs. FAISS/ANNOY (RQ4)
```

## Results — what to expect where

| File | Produced by | Contents |
|---|---|---|
| `data/metadata/metadata_clean.csv` | Phase 1 | cleaned metadata after dropping missing/duplicate/corrupt/small-category rows |
| `data/splits/{train,val,test}_ids.csv` | Phase 1 | the one stratified 70/15/15 split reused by both phases |
| `data/processed/baseline_*.npy` | Phase 1 | pretrained ResNet50 embeddings/ids/categories per split |
| `data/processed/finetuned_*.npy` | Phase 2 | fine-tuned ResNet50 embeddings/ids/categories per split |
| `results/metrics/metrics_baseline_exact_knn.csv` | Phase 1 | Precision/Recall/mAP/Category Consistency, pretrained + exact |
| `results/metrics/metrics_finetuned_exact_knn.csv` | Phase 2 | same metrics, fine-tuned + exact (compare directly to the row above) |
| `results/metrics/phase2_image_only_vs_fused.csv` | Phase 2 | image-only vs. fused, same protocol |
| `results/metrics/exact_vs_{backend}_finetuned.csv` | Phase 2 | exact k-NN vs. FAISS/ANNOY — latency, Precision@K, Top-K overlap |
| `results/logs/latency_log.csv` | Phase 1 | index build time, mean/median/p95 query latency |
| `results/logs/phase2_training_log.csv` | Phase 2 | per-epoch triplet loss + epoch time |
| `results/checkpoints/resnet50_triplet_finetuned.pt` | Phase 2 | fine-tuned model weights |
| `results/figures/*.png` | both | category distribution, Precision/Recall curves, Category Consistency bars, exact-vs-ANN comparison |
| `results/qualitative_examples/*.png` | Phase 1 | query image + Top-5 retrieved products, for failure analysis |

## Reproducibility notes

- All randomness (split, triplet sampling, TF-IDF/SVD, PyTorch) is seeded
  via `config.SEED` (default 42, set in `configs/phase1_config.yaml`).
- The train/val/test split is created **once** by Phase 1 and reused
  unchanged by Phase 2 — `scripts/train_triplet.py` and
  `notebooks/phase2_improvements.ipynb` both load the existing
  `data/splits/*.csv` rather than re-splitting.
- Every comparison (baseline vs. fine-tuned, image-only vs. fused, exact vs.
  approximate) evaluates the *same* test queries under the *same* Relevance
  Protocol A (`articleType` match) — see `src/evaluation.py`.
- Fusion weights (`fusion_image_weight` / `fusion_text_weight` in
  `configs/phase2_config.yaml`) and any ANN parameters should be tuned by
  you on the **validation** split only, never on test.
- To re-run an experiment from scratch, delete the relevant files under
  `data/processed/`, `data/splits/`, and `results/`, then re-run the
  scripts/notebooks in order — nothing is cached silently.

## Known scope limits (documented, not hidden)

- **RQ3 (image+text fusion) is skipped for datasets with no title/description
  field** (`dataset_has_title_column: false`), such as the Clothing dataset
  (full, high resolution). Fusing the category label itself as "text" would
  leak the evaluation target into the fused embedding, making any resulting
  category-consistency improvement meaningless rather than a genuine fusion
  result. `scripts/evaluate_phase2.py` detects this via
  `config.HAS_TITLE_COLUMN` and saves only image-only (fine-tuned) metrics
  in that case — RQ3 is reported as out of scope for that dataset, not
  answered with a misleading number.
- Triplet mining in `src/finetune.py` uses random anchor/positive/negative
  sampling by default. `src/triplet_sampling.py` and
  `configs/phase2_config.yaml` mention `"semi_hard"` mining as an option,
  but batched semi-hard mining is **not** wired into the default training
  loop yet — see `src/finetune.select_semi_hard_negative()` for a documented
  starting point if you want to extend it.
- Only the last `unfreeze_last_n_blocks` ResNet50 block(s) are fine-tuned by
  default (`layer4`), to keep training time and overfitting risk manageable
  on Colab Free with a small-to-medium dataset. Adjust in
  `configs/phase2_config.yaml` if you have more compute.
- `IndexFlatIP` (FAISS) is itself an *exact* inner-product search — it's
  included as a FAISS-backed reference point distinct from scikit-learn's
  brute-force k-NN. For a genuinely *approximate* FAISS comparison, switch
  `faiss_index_type` to `"IndexIVFFlat"` in `configs/phase2_config.yaml`.
