# P2IF: Place-to-Interaction Flow Representations for Urban Places

## 1. Overview

To address limitations of prior place representations, this study proposes **P2IF**, a Transformer-based framework for urban place representation. P2IF models interactions between incoming populations and local place attributes through a **cross-attention** mechanism, in which the demographic composition of origin places serves as the **query** and multiple local feature embeddings serve as **keys** and **values**. **Mobility flows** are incorporated as **attention bias** terms to emphasize salient inter-place interactions. For robust and transferable representations, we use a **self-supervised** objective combining **dual-view augmentation** with **neighborhood-constrained contrastive learning**.

This README follows a journal-style data and code sharing layout: **end-to-end data workflow**, **step-by-step reproduction** for reported tables, figures, and metrics (Sections 4–5), and a companion [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md) with item-by-item commands. Reviewers should **not edit Python source files**; use **CLI arguments** and JSON configs only. Paths below are **relative to the repository root** unless stated otherwise.

---

## 2. Repository structure

```
.
├── config/                 # Training configs (JSON, all relative paths inside)
├── data/                   # Shared tabular / graph inputs (see Section 4)
│   └── weekday/            # Core graph + features for self-supervised training
├── dataload/
│   └── urban_dataset.py    # Builds DGL spatial + OD graphs and node features
├── model/
│   ├── layers/
│   │   ├── od_cross_attention.py   # OD cross-attention + flow bias
│   │   ├── spatial_sage.py
│   │   └── projection_head.py
│   └── nets/               # P2IF (UrbanModelAug), baselines, ablations
├── task/                   # Downstream evaluation, clustering, k selection, AIC
├── train_urban_unsup.py    # Main self-supervised training + embedding export
├── README.md
└── REPRODUCIBILITY_GUIDE.md  # Item-by-item reproduction for Tables 2–3, Figures 2–6, Sections 4–5
```

**Implementation mapping (high level)**

| Component (paper) | Code location |
|-------------------|---------------|
| Cross-attention + flow bias | `model/layers/od_cross_attention.py` (`ODCrossAttention`) |
| Dual-view augmentation + contrastive loss | `model/nets/model_aug.py` (`UrbanModelAug`) and related `model_without_*.py` ablations |
| Graph encoder / fusion | `model/nets/model_aug.py`, `spatial_sage.py` |
| Training loop, checkpoints, `best_embeddings.npz` | `train_urban_unsup.py` |
| Downstream regression / classification | `task/task_lightgbm.py`, `task/predict_landuse_mlp.py` |
| In-sample AIC from downstream predictions | `task/aic_from_predictions_csv.py` |
| Clustering & embedding visualization | `task/cluster_embeddings.py`, `task/find_optimal_k.py` |

---

## 3. Environment setup

### 3.1 Python environment

We recommend Python **3.10+** and a clean virtual environment.

```bash
cd /path/to/P2IF   # replace with your clone path; below we assume repo root as CWD
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3.2 Dependencies (core packages)

Install packages used across training and tasks (adjust versions to match your CUDA build for PyTorch):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # or CPU wheel
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html   # pick DGL wheel matching your torch/CUDA
pip install numpy pandas scikit-learn matplotlib scipy tqdm lightgbm
pip install umap-learn   # optional; for UMAP plots in cluster_embeddings.py
```

Install pinned Python dependencies (see comments inside the file for **torch / DGL** and CUDA wheels):

```bash
pip install -r requirements.txt
```

### 3.3 Off-the-shelf software and screenshots

If any analysis step uses **GUI software** (e.g. **QGIS**, **ArcGIS Pro**, or **Excel** for inspecting raw government tables before export to CSV), authors should:

1. Place step-by-step **screenshots** under `docs/screenshots/` (create the folder if missing).
2. Name files by step, e.g. `docs/screenshots/qgis_01_import_layer.png`, `docs/screenshots/qgis_02_join_by_grid_id.png`.
3. In the manuscript’s data availability appendix, **list every non-default parameter** (CRS, join keys, field names, export format).

**This repository’s released training CSVs** under `data/` are already grid-aligned; reproducing **model code and metrics** does not require GIS if you rely solely on these files.

---

## 4. Data workflow (original inputs → model-ready files)

All steps below use **relative paths** from the repo root.

### 4.1 What is included in the shared package

| Relative path | Role |
|---------------|------|
| `data/weekday/Spatial_Adjacency/adjacency_ids.txt` | Node IDs (one per line); defines node order |
| `data/weekday/Spatial_Adjacency/adjacency_matrix.csv` | Spatial adjacency weights |
| `data/weekday/Spatial_Adjacency/hops_matrix.csv` | Optional hop distances (e.g. Region2Vec-style losses) |
| `data/weekday/OD_Aajacency/Flow_matrix_weekday.csv` | OD / flow matrix for functional edges |
| `data/weekday/features.csv` | Per-node multimodal features (POI / resident / visitor / optional street columns per config) |
| `data/house_price_aligned_grid.csv` | Downstream: house price targets aligned to grid |
| `data/vitality_weekday_aggregated.csv` | Downstream: vitality (e.g. footfall) |
| `data/landuse_aligned_grid.csv` | Downstream: land-use labels |

### 4.2 End-to-end logic (for transparency)

1. **Raw sources** (governmental surveys, mobile OD tables, POI databases, etc.) are cleaned and aggregated to the study grid.
2. **Spatial graph**: kNN or contiguity edges → `adjacency_matrix.csv` + consistent `adjacency_ids.txt`.
3. **OD graph**: flows between grid cells → `Flow_matrix_weekday.csv` (and optional edge list if you switch config to use it).
4. **Node features**: encoded POI / demographic / visitor tensors → `features.csv` (first column = same IDs as `adjacency_ids.txt`).
5. **Downstream labels**: joined by grid ID → files under `data/*.csv`.

`dataload/urban_dataset.py` loads these files, standardizes features, and builds **DGL** graphs. No further manual preprocessing is required for replication **once the above files are present as released**.

---

## 5. Self-supervised training (produce embeddings)

### 5.1 Command

From the repository root:

```bash
python train_urban_unsup.py --config config/model_aug.json --gpu 0
```

- `--config`: JSON with **relative** `data.root`, `feature_file`, `spatial_adj`, `spatial_ids`, `od_matrix`, and `model` / `optim` / `train` blocks.
- `--gpu`: GPU index (ignored on CPU-only machines if you patch device selection locally; default code uses CUDA when available).

**Main P2IF-style model** with dual-view augmentation and contrastive terms: `UrbanModelAug` — select configs that **do not** set `model_type` to a baseline variant (see `config/model_aug.json` and comments in `train_urban_unsup.py`).

### 5.2 Outputs (per run)

Training creates a **timestamped** directory:

- `checkpoints/train_<YYYYMMDD_HHMMSS>/config.json` — frozen copy of the training config  
- `checkpoints/train_<YYYYMMDD_HHMMSS>/urban_model_best.pt` — best weights  
- `checkpoints/train_<YYYYMMDD_HHMMSS>/best_embeddings.npz` — node embeddings (`node_ids`, `h_spatial`, `h_od`, `z`, … depending on model)  
- `checkpoints/train_<YYYYMMDD_HHMMSS>/loss_curve.csv` and `loss_curve.png` — **Figure 4** (training loss curve)

Set a shell variable for later steps (no source-code edits):

```bash
export RUN_DIR=checkpoints/train_20260101_120000   # replace with the directory printed at end of training
```

---

## 6. Reproducibility index: tables, figures, and metrics

This section maps each **paper artifact** to scripts and output files. Replace `RUN_DIR` with your actual `checkpoints/train_<YYYYMMDD_HHMMSS>/` folder from Section 5.2 (the timestamp is printed when training finishes).

**Full step-by-step instructions** (data sources, exact commands, expected outputs, seed notes, and GIS steps) are in [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md).

| Paper artifact | What is reproduced | Command / script | Expected output |
|----------------|-------------------|------------------|-----------------|
| **Table 2:** comparative results across downstream tasks | Test-set RMSE / MAE / R² (regression) and Accuracy / F1 (classification) for P2IF and baseline representations | Train each model (Section 6.1), then run `task/task_lightgbm.py` for all three tasks per checkpoint (see guide § Table 2) | `results/<checkpoint>/house_price_prediction/metrics.csv`, `results/<checkpoint>/vitality_prediction/metrics.csv`, `results/<checkpoint>/landuse_lgb/metrics.csv` |
| **Table 3:** ablation results | Same downstream metrics for P2IF ablations (w/o CL, w/o interaction, w/o vis, w/o res) | Train ablation configs (Section 6.1), then run the same downstream commands as Table 2 (see guide § Table 3) | Same `metrics.csv` paths under each ablation checkpoint |
| **AIC** (model-comparison metric) | In-sample AIC on train predictions | `task/aic_from_predictions_csv.py` on `train_predictions.csv` from each downstream run (see Section 6.2) | AIC printed to stdout |
| **Figure 2:** residential profile statistics | Summary statistics of resident-profile features (`res_0`–`res_27`) | Inspect `data/weekday/features.csv` (see guide § Figure 2) | Descriptive stats / histograms derived from released CSV |
| **Figure 3:** downstream task data visualization | Distributions of house price, vitality, and land-use labels | Inspect `data/house_price_aligned_grid.csv`, `data/vitality_weekday_aggregated.csv`, `data/landuse_aligned_grid.csv` (see guide § Figure 3) | Label distributions / summary plots from released CSVs |
| **Figure 4:** training loss curve | Self-supervised training loss vs. epoch | Produced automatically by Section 5 training command | `${RUN_DIR}/loss_curve.png`, `${RUN_DIR}/loss_curve.csv` |
| **Figure 5:** spatial clustering maps | Choropleth of cluster assignments on the study grid | `task/cluster_embeddings.py` → join `cluster_assignments.csv` to grid shapefile in QGIS (see guide § Figure 5) | `${RUN_DIR}/cluster_results/cluster_assignments.csv` + GIS map export |
| **Figure 6:** t-SNE visualization | 2-D t-SNE of fused embeddings colored by cluster | `task/cluster_embeddings.py --viz_methods tsne` (see guide § Figure 6) | `${RUN_DIR}/cluster_results/clusters_tsne.png` |

**Sections 4–5 metrics (self-supervised training and downstream evaluation)**

- **Self-supervised training:** loss components logged in `${RUN_DIR}/loss_curve.csv` and printed each epoch.  
- **Downstream:** `task/task_lightgbm.py` prints **RMSE, MAE, R²** (regression) or **accuracy, F1** (classification) to stdout and saves **`metrics.csv`**, **`train_predictions.csv`**, **`test_predictions.csv`**, and **`lightgbm_model.txt`** under `results/<checkpoint_name>/<task>/`.

### 6.1 Baselines and ablations

Train with JSON configs under `config/`. **Do not edit** Python source; only change `--config`:

| Config file | Model |
|-------------|-------|
| `config/model_aug.json` | P2IF (full model) |
| `config/model_Region2Vec.json` | Region2Vec baseline |
| `config/model_GAT.json` | GAT baseline |
| `config/model_HREP.json` | HREP baseline |
| `config/model_ReMVC.json` | ReMVC baseline |
| `config/model_without_cl.json` | Ablation: w/o contrastive learning |
| `config/model_without_interaction.json` | Ablation: w/o cross-modal interaction |
| `config/model_without_vis.json` | Ablation: w/o visitor profile |
| `config/model_without_res.json` | Ablation: w/o resident subspace query |

```bash
python train_urban_unsup.py --config config/model_aug.json --gpu 0
python train_urban_unsup.py --config config/model_Region2Vec.json --gpu 0
# ... repeat for each row above
```

Each run creates its own `RUN_DIR`; point downstream commands to the matching `best_embeddings.npz`. See [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md) for the full Table 2 / Table 3 command sequences.

### 6.2 Model comparison via AIC (Akaike Information Criterion)

Reviewers may ask for **AIC** values to compare downstream predictors fitted on different place representations (P2IF vs. baselines). This repository provides a standalone script that computes **Gaussian-regression AIC** from saved prediction CSVs produced by `task/task_lightgbm.py`.

**Formula (same as used in our analysis pipeline):**

\[
\mathrm{AIC} = n \ln\!\left(\frac{\mathrm{RSS}}{n}\right) + 2k
\]

where \(n\) is the sample size, \(\mathrm{RSS} = \sum_i (y_i - \hat{y}_i)^2\) is the residual sum of squares, and \(k\) is the effective parameter count used in the penalty term.

**Important conventions**

| Item | Requirement |
|------|-------------|
| Input CSV | Must contain **`y_true`** and **`y_pred`** columns (defaults; override with `--y-true-col` / `--y-pred-col` if needed). |
| Which predictions | Use **`train_predictions.csv`** — predictions on the **training split used to fit the downstream model** (in-sample AIC). Do **not** substitute `test_predictions.csv` unless you deliberately report a different analysis and interpret it accordingly. |
| Which tasks | AIC applies to all `task_lightgbm.py` downstream runs: regression (**house price**, **vitality**) and classification (**land use**). |
| Where files live | After a downstream run, outputs are under `results/<checkpoint_name>/<task>/`, e.g. `results/train_20260101_120000/vitality_prediction/train_predictions.csv` (regression) or `results/train_20260101_120000/landuse_lgb/train_predictions.csv` (classification), plus `lightgbm_model.txt`. |

**Choosing \(k\)**

- **Linear models:** set \(k\) to the number of estimated coefficients **including the intercept** (e.g. embedding dimension after PCA plus one). Pass it explicitly with `--k`.
- **LightGBM (default downstream head):** there is no single textbook parameter count. This script supports:
  - **`--k <int>`** — you declare \(k\) in the manuscript (recommended when comparing methods under a fixed reporting rule); or
  - **`--k-from-lightgbm-trees <model.txt>`** — count `Tree=` blocks in the exported LightGBM text model as a **heuristic** complexity proxy. This is **not** a strict parameter count; interpret cross-method comparisons (e.g. LightGBM vs. linear regression) with care.

**Workflow: compare AIC across representation models**

1. Train each representation (Section 5 / Section 6.1) and note its `RUN_DIR`.
2. For each representation, run the **same downstream task** (same `--task`, splits, and LightGBM settings). Use these commands (set `RUN_DIR` first):

   **Vitality regression:**

   ```bash
   python task/task_lightgbm.py \
     --task vitality \
     --embeddings ${RUN_DIR}/best_embeddings.npz \
     --vitality_data data/vitality_weekday_aggregated.csv \
     --embedding_key h \
     --concat_keys h_spatial h_od
   ```

   **Land use classification:**

   ```bash
   python task/task_lightgbm.py \
     --task landuse \
     --embeddings ${RUN_DIR}/best_embeddings.npz \
     --landuse_data data/landuse_aligned_grid.csv \
     --embedding_key h \
     --concat_keys h_spatial h_od
   ```

   Repeat with `${RUN_DIR}` pointing to P2IF and each baseline checkpoint. Each run writes `train_predictions.csv` and `lightgbm_model.txt` under `results/<checkpoint_name>/vitality_prediction/`, `house_price_prediction/`, or `landuse_lgb/` depending on `--task`.

3. Compute AIC per run with `task/aic_from_predictions_csv.py`:

   **LightGBM — tree-count heuristic for \(k\)** (vitality example; for land use, swap paths to `landuse_lgb/`):

   ```bash
   python task/aic_from_predictions_csv.py \
     --csv results/train_20260101_120000/vitality_prediction/train_predictions.csv \
     --k-from-lightgbm-trees results/train_20260101_120000/vitality_prediction/lightgbm_model.txt \
     --label "P2IF + LightGBM"
   ```

   **Land use classification** (same script; classification CSVs also provide numeric `y_true` / `y_pred`):

   ```bash
   python task/aic_from_predictions_csv.py \
     --csv results/train_20260101_120000/landuse_lgb/train_predictions.csv \
     --k-from-lightgbm-trees results/train_20260101_120000/landuse_lgb/lightgbm_model.txt \
     --label "P2IF + LightGBM (land use)"
   ```

   **User-specified \(k\) (e.g. linear baseline or a fixed reporting rule):**

   ```bash
   python task/aic_from_predictions_csv.py \
     --csv results/train_20260101_120000/vitality_prediction/train_predictions.csv \
     --k 129 \
     --label "P2IF + linear (128 PCA dims + intercept)"
   ```

4. Compare the printed **AIC** values across `--label` tags. Lower AIC indicates a better trade-off between fit and complexity **under the stated \(k\) definition**.

The script prints \(n\), RSS, \(\widehat{\sigma^2} = \mathrm{RSS}/n\), the \(k\) source, and the final AIC. When using `--k-from-lightgbm-trees`, a short caution is also written to stderr about heuristic \(k\).

---

## 7. Random seeds and hardware

- **Seed:** set in JSON (`"seed": 42` in `config/model_aug.json`) and applied in `train_urban_unsup.py`.  
- **Hardware:** GPU recommended for training; downstream LightGBM / sklearn steps are CPU-friendly. Slight numerical differences across hardware are expected; fixed seeds minimize variance.

---

## 8. Ethics and data access

- Use shared data **only** for research reproducibility.  
- If any raw layer is under license, redistribute **derived CSVs** only (as done here) and cite the provider.

---

## 9. Citation

```bibtex
@article{p2if2026,
  title   = {P2IF: Transformer-based Urban Place Representation with Flow-biased Cross-Attention},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

*(Complete the BibTeX after publication.)*

---


