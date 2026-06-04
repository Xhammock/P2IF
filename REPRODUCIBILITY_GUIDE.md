# Reproducibility Guide

This document provides **item-by-item** instructions for reproducing **Tables 2–3**, **Figures 2–6**, and **Sections 4–5 metrics** reported in the manuscript. Reviewers should **not edit Python source files**; use CLI arguments and JSON configs only.

All paths are **relative to the repository root**. Replace `RUN_DIR` with the timestamped folder printed at the end of training, e.g. `checkpoints/train_20260101_120000`.

---

## Prerequisites

1. Complete [README.md § 3](README.md#3-environment-setup) (Python environment and dependencies).
2. Confirm released data files under `data/` are present (see [README.md § 4](README.md#4-data-workflow-original-inputs--model-ready-files)).
3. Train P2IF (or load a checkpoint) and set:

   ```bash
   export RUN_DIR=checkpoints/train_YYYYMMDD_HHMMSS
   ```

---

## Sections 4–5: self-supervised training metrics

| Field | Detail |
|-------|--------|
| **1. Data** | `data/weekday/` graph and feature files (see README § 4.1) |
| **2. Script** | `train_urban_unsup.py` |
| **3. Command** | `python train_urban_unsup.py --config config/model_aug.json --gpu 0` |
| **4. Expected output files** | `${RUN_DIR}/config.json`, `${RUN_DIR}/urban_model_best.pt`, `${RUN_DIR}/best_embeddings.npz`, `${RUN_DIR}/loss_curve.csv`, `${RUN_DIR}/loss_curve.png` |
| **5. Expected metric** | Total training loss per epoch (console log + `loss_curve.csv` columns); best epoch and best loss printed at end of training |
| **6. Random seed** | `"seed": 42` in `config/model_aug.json`. Fixed seed controls data augmentation and weight init; minor GPU/CPU numerical drift may still occur. |
| **7. GIS required?** | No |

---

## Table 2: comparative results across downstream tasks

Compare P2IF against baseline representations on **house price**, **vitality**, and **land use** downstream tasks.

### Step A — Train each representation

| Model | Config | Command |
|-------|--------|---------|
| P2IF | `config/model_aug.json` | `python train_urban_unsup.py --config config/model_aug.json --gpu 0` |
| Region2Vec | `config/model_Region2Vec.json` | `python train_urban_unsup.py --config config/model_Region2Vec.json --gpu 0` |
| GAT | `config/model_GAT.json` | `python train_urban_unsup.py --config config/model_GAT.json --gpu 0` |
| HREP | `config/model_HREP.json` | `python train_urban_unsup.py --config config/model_HREP.json --gpu 0` |
| ReMVC | `config/model_ReMVC.json` | `python train_urban_unsup.py --config config/model_ReMVC.json --gpu 0` |

Note the printed `RUN_DIR` for each run.

### Step B — Run downstream LightGBM for each checkpoint

Set `RUN_DIR` to the checkpoint folder, then run all three commands:

**House price (regression)**

```bash
python task/task_lightgbm.py \
  --task house_price \
  --embeddings ${RUN_DIR}/best_embeddings.npz \
  --price_data data/house_price_aligned_grid.csv \
  --embedding_key h \
  --concat_keys h_spatial h_od
```

**Vitality (regression)**

```bash
python task/task_lightgbm.py \
  --task vitality \
  --embeddings ${RUN_DIR}/best_embeddings.npz \
  --vitality_data data/vitality_weekday_aggregated.csv \
  --embedding_key h \
  --concat_keys h_spatial h_od
```

**Land use (classification)**

```bash
python task/task_lightgbm.py \
  --task landuse \
  --embeddings ${RUN_DIR}/best_embeddings.npz \
  --landuse_data data/landuse_aligned_grid.csv \
  --embedding_key h \
  --concat_keys h_spatial h_od
```

| Field | Detail |
|-------|--------|
| **1. Data** | `${RUN_DIR}/best_embeddings.npz` + `data/house_price_aligned_grid.csv`, `data/vitality_weekday_aggregated.csv`, `data/landuse_aligned_grid.csv` |
| **2. Script** | `task/task_lightgbm.py` |
| **3. Command** | Three commands above (repeat for each representation checkpoint) |
| **4. Expected output files** | `results/<checkpoint>/house_price_prediction/metrics.csv`, `results/<checkpoint>/vitality_prediction/metrics.csv`, `results/<checkpoint>/landuse_lgb/metrics.csv`; plus `train_predictions.csv`, `test_predictions.csv`, `lightgbm_model.txt` in each folder |
| **5. Expected metric / table** | **Table 2** — Test-set **RMSE, MAE, R²** (regression rows) and **Accuracy, F1_Macro, F1_Weighted** (classification row) from the `Test` row of each `metrics.csv` |
| **6. Random seed** | Downstream split: `random_state=42`, `test_size=0.2` (defaults in `task/task_lightgbm.py` CONFIG). Training seed: 42 in each JSON config. |
| **7. GIS required?** | No |

---

## Table 3: ablation results

Same downstream protocol as Table 2, but train ablation variants instead of baselines.

### Step A — Train ablation models

| Ablation | Config | Command |
|----------|--------|---------|
| P2IF (full) | `config/model_aug.json` | `python train_urban_unsup.py --config config/model_aug.json --gpu 0` |
| w/o contrastive learning | `config/model_without_cl.json` | `python train_urban_unsup.py --config config/model_without_cl.json --gpu 0` |
| w/o cross-modal interaction | `config/model_without_interaction.json` | `python train_urban_unsup.py --config config/model_without_interaction.json --gpu 0` |
| w/o visitor profile | `config/model_without_vis.json` | `python train_urban_unsup.py --config config/model_without_vis.json --gpu 0` |
| w/o resident subspace query | `config/model_without_res.json` | `python train_urban_unsup.py --config config/model_without_res.json --gpu 0` |

### Step B — Downstream evaluation

Run the same three `task/task_lightgbm.py` commands as Table 2 Step B for each ablation checkpoint.

| Field | Detail |
|-------|--------|
| **1. Data** | Same as Table 2 |
| **2. Script** | `train_urban_unsup.py` + `task/task_lightgbm.py` |
| **3. Command** | Ablation training commands above + downstream commands from Table 2 Step B |
| **4. Expected output files** | Same `metrics.csv` paths as Table 2, one set per ablation checkpoint |
| **5. Expected metric / table** | **Table 3** — Test-set metrics from each ablation's `metrics.csv` files |
| **6. Random seed** | Same as Table 2 |
| **7. GIS required?** | No |

---

## AIC (in-sample model comparison)

See [README.md § 6.2](README.md#62-model-comparison-via-aic-akaike-information-criterion) for formula and \(k\) conventions.

| Field | Detail |
|-------|--------|
| **1. Data** | `train_predictions.csv` and `lightgbm_model.txt` from each Table 2 / Table 3 downstream run |
| **2. Script** | `task/aic_from_predictions_csv.py` |
| **3. Command** | `python task/aic_from_predictions_csv.py --csv results/<checkpoint>/vitality_prediction/train_predictions.csv --k-from-lightgbm-trees results/<checkpoint>/vitality_prediction/lightgbm_model.txt --label "P2IF"` (swap paths for house price / land use) |
| **4. Expected output files** | AIC value printed to stdout (no file written) |
| **5. Expected metric** | In-sample AIC per representation and downstream task |
| **6. Random seed** | Deterministic given the same `train_predictions.csv` input |
| **7. GIS required?** | No |

---

## Figure 2: residential profile statistics

| Field | Detail |
|-------|--------|
| **1. Data** | `data/weekday/features.csv` — columns `res_0` through `res_27` (28-dimensional resident profile per grid cell); node IDs in column `GWBH` |
| **2. Script** | None (descriptive statistics computed directly from the released CSV) |
| **3. Command** | Verify summary statistics: `python -c "import pandas as pd; df=pd.read_csv('data/weekday/features.csv'); print(df.filter(regex=r'^res_').describe().to_string())"` |
| **4. Expected output files** | Console summary table; manuscript figure exported from the same CSV columns |
| **5. Expected figure** | **Figure 2** — distribution / summary statistics of resident-profile features across grid cells |
| **6. Random seed** | Not applicable (fixed derived data) |
| **7. GIS required?** | No |

---

## Figure 3: downstream task data visualization

| Field | Detail |
|-------|--------|
| **1. Data** | `data/house_price_aligned_grid.csv` (`GWBH`, `avgprice`), `data/vitality_weekday_aggregated.csv` (`GWBH`, `total_people`), `data/landuse_aligned_grid.csv` (`GWBH`, `landuse`) |
| **2. Script** | None (label distributions computed from released downstream CSVs) |
| **3. Command** | Verify label summaries: `python -c "import pandas as pd; hp=pd.read_csv('data/house_price_aligned_grid.csv'); vt=pd.read_csv('data/vitality_weekday_aggregated.csv'); lu=pd.read_csv('data/landuse_aligned_grid.csv'); print('House price\\n', hp['avgprice'].describe()); print('\\nVitality\\n', vt['total_people'].describe()); print('\\nLand use counts\\n', lu['landuse'].value_counts().sort_index())"` |
| **4. Expected output files** | Console summary; manuscript figure exported from the same CSVs |
| **5. Expected figure** | **Figure 3** — spatial or distributional visualization of downstream task labels |
| **6. Random seed** | Not applicable (fixed derived data) |
| **7. GIS required?** | Optional — if the manuscript map uses a grid shapefile, join on `GWBH` in QGIS (see Figure 5 GIS workflow) |

---

## Figure 4: training loss curve

| Field | Detail |
|-------|--------|
| **1. Data** | Training graph and features under `data/weekday/` |
| **2. Script** | `train_urban_unsup.py` (loss curve saved automatically) |
| **3. Command** | `python train_urban_unsup.py --config config/model_aug.json --gpu 0` |
| **4. Expected output files** | `${RUN_DIR}/loss_curve.png`, `${RUN_DIR}/loss_curve.csv` |
| **5. Expected figure** | **Figure 4** — training loss vs. epoch |
| **6. Random seed** | `"seed": 42` in config; curve may show minor run-to-run noise on different hardware |
| **7. GIS required?** | No |

To replot from CSV:

```bash
python -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('${RUN_DIR}/loss_curve.csv')
plt.figure(figsize=(6,4)); plt.plot(df['epoch'], df['loss']); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout(); plt.savefig('${RUN_DIR}/loss_curve_replot.png', dpi=200); print('saved ${RUN_DIR}/loss_curve_replot.png')
"
```

---

## Figure 5: spatial clustering maps

| Field | Detail |
|-------|--------|
| **1. Data** | `${RUN_DIR}/best_embeddings.npz`; study-grid shapefile (external, join key `GWBH`) |
| **2. Script** | `task/cluster_embeddings.py` (cluster assignments) + QGIS (choropleth map) |
| **3. Command** | `python task/cluster_embeddings.py --embeddings ${RUN_DIR}/best_embeddings.npz --embedding_key z --num_clusters 3 --clustering_method hierarchical --output_dir ${RUN_DIR}/cluster_results --viz_methods pca --random_state 42` |
| **4. Expected output files** | `${RUN_DIR}/cluster_results/cluster_assignments.csv` (columns `node_id`, `cluster_id`) |
| **5. Expected figure** | **Figure 5** — spatial choropleth of cluster labels on the study grid |
| **6. Random seed** | `--random_state 42` in `cluster_embeddings.py`; hierarchical clustering is deterministic given fixed embeddings |
| **7. GIS required?** | **Yes** — QGIS (or ArcGIS Pro). Authors should place step-by-step screenshots under `docs/screenshots/` (e.g. `docs/screenshots/qgis_01_import_grid.png`, `docs/screenshots/qgis_02_join_cluster_csv.png`). **Join parameters:** left layer = study grid polygon layer; join field = `GWBH`; right table = `cluster_assignments.csv`; join field = `node_id`; symbology = categorized by `cluster_id`. |

---

## Figure 6: t-SNE visualization

| Field | Detail |
|-------|--------|
| **1. Data** | `${RUN_DIR}/best_embeddings.npz` |
| **2. Script** | `task/cluster_embeddings.py` |
| **3. Command** | `python task/cluster_embeddings.py --embeddings ${RUN_DIR}/best_embeddings.npz --embedding_key z --num_clusters 3 --clustering_method hierarchical --output_dir ${RUN_DIR}/cluster_results --viz_methods tsne --random_state 42 --tsne_perplexity 30` |
| **4. Expected output files** | `${RUN_DIR}/cluster_results/clusters_tsne.png`, `${RUN_DIR}/cluster_results/cluster_assignments.csv` |
| **5. Expected figure** | **Figure 6** — 2-D t-SNE projection of fused embeddings (`z`), points colored by cluster |
| **6. Random seed** | `--random_state 42`; t-SNE is stochastic but seed-controlled |
| **7. GIS required?** | No |

---

## Quick reference: downstream output layout

After `task/task_lightgbm.py`, outputs follow this layout:

```
results/<checkpoint_name>/
├── house_price_prediction/
│   ├── metrics.csv
│   ├── train_predictions.csv
│   ├── test_predictions.csv
│   └── lightgbm_model.txt
├── vitality_prediction/
│   └── (same files)
└── landuse_lgb/
    ├── metrics.csv
    ├── classification_report.json
    ├── confusion_matrix.csv
    ├── train_predictions.csv
    ├── test_predictions.csv
    └── lightgbm_model.txt
```

After `task/cluster_embeddings.py`:

```
${RUN_DIR}/cluster_results/
├── cluster_assignments.csv
├── clusters_pca.png      # if --viz_methods includes pca
├── clusters_tsne.png     # if --viz_methods includes tsne
└── clusters_umap.png     # if --viz_methods includes umap
```

---

## Optional: optimal cluster count (not a manuscript figure)

If needed for sensitivity analysis:

```bash
python task/find_optimal_k.py \
  --embeddings ${RUN_DIR}/best_embeddings.npz \
  --embedding_key h \
  --concat_keys h_spatial h_od \
  --k_min 2 \
  --k_max 8 \
  --output_dir ${RUN_DIR}/k_selection \
  --random_state 42 \
  --use_pca \
  --pca_components 128
```

Outputs: `${RUN_DIR}/k_selection/k_selection_analysis.png`, `k_evaluation_results.csv`, `k_recommendations.txt`.
