<div align="center">

# AE_CDR — Autoencoder Census Dimensionality Reduction

Reproduction repository for the paper
**"Capturing Non-linear Neighbourhood Structure: An Autoencoder Approach to Census-based Dimensionality Reduction for Residential Differentiation"**

**FigShare data bundle:** [https://figshare.com/s/7aeba52e2b0af082d22d](https://figshare.com/s/7aeba52e2b0af082d22d)

</div>

This repository contains the notebooks and scripts that produce every figure and table in the paper. It is a curated copy of the working research repository: exploratory notebooks and superseded experiments have been removed, and only the final pipeline is kept.

## Abstract

Dimensionality reduction is fundamental to applied spatial data analysis, condensing high-dimensional indicators into parsimonious representations for mapping and modelling. Principal Component Analysis (PCA) remains a dominant approach, yet its linearity assumptions constrain its capacity to capture the complex, non-linear dependencies characteristic of spatial socio-economic data. We evaluate the effectiveness of autoencoders (AEs) as a flexible, parametric alternative for constructing composite measures of neighbourhood structure. We demonstrate the approach using small-area data from the 2021 Census of England and Wales, evaluating how effectively AEs summarise spatial variability and recover composite dimensions of residential differentiation compared with PCA.

## Project structure

```
├── README.md
├── pyproject.toml / uv.lock / requirements.txt   # environment specification
├── data/                     # input data (NOT in git — see Data sources)
│   ├── census_data/          #   raw census tables + cleaned parquet from notebook 1
│   ├── geofiles/             #   boundaries, lookups, IMD
│   ├── OAC/                  #   2021 Output Area Classification
│   └── cache/                #   auto-regenerated caches (spatial weights etc.)
├── AE_outputs/               # model outputs (NOT in git — created by notebooks 2*/scripts)
├── torchgeodemo/             # vendored autoencoder package (exact version used for the paper)
└── notebooks/
    ├── 1_prepare_data.ipynb
    ├── 2a_train_ae_stability.ipynb
    ├── 2b_identity_ae_control.ipynb
    ├── 2c_lad_blocked_cv.ipynb
    ├── run_lad_spcv.py / run_lad_parallel.sh
    ├── make_pca_reconstructions.py
    ├── 3_rmse_vs_dimension.ipynb
    ├── 4_cv_generalisation_table.ipynb
    ├── 5_spatial_autocorrelation.ipynb
    ├── 6_error_by_geography.ipynb
    ├── 7a…7d clustering notebooks
    ├── tables/               # shipped reference outputs (Table 3, CV RMSE)
    ├── clustering_results/   # shipped reference outputs (Section 5)
    └── plots/                # figures written here when notebooks run (not in git)
```

## Installation

Python 3.12 with CUDA-enabled PyTorch (wheels pinned to the CUDA 12.8 index).

```bash
git clone <this repo> AE_CDR
cd AE_CDR
uv sync          # creates .venv from uv.lock (exact paper environment)
```

Without uv: create a venv and `pip install -r requirements.txt` (which installs the vendored autoencoder package from `./torchgeodemo`).

### The `torchgeodemo` autoencoder package

The autoencoder implementation lives in the `torchgeodemo` package. A copy of the package source is vendored in this repository at [`torchgeodemo/`](torchgeodemo/) — **this is the exact version (0.0.4) used to produce the results in the paper**, and it is what `uv sync` / `requirements.txt` install. It is included so the results remain reproducible independently of upstream development.

The actively maintained release of this package is **deepgeodemo**: [https://deepgeodemo.readthedocs.io/en/latest/](https://deepgeodemo.readthedocs.io/en/latest/). For new work, use deepgeodemo; for reproducing this paper, use the vendored copy.

## Data sources

`data/` is not tracked in git. The complete, ready-to-run `data/` directory is available in the [FigShare bundle](https://figshare.com/s/7aeba52e2b0af082d22d). To rebuild it from primary sources instead, the required files are:

### `data/census_data/`
| File | Source |
|---|---|
| `eng_raw_csvs/ts0XX.csv` (52 bulk tables: ts001–ts075, OA level, England & Wales) | [Nomis Census 2021 bulk downloads](https://www.nomisweb.co.uk/sources/census_2021_bulk) |
| `engcensus_cleaned_scaled.parquet` | **derived** — produced by notebook 1 |

### `data/geofiles/`
| File | Source |
|---|---|
| `Output_Areas_(December_2021)_Boundaries_EW_BFE_(V9)_and_RUC.geojson` | OA Dec 2021 boundaries (BFE V9) joined with the 2021 Rural-Urban Classification; both from the [ONS Open Geography Portal](https://geoportal.statistics.gov.uk). Pre-joined file included in the FigShare bundle |
| `Middle_layer_Super_Output_Areas_(December_2021)_Boundaries_EW_BFE_(V8)_and_RUC/` (shapefile) | as above, MSOA Dec 2021 boundaries (BFE V8) |
| `Lower_layer_Super_Output_Areas_Dec_2011_Boundaries_Full_Clipped_BFC_EW_V3_2022_*.gpkg` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) |
| `Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) |
| `lookup_oa2022_lsoa11_EW.csv` | **derived** — OA 2021 → LSOA 2011 largest-overlap lookup, produced by notebook 1 from the two boundary files above |
| `oabounds_2021approx.parquet` | simplified OA boundaries for fast plotting — included in the FigShare bundle |
| `uk_imd2019.csv` | [Index of Multiple Deprivation 2019](https://data.geods.ac.uk/dataset/index-of-multiple-deprivation-imd) |

### `data/OAC/`
| File | Source |
|---|---|
| `OAC_assignment.csv`, `OAC_cats.csv` | [Output Area Classification 2021](https://data.geods.ac.uk/dataset/output-area-classification-2021) |

`data/cache/` is created automatically (spatial weights matrix, plotting caches) the first time notebook 5 or 6 runs.

## Reproducing the paper

Run the notebooks in numeric order from the `notebooks/` directory. The stage-2 notebooks (2a–2c) train models and are GPU-intensive; everything else runs in minutes from their saved outputs.

| Step | What it does | Paper output | Runtime (approx.) |
|---|---|---|---|
| `1_prepare_data.ipynb` | Cleans and scales the raw census tables to the 408-variable model input | — | minutes |
| `2a_train_ae_stability.ipynb` | Trains the AE at 8 latent dimensionalities × 10 random seeds (500 epochs) | training-stability claims (§2.2), inputs to Fig 2 | ~8 dims × 10 runs × ~10 min on GPU (~1 day serial) |
| `2b_identity_ae_control.ipynb` | Identity-activation (linear) AE control | robustness check (not a paper figure) | as notebook 2a |
| `run_lad_parallel.sh` → `run_lad_spcv.py` | 5-fold LAD-blocked spatial CV of the AE (whole Local Authority Districts held out); parallel launcher over (dim, repeat) pairs | §3.1 | ~11 h serial; ~30–60 min parallel on a large GPU |
| `2c_lad_blocked_cv.ipynb` | Builds folds, PCA CV comparator, and aggregates the spatial-CV results | §3.1 | minutes (after the script) |
| `make_pca_reconstructions.py` | Fits PCA once and writes per-dimensionality baseline reconstructions | PCA baseline used by notebooks 3, 5, 6 | ~30 min, ~12 GB output |
| `3_rmse_vs_dimension.ipynb` | RMSE vs latent dimensionality, AE vs PCA | **Figure 2** | minutes |
| `4_cv_generalisation_table.ipynb` | Train vs held-out RMSE table from the LAD-blocked CV | **Table 3 (§3.1)** | minutes |
| `5_spatial_autocorrelation.ipynb` | Global Moran's I of reconstruction errors (queen contiguity) | **Table 3 (Moran's I)** | ~1 h first run (builds weights cache), then minutes |
| `6_error_by_geography.ipynb` | Error by IMD/density decile, by OAC group; exports per-OA/MSOA error data for mapping | **Figures 3, 4**; data behind **Figure 5** | ~30 min uncached |
| `7a_geodemographic_clustering.ipynb` | k-means (k=8, 10,000 inits) on the 100-d AE latent space and PCA comparator | clustering used in §5 | ~1 h |
| `7b_clustering_quality_maps.ipynb` | Cluster quality metrics and regional maps | §5.3 metrics | minutes |
| `7c_clustering_pub_plots.ipynb` | Publication cluster maps and OAC cross-tabulation | **Figures 6, 7** | minutes |
| `7d_clustering_decoding.ipynb` | Decodes cluster centroids back to census variables | §5.4 | minutes |

Figures are written to `notebooks/plots/` (publication versions in `plots/pub_plots/`), tables to `notebooks/tables/`, clustering outputs to `notebooks/clustering_results/`.

**Figure 5** (MSOA reconstruction-error choropleths) is not drawn by a notebook: notebook 6 exports `plots/maps/500ep_linscaling_run5_error_diff_by_MSOA_100d.parquet`, from which the published maps were styled in GIS software.

### Reference outputs

`notebooks/tables/` and `notebooks/clustering_results/` ship with the repository as computed for the paper, so reproduced numbers can be checked without re-running the full training. Re-running the notebooks overwrites them in place.

### Hardware

Models were trained on a pair of NVIDIA A6000 GPUs (each individual model trains in under 10 minutes with a <2 MB footprint); the LAD-blocked cross-validation runs were executed on an NVIDIA RTX PRO 6000. Approximately 130 GB of disk is needed for a full end-to-end reproduction (model checkpoints and PCA reconstructions dominate).
