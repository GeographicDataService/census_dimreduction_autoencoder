<div align="center">

# AE_CDR — Autoencoder Census Dimensionality Reduction

Reproduction repository for the paper
**"Capturing Non-linear Neighbourhood Structure: An Autoencoder Approach to Census-based Dimensionality Reduction for Residential Differentiation"**

</div>




## Abstract

Dimensionality reduction is fundamental to applied spatial data analysis, condensing high-dimensional indicators into parsimonious representations for mapping and modelling. Principal Component Analysis (PCA) remains a dominant approach, yet its linearity assumptions constrain its capacity to capture the complex, non-linear dependencies characteristic of spatial socio-economic data. We evaluate the effectiveness of autoencoders (AEs) as a flexible, parametric alternative for constructing composite measures of neighbourhood structure. We demonstrate the approach using small-area data from the 2021 Census of England and Wales, evaluating how effectively AEs summarise spatial variability and recover composite dimensions of residential differentiation compared with PCA.

## Project structure

```
├── README.md
├── pyproject.toml / uv.lock / requirements.txt   # environment specification
├── data/                     # input data (NOT in git — see Data sources)
│   ├── census_data/          #   raw census tables
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
    ├── tables/               
    ├── clustering_results/  
    └── plots/                
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

The autoencoder implementation lives in the `torchgeodemo` package. A copy of the package source is in this repository at [`torchgeodemo/`](torchgeodemo/) — **this is the version (0.0.4) used to produce the results in the paper**, and it is what `uv sync` / `requirements.txt` install. 

The actively maintained release of this package is **deepgeodemo**: [https://deepgeodemo.readthedocs.io/en/latest/](https://deepgeodemo.readthedocs.io/en/latest/).
## Data sources

`data/` is not tracked in git. The required input files and their sources are:

### `data/census_data/`
| File | Source |
|---|---|
| `eng_raw_csvs/ts0XX.csv` (52 bulk tables: ts001–ts075, OA level, England & Wales) | [Nomis Census 2021 bulk downloads](https://www.nomisweb.co.uk/sources/census_2021_bulk) |

### `data/geofiles/`
| File | Source |
|---|---|
| `Output_Areas_(December_2021)_Boundaries_EW_BFE_(V9)_and_RUC.geojson` | ONS Open Geography Portal: ["Output Areas (December 2021) Boundaries EW BFE (V9) and Rural Urban Classification"](https://www.data.gov.uk/dataset/b60ed838-a272-4d21-9d13-f1ac5ffe9943/output-areas-december-2021-boundaries-ew-bfe-v9-and-rural-urban-classification2) (GeoJSON) |
| `Middle_layer_Super_Output_Areas_(December_2021)_Boundaries_EW_BFE_(V8)_and_RUC/` | ONS Open Geography Portal: ["Middle layer Super Output Areas (December 2021) Boundaries EW BFE (V8) and Rural Urban Classification"](https://www.data.gov.uk/dataset/7612c54f-a235-4e3b-89fc-c2085cc16b17/middle-layer-super-output-areas-december-2021-boundaries-ew-bfe-v8-and-rural-urban-classificati2) (Shapefile) |
| `Lower_layer_Super_Output_Areas_Dec_2011_Boundaries_Full_Clipped_BFC_EW_V3_2022_*.gpkg` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) |
| `Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) |
| `uk_imd2019.csv` | [Index of Multiple Deprivation 2019](https://data.geods.ac.uk/dataset/index-of-multiple-deprivation-imd) |

### `data/OAC/`
| File | Source |
|---|---|
| `OAC_assignment.csv`, `OAC_cats.csv` | [Output Area Classification 2021](https://data.geods.ac.uk/dataset/output-area-classification-2021) |


## Reproducing the paper

Run the notebooks in numeric order from the `notebooks/` directory. The stage-2 notebooks (2a–2c) train models and are GPU-intensive.

| Step | What it does |
|---|---|
| `1_prepare_data.ipynb` | Cleans and scales the raw census tables to the 408-variable model input |
| `2a_train_ae_stability.ipynb` | Trains the AE at 8 latent dimensionalities × 10 random seeds (500 epochs) |
| `2b_identity_ae_control.ipynb` | Identity-activation (linear) AE control |
| `run_lad_parallel.sh` → `run_lad_spcv.py` | 5-fold LAD-blocked spatial CV of the AE (whole Local Authority Districts held out); parallel launcher over (dim, repeat) pairs |
| `2c_lad_blocked_cv.ipynb` | Builds folds, PCA CV comparator, and aggregates the spatial-CV results |
| `make_pca_reconstructions.py` | Fits PCA once and writes the per-dimensionality baseline reconstructions used by notebooks 3, 5 and 6 |
| `3_rmse_vs_dimension.ipynb` | RMSE vs latent dimensionality, AE vs PCA |
| `4_cv_generalisation_table.ipynb` | Train vs held-out RMSE table from the LAD-blocked CV |
| `5_spatial_autocorrelation.ipynb` | Global Moran's I of reconstruction errors (queen contiguity) |
| `6_error_by_geography.ipynb` | Error by IMD/density decile and OAC group; exports per-OA/MSOA error data for mapping |
| `7a_geodemographic_clustering.ipynb` | k-means (k=8, 10,000 inits) on the 100-d AE latent space and PCA comparator |
| `7b_clustering_quality_maps.ipynb` | Cluster quality metrics and regional maps |
| `7c_clustering_pub_plots.ipynb` | Publication cluster maps and OAC cross-tabulation |
| `7d_clustering_decoding.ipynb` | Decodes cluster centroids back to census variables |


### Hardware

Models were trained on a pair of NVIDIA A6000 GPUs (each individual model trains in under 10 minutes with a <2 MB footprint). Approximately 130 GB of disk is needed for a full end-to-end reproduction. 
