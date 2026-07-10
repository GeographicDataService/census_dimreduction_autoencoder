"""Generate the PCA baseline reconstructions consumed by notebooks 3, 4 and 5.

Fits PCA once on the cleaned/scaled census data, then for each bottleneck size
writes the reconstruction from the first n components to
../AE_outputs/engcensus_all/PCA/{n}_components.csv (~1.5 GB each).

Run from the notebooks/ directory:  python make_pca_reconstructions.py
"""

import os

import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

bottleneck_sizes = [2, 4, 8, 16, 32, 64, 100, 128]

DATA_PATH = "../data/census_data/engcensus_cleaned_scaled.parquet"
PCA_SAVE_DIR = "../AE_outputs/engcensus_all/PCA/"

os.makedirs(PCA_SAVE_DIR, exist_ok=True)

df_scaled = pd.read_parquet(DATA_PATH)
df_scaled.set_index(df_scaled.columns[0], inplace=True)

n_features = df_scaled.shape[1]
pca = PCA(n_components=n_features - 1)  # Fit PCA once with the maximum number of components
transformed = pca.fit_transform(df_scaled)

for n in tqdm(bottleneck_sizes, desc="Processing PCA"):
    save_path = f"{PCA_SAVE_DIR}{n}_components.csv"
    # Project onto the first n components and invert back to the original
    # feature space (equivalent to PCA inverse transform with n components).
    reconstructed_pca = (transformed[:, :n] @ pca.components_[:n]) + pca.mean_
    pd.DataFrame(reconstructed_pca, index=df_scaled.index, columns=df_scaled.columns).to_csv(save_path)
