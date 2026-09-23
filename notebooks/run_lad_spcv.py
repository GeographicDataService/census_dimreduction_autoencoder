#!/usr/bin/env python
"""
Parallel worker for the LAD-grouped spatial CV (notebook 2c, 2c_lad_blocked_cv).

Trains the autoencoder for a SINGLE (bottleneck_dim, repeat) pair and writes the
same per-(dim, repeat) checkpoint the notebook produces, so the notebook's
aggregation/plot cells can consume the results unchanged.

Why (dim, repeat) is the parallel unit: checkpoints are keyed per (dim, repeat)
and updated fold-by-fold via read-modify-write. Splitting finer (per fold) would
race on the same checkpoint file; splitting coarser wastes the idle GPU. Each
(dim, repeat) owns a distinct checkpoint + distinct model/temp files, so any set
of them can run concurrently with zero collisions.

Folds are rebuilt here from the OA->LAD lookup with the exact same seeds as the
notebook (balanced greedy bin-packing of whole LADs). The centroid step in the
notebook is visualization-only and drops 0 OAs, so script-built folds are
identical to notebook-built folds.

Usage:
    python run_lad_spcv.py --dim 128 --repeat 0
    python run_lad_spcv.py --dims 128 100 --repeats 0 1   # loops the product
Resumable: already-completed folds are skipped via the checkpoint.
"""
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch

from torchgeodemo import autoencoder_train_latent

# --- Paths (resolved from this file, so location-independent) ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data/census_data/engcensus_cleaned_scaled.parquet"
LAD_LOOKUP  = ROOT / ("data/geofiles/Output_Area_to_Lower_layer_Super_Output_Area_"
                      "to_Middle_layer_Super_Output_Area_to_Local_Authority_District_"
                      "(December_2021)_Lookup_in_England_and_Wales_v3.csv")

# --- Constants (must match notebook 2c) --------------------------------------
# LAD_SPCV_EPOCHS env var overrides epochs for smoke-testing only; leave unset
# for real runs so results match notebook 2c (500 epochs).
N_EPOCHS       = int(os.environ.get("LAD_SPCV_EPOCHS", 500))
BATCH_SIZE     = 0.01
SCALING_TYPE   = "lin"
N_FOLDS        = 5
N_REPEATS      = 10
BASE_SPLIT_SEED = 42          # LAD-shuffle seed base
BASE_SEED       = 20210321    # AE training seed base
OUTPUT_DIR = ROOT / (f"AE_outputs/retraining_stability_{N_EPOCHS}epochs_"
                     f"{SCALING_TYPE}scaling_lad_blocked")


def gen_layer_sizes(input_size, latent_size, num_layers, scaling_type="lin"):
    if scaling_type == "mul":
        scale = (latent_size / input_size) ** (1 / (num_layers - 1))
        return [int(input_size * scale ** i) for i in range(1, num_layers)]
    if scaling_type == "lin":
        step = (latent_size - input_size) / (num_layers - 1)
        return [int(input_size + step * i) for i in range(1, num_layers)]
    raise ValueError("Invalid scaling type. Use 'mul' or 'lin'.")


def build_lad_folds(oa_lad, lad_oa_counts, n_folds, seed):
    """Greedy balanced bin-packing of whole LADs into n_folds (matches notebook 2c)."""
    rng = np.random.default_rng(seed)
    lads = lad_oa_counts.index.to_numpy()
    lads = lads[rng.permutation(len(lads))]
    fold_load = np.zeros(n_folds, dtype=np.int64)
    lad_to_fold = {}
    for lad in lads:
        f = int(np.argmin(fold_load))
        lad_to_fold[lad] = f
        fold_load[f] += int(lad_oa_counts[lad])
    return np.array([lad_to_fold[l] for l in oa_lad], dtype=int)


def train_ae_via_yaml(data_df, run_name, latent_dim, working_dir, seed,
                      n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, scaling_type=SCALING_TYPE):
    """Train AE via torchgeodemo YAML config. Skips if latent CSV already exists."""
    model_nickname = f"stability_{run_name}__ae_{latent_dim}d_{run_name}_{n_epochs}ep_v1"
    latent_csv_path = f"{working_dir}/{model_nickname}__latent.csv"
    if os.path.exists(latent_csv_path):
        print(f"[{run_name}] latent CSV exists, skipping training", flush=True)
        return

    print(f"[{run_name}] training (seed={seed}, epochs={n_epochs})", flush=True)
    torch.manual_seed(seed)
    if 'OA' not in data_df.columns:
        data_df = data_df.reset_index()

    input_dim = data_df.shape[1] - 1
    encoder_sizes = gen_layer_sizes(input_dim, latent_dim, num_layers=4, scaling_type=scaling_type)

    yaml_config = {
        "data": {"source": "TEMP", "nickname": f"stability_{run_name}", "id_col": "OA"},
        "working_dir": working_dir,
        "autoencoder": {
            "nickname": f"ae_{latent_dim}d_{run_name}_{n_epochs}ep",
            "version": "1",
            "save_latent": "csv",
            "max_epochs": n_epochs,
            "batch_size": batch_size,
            "use_covariance_loss": False,
            "random_seed": seed,
            "encoder": {"sizes": encoder_sizes, "activation": "LeakyReLU"},
            "decoder": {"sizes": encoder_sizes[::-1], "activation": "LeakyReLU"},
        },
    }
    temp_data_path = f"{working_dir}/temp_data_{run_name}.parquet"
    data_df.to_parquet(temp_data_path)
    yaml_config["data"]["source"] = temp_data_path

    yaml_dir = f"{working_dir}/yamls"
    os.makedirs(yaml_dir, exist_ok=True)
    config_path = f"{yaml_dir}/config_{run_name}_{latent_dim}d_{n_epochs}ep.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    # verbose=False disables the per-step Lightning progress bar (pure overhead
    # for this tiny model) -> numerically identical, just faster.
    autoencoder_train_latent.main(config_path, create_latent=True,
                                  save_reco_error=False, verbose=False)
    print(f"[{run_name}] training complete", flush=True)


def get_checkpoint_path(dim, repeat_idx):
    return OUTPUT_DIR / "data" / f"spcv_checkpoint_{dim}d_rep{repeat_idx}.pkl"


def load_checkpoint(dim, repeat_idx):
    p = get_checkpoint_path(dim, repeat_idx)
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


def save_checkpoint(dim, repeat_idx, payload):
    p = get_checkpoint_path(dim, repeat_idx)
    with open(p, 'wb') as f:
        pickle.dump(payload, f)
    print(f"  saved checkpoint: {p.name}", flush=True)


def load_data_and_folds():
    """Rebuild df (census order) and LAD folds; write spatial_folds.pkl if absent."""
    df = pd.read_parquet(DATA_PATH)
    if 'OA' not in df.columns:
        df = df.reset_index()

    lad = pd.read_csv(LAD_LOOKUP, usecols=['OA21CD', 'LAD22CD'], low_memory=False)
    lad = lad.rename(columns={'OA21CD': 'OA', 'LAD22CD': 'LAD'})
    merged = df[['OA']].merge(lad, on='OA', how='left')
    if int(merged['LAD'].isna().sum()):
        raise ValueError("some OAs have no LAD match")
    oa_lad = merged['LAD'].values
    lad_oa_counts = pd.Series(oa_lad).value_counts()

    fold_assignments = [build_lad_folds(oa_lad, lad_oa_counts, N_FOLDS, BASE_SPLIT_SEED + r)
                        for r in range(N_REPEATS)]

    folds_pkl = OUTPUT_DIR / "data" / "spatial_folds.pkl"
    if not folds_pkl.exists():
        with open(folds_pkl, 'wb') as f:
            pickle.dump({
                'grouping': 'LAD22CD', 'method': 'greedy_balanced_binpack',
                'n_folds': N_FOLDS, 'n_repeats': N_REPEATS,
                'base_split_seed': BASE_SPLIT_SEED,
                'fold_assignments': fold_assignments,
                'oa_ids': df['OA'].values, 'oa_lad': oa_lad, 'coords': None,
            }, f)
        print(f"  wrote {folds_pkl.name}", flush=True)

    X_full = df.drop(columns=['OA']).values
    return df, X_full, fold_assignments


def run_pair(df, X_full, fold_assignments, latent_dim, repeat_idx):
    ckpt = load_checkpoint(latent_dim, repeat_idx)
    if ckpt is not None and ckpt.get('n_folds_done', 0) >= N_FOLDS:
        print(f"[skip] {latent_dim}d rep{repeat_idx} fully cached", flush=True)
        return
    repeat_folds = ckpt['folds'] if ckpt else {}
    labels = fold_assignments[repeat_idx]
    models_dir = str(OUTPUT_DIR / "models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold_idx in range(N_FOLDS):
        if fold_idx in repeat_folds:
            continue
        test_idx  = np.where(labels == fold_idx)[0]
        train_idx = np.where(labels != fold_idx)[0]
        df_train = df.iloc[train_idx].reset_index(drop=True)
        X_train  = X_full[train_idx]
        X_test   = X_full[test_idx]

        run_name = f"{latent_dim}d_r{repeat_idx}_f{fold_idx}"
        seed = BASE_SEED + repeat_idx * 10_000 + fold_idx * 1000
        print(f"\n[{run_name}] train={len(train_idx)} test={len(test_idx)} seed={seed}", flush=True)

        train_ae_via_yaml(df_train, run_name, latent_dim, models_dir, seed)

        model_nickname = f"stability_{run_name}__ae_{latent_dim}d_{run_name}_{N_EPOCHS}ep_v1"
        model_path = f"{models_dir}/{model_nickname}__model.pth"
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        with torch.no_grad():
            Xt  = torch.FloatTensor(X_test).to(device)
            Xtr = torch.FloatTensor(X_train).to(device)
            test_reco  = model.decoder(model.encode(Xt)).cpu().numpy()
            train_reco = model.decoder(model.encode(Xtr)).cpu().numpy()
            test_err  = np.mean((X_test  - test_reco)  ** 2, axis=1)
            train_err = np.mean((X_train - train_reco) ** 2, axis=1)

        payload = {
            'test_idx': test_idx, 'train_idx': train_idx,
            'test_reco_error': test_err, 'train_reco_error': train_err,
            'test_rmse':  float(np.sqrt(test_err.mean())),
            'train_rmse': float(np.sqrt(train_err.mean())),
        }
        print(f"  train RMSE={payload['train_rmse']*100:.4f}%  "
              f"test RMSE={payload['test_rmse']*100:.4f}%", flush=True)

        repeat_folds[fold_idx] = payload
        save_checkpoint(latent_dim, repeat_idx, {
            'bottleneck_size': latent_dim, 'repeat_idx': repeat_idx,
            'folds': repeat_folds, 'n_folds_done': len(repeat_folds),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, help='single bottleneck dim')
    ap.add_argument('--repeat', type=int, help='single repeat index')
    ap.add_argument('--dims', type=int, nargs='+', help='multiple dims')
    ap.add_argument('--repeats', type=int, nargs='+', help='multiple repeats')
    args = ap.parse_args()

    dims    = args.dims    if args.dims    else ([args.dim]    if args.dim    is not None else [128, 100, 64, 32, 16, 8, 4, 2])
    repeats = args.repeats if args.repeats else ([args.repeat] if args.repeat is not None else [0]) 

    for d in (OUTPUT_DIR, OUTPUT_DIR / "data", OUTPUT_DIR / "models", OUTPUT_DIR / "yamls"):
        os.makedirs(d, exist_ok=True)

    df, X_full, fold_assignments = load_data_and_folds()
    print(f"data ready: X_full={X_full.shape}  dims={dims}  repeats={repeats}", flush=True)
    for d in dims:
        for r in repeats:
            print(f"\n{'='*70}\nDIM {d}  REPEAT {r}\n{'='*70}", flush=True)
            run_pair(df, X_full, fold_assignments, d, r)


if __name__ == '__main__':
    main()
