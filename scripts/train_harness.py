"""
train_harness.py
================
Orchestrator for training all copolymer model paradigms on D_train.
Imports modular trainers from scripts/models/:
- Tabular: Random Forest & XGBoost
- Neural: Concatenation, Latent Convex, Thermodynamic Hybrid
- Sequence: BigSMILES Language Model Head
- Physics: Gordon-Taylor pair parameter fit

Runs across 5 independent seeds (42..46) and checkpoints all models to results/models/.
"""

import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

# Configure writable matplotlib cache directory
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add scripts directory to path to allow importing models
sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import train_random_forest, train_xgboost, save_tabular_model
from models.neural_models import build_neural_model, train_neural_model, save_neural_model
from models.sequence_models import build_sequence_model, train_sequence_model, save_sequence_model
from models.physics_baselines import fit_all_pairs_gordon_taylor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_harness")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    base_dir = Path("C:/Projects/copolymers")
    proc_dir = base_dir / "data/processed"
    raw_path = base_dir / "data/raw/experimental_dataset.csv"
    models_dir = base_dir / "results/models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load split manifest
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    train_idx = manifest["train_indices"]
    log.info(f"Loaded train indices: {len(train_idx)} samples in D_train.")

    # 2. Load dataset tables
    df_raw = pd.read_csv(raw_path)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    blends["pair_id"] = blends["Poly1"].astype(str) + "-" + blends["Poly2"].astype(str)
    blends["f"] = blends["comp_0"].astype(float) / 100.0
    blends["Tg_actual"] = blends["Tg(K)"].astype(float)

    # 3. Load Tabular RDKit features
    rdkit_data = np.load(proc_dir / "blend_rdkit_features.npz")
    X_rdkit_tr = rdkit_data["features"][train_idx]
    y_tr = rdkit_data["targets"][train_idx]
    log.info(f"Loaded Tabular RDKit features: {X_rdkit_tr.shape}")

    # 4. Load PolyBERT embeddings (600-d)
    pb_data = np.load(proc_dir / "monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])  # 600

    ea_tr = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends.iloc[train_idx]["SMILES_0"]]), dtype=torch.float32)
    eb_tr = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends.iloc[train_idx]["SMILES_1"]]), dtype=torch.float32)
    f_tr = torch.tensor(blends.iloc[train_idx]["f"].values, dtype=torch.float32).unsqueeze(-1)

    # 5. Load BigSMILES ChemBERTa cache
    bs_cache = np.load(proc_dir / "bigsmiles_cache.npz")
    zlm_tr = torch.tensor(bs_cache["blend_embs"][train_idx], dtype=torch.float32)

    # Standardize target strictly on train set (Zero data leakage)
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() + 1e-7)
    y_tr_norm = torch.tensor((y_tr - y_mean) / y_std, dtype=torch.float32)

    with open(models_dir / "target_normalization.json", "w") as f:
        json.dump({"y_mean": y_mean, "y_std": y_std}, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Target norm stats: mean = {y_mean:.2f} K, std = {y_std:.2f} K. Training device: {device}")

    SEEDS = [42, 43, 44, 45, 46]

    # DataLoaders for neural models
    neural_ds = TensorDataset(ea_tr, eb_tr, f_tr, y_tr_norm)
    seq_ds = TensorDataset(zlm_tr, y_tr_norm)

    for seed in SEEDS:
        log.info(f"\n=======================================================")
        log.info(f"             TRAINING SEED {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        log.info(f"=======================================================")
        set_seed(seed)

        # ── A. Tabular Models ──
        log.info("  Training Random Forest...")
        rf = train_random_forest(X_rdkit_tr, y_tr, seed=seed)
        save_tabular_model(rf, models_dir / f"rf_seed{seed}.pkl")

        log.info("  Training XGBoost...")
        xgb = train_xgboost(X_rdkit_tr, y_tr, seed=seed)
        save_tabular_model(xgb, models_dir / f"xgb_seed{seed}.pkl")

        # ── B. Neural Latent Models (PolyBERT d=600) ──
        train_loader = DataLoader(neural_ds, batch_size=32, shuffle=True)
        for m_name in ["Concatenation", "LatentConvex", "Thermodynamic"]:
            set_seed(seed)
            model = build_neural_model(m_name, raw_dim=raw_dim, d=128)
            final_loss = train_neural_model(model, train_loader, epochs=60, lr=1e-3, device=device)
            save_path = models_dir / f"{m_name}_seed{seed}.pt"
            save_neural_model(model, save_path)
            log.info(f"  Trained {m_name:15s} (final MSE: {final_loss:.4f}) -> {save_path.name}")

        # ── C. Sequence Model (BigSMILES) ──
        set_seed(seed)
        seq_loader = DataLoader(seq_ds, batch_size=32, shuffle=True)
        seq_model = build_sequence_model(lm_dim=768, d=128)
        seq_loss = train_sequence_model(seq_model, seq_loader, epochs=60, lr=1e-3, device=device)
        seq_path = models_dir / f"BigSMILES_LM_seed{seed}.pt"
        save_sequence_model(seq_model, seq_path)
        log.info(f"  Trained {'BigSMILES_LM':15s} (final MSE: {seq_loss:.4f}) -> {seq_path.name}")

    # ── D. Classical Gordon-Taylor Parameter Fit on D_train ──
    log.info("\nFitting classical Gordon-Taylor parameter k on D_train pairs...")
    tr_blends = blends.iloc[train_idx].copy()
    tg_lookup = manifest["homopolymer_tg_lookup"]
    gt_fits = fit_all_pairs_gordon_taylor(tr_blends, tg_lookup)

    with open(models_dir / "gordon_taylor_fits.json", "w") as f:
        json.dump(gt_fits, f, indent=2)
    log.info(f"Saved Gordon-Taylor fitted k parameters for {len(gt_fits)} pairs.")
    log.info("\nALL CANDIDATE PARADIGMS TRAINED & CHECKPOINTED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
