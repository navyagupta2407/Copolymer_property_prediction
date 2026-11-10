"""
train_all_paradigms.py
======================
Step 1: Trains all industry-standard copolymer models ONCE on the unified
training set (D_train = 214 samples) across 5 independent random seeds.

Paradigms Trained:
1. Tabular: Random Forest & XGBoost on RDKit physical descriptors
2. Latent Neural (PolyBERT d=600): Concat, Convex, Thermodynamic Hybrid
3. Polymer Language Model: BigSMILES MLP
4. Classical Physics: Gordon-Taylor pair parameter fit

Saves all trained models and checkpoint weights to results/models/.
"""

import json
import logging
import math
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

# Configure writable matplotlib cache directory
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Neural Architectures ───────────────────────────────────────────────────────

class S1_ThermodynamicHybrid(nn.Module):
    def __init__(self, raw_dim: int = 600, d: int = 128):
        super().__init__()
        self.proj = nn.Linear(raw_dim, d)
        self.k_estimator = nn.Sequential(
            nn.Linear(2 * raw_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.excess_net = nn.Sequential(
            nn.Linear(raw_dim, 64),
            nn.Tanh(),
            nn.Linear(64, d, bias=False),
        )

    def forward(self, e_A, e_B, f, z_lm=None):
        z_A = self.proj(e_A)
        z_B = self.proj(e_B)
        k_hat = torch.nn.functional.softplus(self.k_estimator(torch.cat([e_A, e_B], dim=-1))) + 0.1
        denom = f + k_hat * (1.0 - f)
        z_gt = (f * z_A + k_hat * (1.0 - f) * z_B) / denom
        excess = f * (1.0 - f) * self.excess_net(torch.abs(e_A - e_B))
        return z_gt + excess

class S2_LatentConvex(nn.Module):
    def __init__(self, raw_dim: int = 600, d: int = 128):
        super().__init__()
        self.proj = nn.Linear(raw_dim, d)

    def forward(self, e_A, e_B, f, z_lm=None):
        z_A = self.proj(e_A)
        z_B = self.proj(e_B)
        return f * z_A + (1.0 - f) * z_B

class S3_UnconstrainedConcat(nn.Module):
    def __init__(self, raw_dim: int = 600, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * raw_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, e_A, e_B, f, z_lm=None):
        x = torch.cat([e_A, e_B, f], dim=-1)
        return self.net(x)

class S6_BigSMILES_LM(nn.Module):
    def __init__(self, lm_dim: int = 768, d: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(lm_dim, 256),
            nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, e_A, e_B, f, z_lm=None):
        if z_lm is None:
            raise ValueError("z_lm required for BigSMILES representation")
        return self.proj(z_lm)

class FixedDownstreamRegressor(nn.Module):
    def __init__(self, d: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d)
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        h = self.drop(self.act(self.ln1(self.fc1(z))))
        z_res = self.act(self.ln2(self.fc2(h)) + z)
        return self.head(z_res).squeeze(-1)

class FullCopolymerModel(nn.Module):
    def __init__(self, rep_module: nn.Module, head_module: nn.Module):
        super().__init__()
        self.rep = rep_module
        self.head = head_module

    def forward(self, e_A, e_B, f, z_lm=None):
        z = self.rep(e_A, e_B, f, z_lm=z_lm)
        return self.head(z)

# ── Main Training Loop ─────────────────────────────────────────────────────────

def main():
    base_dir = Path("C:/Projects/copolymers")
    raw_path = base_dir / "data/raw/experimental_dataset.csv"
    proc_dir = base_dir / "data/processed"
    models_dir = base_dir / "results/models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits manifest
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
    X_rdkit_all = rdkit_data["features"]
    y_all = rdkit_data["targets"]

    X_rdkit_tr = X_rdkit_all[train_idx]
    y_tr = y_all[train_idx]
    log.info(f"Loaded RDKit tabular features: {X_rdkit_tr.shape}")

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

    # Save target normalization parameters
    norm_meta = {"y_mean": y_mean, "y_std": y_std}
    with open(models_dir / "target_normalization.json", "w") as f:
        json.dump(norm_meta, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Target norm stats: mean = {y_mean:.2f} K, std = {y_std:.2f} K. Training device: {device}")

    # 6. Train across 5 seeds
    SEEDS = [42, 43, 44, 45, 46]
    NEURAL_MODELS = {
        "Concatenation": lambda: FullCopolymerModel(S3_UnconstrainedConcat(raw_dim, d=128), FixedDownstreamRegressor(128)),
        "LatentConvex": lambda: FullCopolymerModel(S2_LatentConvex(raw_dim, d=128), FixedDownstreamRegressor(128)),
        "Thermodynamic": lambda: FullCopolymerModel(S1_ThermodynamicHybrid(raw_dim, d=128), FixedDownstreamRegressor(128)),
        "BigSMILES_LM": lambda: FullCopolymerModel(S6_BigSMILES_LM(768, d=128), FixedDownstreamRegressor(128)),
    }

    train_ds = TensorDataset(ea_tr, eb_tr, f_tr, zlm_tr, y_tr_norm)

    for seed in SEEDS:
        log.info(f"\n=======================================================")
        log.info(f"             TRAINING SEED {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        log.info(f"=======================================================")
        set_seed(seed)

        # ── A. Tabular Models ──
        log.info("  Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed)
        rf.fit(X_rdkit_tr, y_tr)
        with open(models_dir / f"rf_seed{seed}.pkl", "wb") as f:
            pickle.dump(rf, f)

        log.info("  Training XGBoost...")
        xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=seed)
        xgb.fit(X_rdkit_tr, y_tr)
        with open(models_dir / f"xgb_seed{seed}.pkl", "wb") as f:
            pickle.dump(xgb, f)

        # ── B. Neural Models ──
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        for m_name, model_fn in NEURAL_MODELS.items():
            set_seed(seed)
            model = model_fn().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(60):
                running_loss = 0.0
                for b_ea, b_eb, b_f, b_zlm, b_y in train_loader:
                    b_ea, b_eb, b_f, b_zlm, b_y = (
                        b_ea.to(device), b_eb.to(device), b_f.to(device), b_zlm.to(device), b_y.to(device)
                    )
                    optimizer.zero_grad()
                    preds = model(b_ea, b_eb, b_f, z_lm=b_zlm)
                    loss = criterion(preds, b_y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * len(b_y)
                ep_loss = running_loss / len(train_ds)
                scheduler.step(ep_loss)

            save_path = models_dir / f"{m_name}_seed{seed}.pt"
            torch.save(model.state_dict(), save_path)
            log.info(f"  Trained & checkpointed {m_name:15s} (final train MSE: {ep_loss:.4f}) -> {save_path.name}")

    # ── C. Classical Gordon-Taylor Fit per pair on D_train ──
    log.info("\nFitting classical Gordon-Taylor parameter k on D_train pairs...")
    from scipy.optimize import minimize_scalar

    tr_blends = blends.iloc[train_idx].copy()
    tg_lookup = manifest["homopolymer_tg_lookup"]
    gt_fits = {}

    for pid in tr_blends["pair_id"].unique():
        sub = tr_blends[tr_blends["pair_id"] == pid]
        p_A, p_B = pid.split("-")
        tg_A = tg_lookup.get(p_A, sub["Tg_actual"].max())
        tg_B = tg_lookup.get(p_B, sub["Tg_actual"].min())

        if len(sub) >= 2:
            def loss_k(k_val):
                p = (sub["f"] * tg_A + k_val * (1.0 - sub["f"]) * tg_B) / (sub["f"] + k_val * (1.0 - sub["f"]))
                return float(np.mean((p - sub["Tg_actual"]) ** 2))
            res = minimize_scalar(loss_k, bounds=(0.05, 10.0), method="bounded")
            best_k = float(res.x if res.success else 1.0)
        else:
            best_k = 1.0
        gt_fits[pid] = best_k

    with open(models_dir / "gordon_taylor_fits.json", "w") as f:
        json.dump(gt_fits, f, indent=2)
    log.info(f"Saved Gordon-Taylor fitted k parameters for {len(gt_fits)} pairs.")
    log.info("\nALL CANDIDATE PARADIGMS TRAINED & CHECKPOINTED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
