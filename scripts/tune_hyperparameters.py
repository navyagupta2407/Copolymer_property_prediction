"""
tune_hyperparameters.py
=======================
Systematic Hyperparameter Tuning for Conventional Models WITHOUT Structural Changes.

Objective:
Optimize hyperparameters strictly on D_train (using 5-Fold Cross-Validation) to:
1. Reduce Overfitting in Tree Models (XGBoost, Random Forest) via regularization, depth control, and leaf constraints.
2. Reduce Underfitting in Deep Learning Models (Concatenation MLP, BigSMILES LM) via learning rate schedules, weight decay, and training dynamics.

Maintains strict separation:
- Original baseline models and results are 100% PRESERVED in results/models/.
- Tuned models saved in results/models_tuned/.
- Exports comparison table: results/hyperparameter_tuning_results.csv.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Matplotlib configuration for sandbox/environment stability
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import save_tabular_model, predict_tabular
from models.neural_models import build_neural_model, train_neural_model, save_neural_model, predict_neural
from models.sequence_models import build_sequence_model, train_sequence_model, save_sequence_model, predict_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tune_hp")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}

def main():
    base_dir = Path("C:/Projects/copolymers")
    proc_dir = base_dir / "data/processed"
    raw_path = base_dir / "data/raw/experimental_dataset.csv"
    results_dir = base_dir / "results"
    tuned_dir = results_dir / "models_tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    train_idx = manifest["train_indices"]
    s1_idx = manifest["suite1_random_indices"]
    s3_idx = manifest["suite3_one_sided_indices"]

    # 2. Load dataset tables
    df_raw = pd.read_csv(raw_path)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    blends["f"] = blends["comp_0"].astype(float) / 100.0

    # 3. Load Tabular RDKit features
    rdkit_data = np.load(proc_dir / "blend_rdkit_features.npz")
    X_rdkit_all = rdkit_data["features"]
    y_all = rdkit_data["targets"]

    X_rdkit_tr = X_rdkit_all[train_idx]
    y_tr = y_all[train_idx]

    # 4. Load PolyBERT embeddings
    pb_data = np.load(proc_dir / "monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])

    ea_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_0"]]), dtype=torch.float32)
    eb_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_1"]]), dtype=torch.float32)
    f_all = torch.tensor(blends["f"].values, dtype=torch.float32).unsqueeze(-1)

    ea_tr = ea_all[train_idx]
    eb_tr = eb_all[train_idx]
    f_tr = f_all[train_idx]

    # 5. Load BigSMILES ChemBERTa cache
    bs_cache = np.load(proc_dir / "bigsmiles_cache.npz")
    zlm_all = torch.tensor(bs_cache["blend_embs"], dtype=torch.float32)
    zlm_tr = zlm_all[train_idx]

    # Normalization parameters strictly from D_train
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() + 1e-7)
    y_tr_norm = torch.tensor((y_tr - y_mean) / y_std, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================================
    # PART 1: TUNING TABULAR TREE MODELS (Address Overfitting)
    # =========================================================================
    log.info("=" * 60)
    log.info("PART 1: Cross-Validation Tuning for Tree Models (XGBoost & RF)")
    log.info("=" * 60)

    # --- A. XGBoost Tuning ---
    xgb_param_grid = [
        # Baseline was: max_depth=5, lr=0.08, min_child=1, lambda=1.0
        {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3, "reg_lambda": 3.0, "subsample": 0.8},
        {"max_depth": 3, "learning_rate": 0.08, "min_child_weight": 5, "reg_lambda": 5.0, "subsample": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 4, "reg_lambda": 5.0, "subsample": 0.85},
        {"max_depth": 4, "learning_rate": 0.08, "min_child_weight": 3, "reg_lambda": 3.0, "subsample": 0.85},
        {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 5, "reg_lambda": 10.0, "subsample": 0.8},
    ]

    best_xgb_score = float("inf")
    best_xgb_params = None

    for params in xgb_param_grid:
        cv_rmses = []
        for tr_k, val_k in kf.split(X_rdkit_tr):
            m = XGBRegressor(
                n_estimators=100,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                min_child_weight=params["min_child_weight"],
                reg_lambda=params["reg_lambda"],
                subsample=params["subsample"],
                random_state=42,
                n_jobs=-1,
            )
            m.fit(X_rdkit_tr[tr_k], y_tr[tr_k])
            preds_val = m.predict(X_rdkit_tr[val_k])
            cv_rmses.append(np.sqrt(np.mean((preds_val - y_tr[val_k])**2)))
        mean_cv = np.mean(cv_rmses)
        log.info(f"  XGB params {params} -> 5-Fold CV RMSE: {mean_cv:.2f} K")
        if mean_cv < best_xgb_score:
            best_xgb_score = mean_cv
            best_xgb_params = params

    log.info(f"--> Optimal XGBoost configuration: {best_xgb_params} (CV RMSE = {best_xgb_score:.2f} K)")

    # --- B. Random Forest Tuning ---
    rf_param_grid = [
        # Baseline was: max_depth=12, min_samples_leaf=1, min_samples_split=2
        {"max_depth": 6, "min_samples_leaf": 2, "min_samples_split": 4, "max_features": "sqrt"},
        {"max_depth": 6, "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.7},
        {"max_depth": 8, "min_samples_leaf": 2, "min_samples_split": 4, "max_features": 0.7},
        {"max_depth": 8, "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.8},
        {"max_depth": 10, "min_samples_leaf": 2, "min_samples_split": 4, "max_features": 0.8},
    ]

    best_rf_score = float("inf")
    best_rf_params = None

    for params in rf_param_grid:
        cv_rmses = []
        for tr_k, val_k in kf.split(X_rdkit_tr):
            m = RandomForestRegressor(
                n_estimators=120,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                min_samples_split=params["min_samples_split"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1,
            )
            m.fit(X_rdkit_tr[tr_k], y_tr[tr_k])
            preds_val = m.predict(X_rdkit_tr[val_k])
            cv_rmses.append(np.sqrt(np.mean((preds_val - y_tr[val_k])**2)))
        mean_cv = np.mean(cv_rmses)
        log.info(f"  RF params {params} -> 5-Fold CV RMSE: {mean_cv:.2f} K")
        if mean_cv < best_rf_score:
            best_rf_score = mean_cv
            best_rf_params = params

    log.info(f"--> Optimal Random Forest configuration: {best_rf_params} (CV RMSE = {best_rf_score:.2f} K)")

    # =========================================================================
    # PART 2: TUNING DEEP LEARNING & SEQUENCE MODELS (Address Underfitting)
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("PART 2: Tuning Training Dynamics for Neural Models (Concat MLP & BigSMILES)")
    log.info("=" * 60)

    # For neural models without structural change:
    # Baseline was: lr=1e-3, epochs=100, weight_decay=1e-4, batch_size=32, no scheduler
    # Tuning strategies:
    # - Lower LR (3e-4 or 5e-4) to prevent premature plateauing
    # - Cosine Annealing learning rate schedule
    # - Slightly higher epochs (150) with lower weight decay (1e-5)
    mlp_opts = [
        {"lr": 5e-4, "weight_decay": 1e-5, "epochs": 150, "batch_size": 16},
        {"lr": 3e-4, "weight_decay": 1e-4, "epochs": 150, "batch_size": 16},
        {"lr": 5e-4, "weight_decay": 1e-4, "epochs": 120, "batch_size": 32},
    ]

    best_mlp_score = float("inf")
    best_mlp_opt = None

    for opt in mlp_opts:
        cv_rmses = []
        for tr_k, val_k in kf.split(X_rdkit_tr):
            # Build standard Concatenation model (NO structural change)
            m_cv = build_neural_model("Concatenation", raw_dim=raw_dim, d=128)
            tr_ds = TensorDataset(ea_tr[tr_k], eb_tr[tr_k], f_tr[tr_k], y_tr_norm[tr_k])
            loader = DataLoader(tr_ds, batch_size=opt["batch_size"], shuffle=True)
            optimizer = torch.optim.AdamW(m_cv.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])
            criterion = nn.MSELoss()

            m_cv.train()
            for _ in range(opt["epochs"]):
                for b_ea, b_eb, b_f, b_y in loader:
                    optimizer.zero_grad()
                    out = m_cv(b_ea, b_eb, b_f)
                    loss = criterion(out, b_y)
                    loss.backward()
                    optimizer.step()

            m_cv.eval()
            p_val = predict_neural(m_cv, ea_tr[val_k], eb_tr[val_k], f_tr[val_k], y_mean, y_std)
            cv_rmses.append(np.sqrt(np.mean((p_val - y_tr[val_k])**2)))
        mean_cv = np.mean(cv_rmses)
        log.info(f"  Concat MLP opts {opt} -> 5-Fold CV RMSE: {mean_cv:.2f} K")
        if mean_cv < best_mlp_score:
            best_mlp_score = mean_cv
            best_mlp_opt = opt

    log.info(f"--> Optimal Concat MLP configuration: {best_mlp_opt} (CV RMSE = {best_mlp_score:.2f} K)")

    # Sequence BigSMILES LM opts
    bs_opts = [
        {"lr": 5e-4, "weight_decay": 1e-5, "epochs": 140, "batch_size": 16},
        {"lr": 3e-4, "weight_decay": 1e-4, "epochs": 150, "batch_size": 16},
    ]
    best_bs_score = float("inf")
    best_bs_opt = None
    for opt in bs_opts:
        cv_rmses = []
        for tr_k, val_k in kf.split(X_rdkit_tr):
            m_cv = build_sequence_model(lm_dim=768, d=128)
            tr_ds = TensorDataset(zlm_tr[tr_k], y_tr_norm[tr_k])
            loader = DataLoader(tr_ds, batch_size=opt["batch_size"], shuffle=True)
            optimizer = torch.optim.AdamW(m_cv.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])
            criterion = nn.MSELoss()
            m_cv.train()
            for _ in range(opt["epochs"]):
                for b_z, b_y in loader:
                    optimizer.zero_grad()
                    out = m_cv(b_z)
                    loss = criterion(out, b_y)
                    loss.backward()
                    optimizer.step()
            m_cv.eval()
            p_val = predict_sequence(m_cv, zlm_tr[val_k], y_mean, y_std)
            cv_rmses.append(np.sqrt(np.mean((p_val - y_tr[val_k])**2)))
        mean_cv = np.mean(cv_rmses)
        log.info(f"  BigSMILES opts {opt} -> 5-Fold CV RMSE: {mean_cv:.2f} K")
        if mean_cv < best_bs_score:
            best_bs_score = mean_cv
            best_bs_opt = opt

    log.info(f"--> Optimal BigSMILES configuration: {best_bs_opt} (CV RMSE = {best_bs_score:.2f} K)")

    # =========================================================================
    # PART 3: TRAIN OPTIMAL TUNED MODELS ACROSS 5 SEEDS & EVALUATE
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("PART 3: Training Tuned Models on Full D_train (5 Seeds: 42..46)")
    log.info("=" * 60)

    SEEDS = [42, 43, 44, 45, 46]
    MODELS_TO_EVAL = ["Random Forest", "XGBoost", "Concatenation", "BigSMILES_LM"]

    # Storage for evaluation: {model: {split: [seed_metrics]}}
    tuned_results = {m: {"train": [], "s1": [], "s3": []} for m in MODELS_TO_EVAL}

    for seed in SEEDS:
        # A. Tuned XGBoost
        xgb_tuned = XGBRegressor(
            n_estimators=100,
            max_depth=best_xgb_params["max_depth"],
            learning_rate=best_xgb_params["learning_rate"],
            min_child_weight=best_xgb_params["min_child_weight"],
            reg_lambda=best_xgb_params["reg_lambda"],
            subsample=best_xgb_params["subsample"],
            random_state=seed,
            n_jobs=-1,
        )
        xgb_tuned.fit(X_rdkit_tr, y_tr)
        save_tabular_model(xgb_tuned, tuned_dir / f"xgb_tuned_seed{seed}.pkl")

        # B. Tuned RF
        rf_tuned = RandomForestRegressor(
            n_estimators=120,
            max_depth=best_rf_params["max_depth"],
            min_samples_leaf=best_rf_params["min_samples_leaf"],
            min_samples_split=best_rf_params["min_samples_split"],
            max_features=best_rf_params["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
        rf_tuned.fit(X_rdkit_tr, y_tr)
        save_tabular_model(rf_tuned, tuned_dir / f"rf_tuned_seed{seed}.pkl")

        # C. Tuned Concat MLP
        torch.manual_seed(seed)
        m_concat_tuned = build_neural_model("Concatenation", raw_dim=raw_dim, d=128)
        tr_ds_mlp = TensorDataset(ea_tr, eb_tr, f_tr, y_tr_norm)
        loader_mlp = DataLoader(tr_ds_mlp, batch_size=best_mlp_opt["batch_size"], shuffle=True)
        opt_mlp = torch.optim.AdamW(m_concat_tuned.parameters(), lr=best_mlp_opt["lr"], weight_decay=best_mlp_opt["weight_decay"])
        crit_mlp = nn.MSELoss()
        m_concat_tuned.train()
        for _ in range(best_mlp_opt["epochs"]):
            for b_ea, b_eb, b_f, b_y in loader_mlp:
                opt_mlp.zero_grad()
                out = m_concat_tuned(b_ea, b_eb, b_f)
                loss = crit_mlp(out, b_y)
                loss.backward()
                opt_mlp.step()
        save_neural_model(m_concat_tuned, tuned_dir / f"Concatenation_tuned_seed{seed}.pt")

        # D. Tuned BigSMILES
        m_bs_tuned = build_sequence_model(lm_dim=768, d=128)
        tr_ds_bs = TensorDataset(zlm_tr, y_tr_norm)
        loader_bs = DataLoader(tr_ds_bs, batch_size=best_bs_opt["batch_size"], shuffle=True)
        opt_bs = torch.optim.AdamW(m_bs_tuned.parameters(), lr=best_bs_opt["lr"], weight_decay=best_bs_opt["weight_decay"])
        m_bs_tuned.train()
        for _ in range(best_bs_opt["epochs"]):
            for b_z, b_y in loader_bs:
                opt_bs.zero_grad()
                out = m_bs_tuned(b_z)
                loss = crit_mlp(out, b_y)
                loss.backward()
                opt_bs.step()
        save_sequence_model(m_bs_tuned, tuned_dir / f"BigSMILES_LM_tuned_seed{seed}.pt")

        # Evaluate across Train, S1, S3
        splits = {"train": train_idx, "s1": s1_idx, "s3": s3_idx}
        for split_key, idxs in splits.items():
            y_sub = y_all[idxs]

            p_rf = predict_tabular(rf_tuned, X_rdkit_all[idxs])
            p_xgb = predict_tabular(xgb_tuned, X_rdkit_all[idxs])
            p_cat = predict_neural(m_concat_tuned, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)
            p_bs = predict_sequence(m_bs_tuned, zlm_all[idxs], y_mean, y_std)

            preds_dict = {
                "Random Forest": p_rf,
                "XGBoost": p_xgb,
                "Concatenation": p_cat,
                "BigSMILES_LM": p_bs,
            }

            for m_name in MODELS_TO_EVAL:
                metrics = compute_metrics(y_sub, preds_dict[m_name])
                tuned_results[m_name][split_key].append(metrics)

    # =========================================================================
    # PART 4: COMPILE COMPARISON TABLE (Baseline vs. Tuned)
    # =========================================================================
    # Load original baseline diagnostic summary for head-to-head comparison
    diag_summary = pd.read_csv(results_dir / "diagnostics/overfitting_underfitting_summary.csv")
    diag_map = {r["Paradigm"]: r for _, r in diag_summary.iterrows()}

    comparison_rows = []
    for m in MODELS_TO_EVAL:
        base_row = diag_map[m]

        # Baseline stats
        b_tr_rmse = float(base_row["Train RMSE (K)"].split("±")[0].strip())
        b_s1_rmse = float(base_row["Suite 1 (IID) RMSE (K)"].split("±")[0].strip())
        b_s3_rmse = float(base_row["Suite 3 (OOD) RMSE (K)"].split("±")[0].strip())
        b_gap_iid = float(base_row["Generalization Gap (IID)"].replace("K", "").replace("+", "").strip())

        # Tuned stats across seeds
        t_tr_rmses = [x["rmse"] for x in tuned_results[m]["train"]]
        t_s1_rmses = [x["rmse"] for x in tuned_results[m]["s1"]]
        t_s3_rmses = [x["rmse"] for x in tuned_results[m]["s3"]]

        t_tr_mean = np.mean(t_tr_rmses)
        t_s1_mean = np.mean(t_s1_rmses)
        t_s3_mean = np.mean(t_s3_rmses)
        t_gap_iid = t_s1_mean - t_tr_mean

        delta_s1_pct = ((t_s1_mean - b_s1_rmse) / b_s1_rmse) * 100.0

        comparison_rows.append({
            "Model": m,
            "Baseline Train (K)": f"{b_tr_rmse:.2f}",
            "Tuned Train (K)": f"{t_tr_mean:.2f} ± {np.std(t_tr_rmses):.2f}",
            "Baseline Suite 1 (K)": f"{b_s1_rmse:.2f}",
            "Tuned Suite 1 (K)": f"{t_s1_mean:.2f} ± {np.std(t_s1_rmses):.2f}",
            "Suite 1 Change": f"{delta_s1_pct:+.1f}%",
            "Baseline Gap (K)": f"+{b_gap_iid:.2f}",
            "Tuned Gap (K)": f"{t_gap_iid:+.2f}",
            "Baseline Suite 3 OOD (K)": f"{b_s3_rmse:.2f}",
            "Tuned Suite 3 OOD (K)": f"{t_s3_mean:.2f} ± {np.std(t_s3_rmses):.2f}",
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = results_dir / "hyperparameter_tuning_results.csv"
    comp_df.to_csv(comp_path, index=False)
    log.info(f"Saved comparison to {comp_path}")

    # Output formatted matrix
    print("\n" + "=" * 120)
    print("                HYPERPARAMETER TUNING (WITHOUT STRUCTURAL CHANGE) vs. BASELINE")
    print("=" * 120)
    print(comp_df.to_string(index=False))
    print("=" * 120)

    # Save best hyperparameter configs
    best_configs = {
        "XGBoost": best_xgb_params,
        "RandomForest": best_rf_params,
        "Concatenation": best_mlp_opt,
        "BigSMILES_LM": best_bs_opt,
    }
    with open(tuned_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_configs, f, indent=2)

if __name__ == "__main__":
    main()
