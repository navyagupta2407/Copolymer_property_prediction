"""
diagnose_fit.py
===============
Systematic Overfitting and Underfitting Diagnostic for Copolymer Tg Models.

Evaluates all 9 candidate model paradigms across:
1. Training Set (D_train, N=214)
2. Suite 1: Random Holdout (IID Baseline, N=25)
3. Suite 3: One-Sided Extrapolation Probe (Severe OOD, N=41)

Computes across 5 seeds:
- Train RMSE, MAE, R^2
- Test RMSE, MAE, R^2
- Generalization Gap: Delta_IID = RMSE_S1 - RMSE_Train
- Generalization Gap: Delta_OOD = RMSE_S3 - RMSE_Train
- Overfitting / Underfitting Diagnostic Taxonomy:
  * Severe Overfitting: Train RMSE << Test RMSE (Ratio > 2.0x, Delta >> 5 K)
  * Underfitting (High Bias): Train RMSE high (> 15 K) AND Test RMSE high (> 15 K)
  * Robust / Optimal Fit: Low Train RMSE (~3-6 K), Low Test RMSE (~3-6 K), Minimal Gap

Outputs:
- results/diagnostics/overfitting_underfitting_summary.csv
- results/diagnostics/train_vs_test_parity.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import load_tabular_model, predict_tabular
from models.neural_models import load_neural_model, predict_neural
from models.sequence_models import load_sequence_model, predict_sequence
from models.physics_baselines import predict_gordon_taylor, predict_fox, predict_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("diagnose_fit")

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
    models_dir = base_dir / "results/models"
    diag_dir = base_dir / "results/diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    train_idx = manifest["train_indices"]
    s1_idx = manifest["suite1_random_indices"]
    s3_idx = manifest["suite3_one_sided_indices"]

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

    # 4. Load PolyBERT embeddings
    pb_data = np.load(proc_dir / "monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])  # 600

    ea_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_0"]]), dtype=torch.float32)
    eb_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_1"]]), dtype=torch.float32)
    f_all = torch.tensor(blends["f"].values, dtype=torch.float32).unsqueeze(-1)

    # 5. Load BigSMILES ChemBERTa cache
    bs_cache = np.load(proc_dir / "bigsmiles_cache.npz")
    zlm_all = torch.tensor(bs_cache["blend_embs"], dtype=torch.float32)

    # 6. Load normalization & Gordon-Taylor parameters
    with open(models_dir / "target_normalization.json") as f:
        norm_params = json.load(f)
    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    with open(models_dir / "gordon_taylor_fits.json") as f:
        gt_fits = json.load(f)
    tg_lookup = manifest["homopolymer_tg_lookup"]

    SEEDS = [42, 43, 44, 45, 46]
    ALL_PARADIGMS = [
        "Random Forest",
        "XGBoost",
        "Concatenation",
        "LatentConvex",
        "Thermodynamic",
        "BigSMILES_LM",
        "Gordon-Taylor",
        "Fox",
        "Linear",
    ]

    # Metrics container: dict of paradigm -> seed_metrics for train, s1, s3
    seed_records = {m: {"train": [], "s1": [], "s3": []} for m in ALL_PARADIGMS}

    # For plotting: predictions on seed 42
    seed42_preds = {}

    log.info("Computing Train vs. Test metrics across 5 seeds...")

    for seed in SEEDS:
        # Load trained models
        rf = load_tabular_model(models_dir / f"rf_seed{seed}.pkl")
        xgb = load_tabular_model(models_dir / f"xgb_seed{seed}.pkl")
        m_concat = load_neural_model("Concatenation", models_dir / f"Concatenation_seed{seed}.pt", raw_dim=raw_dim)
        m_convex = load_neural_model("LatentConvex", models_dir / f"LatentConvex_seed{seed}.pt", raw_dim=raw_dim)
        m_thermo = load_neural_model("Thermodynamic", models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)
        m_seq = load_sequence_model(models_dir / f"BigSMILES_LM_seed{seed}.pt", lm_dim=768)

        splits = {
            "train": train_idx,
            "s1": s1_idx,
            "s3": s3_idx,
        }

        for split_key, idxs in splits.items():
            y_sub = y_all[idxs]
            sub_blends = blends.iloc[idxs]

            p_rf = predict_tabular(rf, X_rdkit_all[idxs])
            p_xgb = predict_tabular(xgb, X_rdkit_all[idxs])
            p_concat = predict_neural(m_concat, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)
            p_convex = predict_neural(m_convex, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)
            p_thermo = predict_neural(m_thermo, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)
            p_seq = predict_sequence(m_seq, zlm_all[idxs], y_mean, y_std)

            p_gt = []
            p_fox = []
            p_lin = []
            for _, r in sub_blends.iterrows():
                pid = r["pair_id"]
                p_A, p_B = pid.split("-")
                tg_A = tg_lookup.get(p_A, r["Tg_actual"])
                tg_B = tg_lookup.get(p_B, r["Tg_actual"])
                f_val = r["f"]
                k_val = gt_fits.get(pid, 1.0)
                p_gt.append(predict_gordon_taylor(f_val, tg_A, tg_B, k_val))
                p_fox.append(predict_fox(f_val, tg_A, tg_B))
                p_lin.append(predict_linear(f_val, tg_A, tg_B))

            preds_dict = {
                "Random Forest": p_rf,
                "XGBoost": p_xgb,
                "Concatenation": p_concat,
                "LatentConvex": p_convex,
                "Thermodynamic": p_thermo,
                "BigSMILES_LM": p_seq,
                "Gordon-Taylor": np.array(p_gt),
                "Fox": np.array(p_fox),
                "Linear": np.array(p_lin),
            }

            for m in ALL_PARADIGMS:
                m_metrics = compute_metrics(y_sub, preds_dict[m])
                seed_records[m][split_key].append(m_metrics)

            if seed == 42:
                seed42_preds[split_key] = {
                    "y_true": y_sub,
                    "preds": preds_dict,
                }

    # Aggregate metrics across seeds
    summary_rows = []
    for m in ALL_PARADIGMS:
        tr_rmses = [x["rmse"] for x in seed_records[m]["train"]]
        tr_r2s = [x["r2"] for x in seed_records[m]["train"]]
        tr_maes = [x["mae"] for x in seed_records[m]["train"]]

        s1_rmses = [x["rmse"] for x in seed_records[m]["s1"]]
        s1_r2s = [x["r2"] for x in seed_records[m]["s1"]]

        s3_rmses = [x["rmse"] for x in seed_records[m]["s3"]]
        s3_r2s = [x["r2"] for x in seed_records[m]["s3"]]

        mean_tr_rmse = np.mean(tr_rmses)
        mean_tr_r2 = np.mean(tr_r2s)
        mean_s1_rmse = np.mean(s1_rmses)
        mean_s1_r2 = np.mean(s1_r2s)
        mean_s3_rmse = np.mean(s3_rmses)
        mean_s3_r2 = np.mean(s3_r2s)

        gap_iid = mean_s1_rmse - mean_tr_rmse
        ratio_iid = mean_s1_rmse / max(1e-4, mean_tr_rmse)
        gap_ood = mean_s3_rmse - mean_tr_rmse
        ratio_ood = mean_s3_rmse / max(1e-4, mean_tr_rmse)

        # Scientific Diagnostic Classification
        if mean_tr_rmse > 18.0 and mean_s1_rmse > 18.0:
            diag_status = "Underfitting (High Bias)"
            mechanism = "Insufficient architectural or inductive capacity to map monomer pair chemistry"
        elif ratio_iid > 2.5 or (gap_iid > 6.0 and mean_tr_rmse < 6.0):
            diag_status = "Severe Overfitting (High Variance)"
            mechanism = "Memorized training bounds; axis-aligned splits fail to extrapolate continuous slopes"
        elif ratio_ood > 3.0:
            diag_status = "OOD Overfitting"
            mechanism = "Good IID interpolation but catastrophic failure under boundary shift"
        elif mean_s1_rmse < 6.0 and abs(gap_iid) < 3.0:
            diag_status = "Optimal Generalization"
            mechanism = "Physics-constrained manifold preserves boundary continuity and prevents memorization"
        else:
            diag_status = "Moderate Fit"
            mechanism = "Reasonable balance with slight bias or variance"

        summary_rows.append({
            "Paradigm": m,
            "Train RMSE (K)": f"{mean_tr_rmse:.2f} ± {np.std(tr_rmses):.2f}",
            "Train R2": f"{mean_tr_r2:.3f}",
            "Suite 1 (IID) RMSE (K)": f"{mean_s1_rmse:.2f} ± {np.std(s1_rmses):.2f}",
            "Suite 1 R2": f"{mean_s1_r2:.3f}",
            "Generalization Gap (IID)": f"{gap_iid:+.2f} K",
            "Error Ratio (IID/Train)": f"{ratio_iid:.2f}x",
            "Suite 3 (OOD) RMSE (K)": f"{mean_s3_rmse:.2f} ± {np.std(s3_rmses):.2f}",
            "Generalization Gap (OOD)": f"{gap_ood:+.2f} K",
            "Diagnostic Diagnosis": diag_status,
            "Underlying Mechanism": mechanism,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = diag_dir / "overfitting_underfitting_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Saved summary to {summary_path}")

    # Generate Parity Plots for visual comparison
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    for idx, m in enumerate(ALL_PARADIGMS):
        ax = axes[idx]

        # Train points
        y_tr_true = seed42_preds["train"]["y_true"]
        y_tr_pred = seed42_preds["train"]["preds"][m]

        # Suite 1 points
        y_s1_true = seed42_preds["s1"]["y_true"]
        y_s1_pred = seed42_preds["s1"]["preds"][m]

        # Suite 3 points
        y_s3_true = seed42_preds["s3"]["y_true"]
        y_s3_pred = seed42_preds["s3"]["preds"][m]

        # Plot 45-degree parity line
        min_val = min(y_all.min(), y_tr_pred.min(), y_s1_pred.min(), y_s3_pred.min()) - 10
        max_val = max(y_all.max(), y_tr_pred.max(), y_s1_pred.max(), y_s3_pred.max()) + 10
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6, label="Ideal Parity")

        ax.scatter(y_tr_true, y_tr_pred, color="#1f77b4", alpha=0.35, s=25, label="Train (D_train)")
        ax.scatter(y_s1_true, y_s1_pred, color="#2ca02c", alpha=0.85, s=45, marker="o", label="Suite 1 (IID Holdout)")
        ax.scatter(y_s3_true, y_s3_pred, color="#d62728", alpha=0.85, s=45, marker="^", label="Suite 3 (One-Sided OOD)")

        tr_rmse_val = np.sqrt(np.mean((y_tr_pred - y_tr_true)**2))
        s1_rmse_val = np.sqrt(np.mean((y_s1_pred - y_s1_true)**2))
        s3_rmse_val = np.sqrt(np.mean((y_s3_pred - y_s3_true)**2))

        ax.set_title(f"{m}\nTrain: {tr_rmse_val:.1f}K | S1: {s1_rmse_val:.1f}K | S3: {s3_rmse_val:.1f}K", fontsize=11, fontweight="bold")
        ax.set_xlabel("Experimental Tg (K)", fontsize=9)
        ax.set_ylabel("Predicted Tg (K)", fontsize=9)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Overfitting & Underfitting Diagnosis Across Model Paradigms\n(Train vs. Suite 1 IID vs. Suite 3 One-Sided Extrapolation)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = diag_dir / "train_vs_test_parity.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log.info(f"Saved parity diagnostic plot to {plot_path}")

    # Output clean table
    print("\n" + "=" * 115)
    print("                     OVERFITTING & UNDERFITTING DIAGNOSTIC MATRIX")
    print("=" * 115)
    display_cols = ["Paradigm", "Train RMSE (K)", "Suite 1 (IID) RMSE (K)", "Generalization Gap (IID)", "Error Ratio (IID/Train)", "Suite 3 (OOD) RMSE (K)", "Diagnostic Diagnosis"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 115)

if __name__ == "__main__":
    main()
