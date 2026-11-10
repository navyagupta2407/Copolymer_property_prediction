"""
experiment_physics_trees.py
===========================
Experiments on Rescuing TUNED Tabular Decision Trees using Physics-Informed Priors:

1. Method A: Physical Feature Augmentation (Input-Level Physics)
   - Augments 41 RDKit features with:
     * Fox harmonic estimate: T_g,Fox(f)
     * Linear mixture estimate: T_g,Linear(f)
     * Bilinear interaction term: f * (1 - f)
     * Homopolymer endpoints: T_g,A, T_g,B
     * Endpoint contrast: |T_g,A - T_g,B|

2. Method B: Physics-Residual Hybrid Trees (Structural Physics)
   - Sub-variant B1: Fox Equation + Tuned Tree Residual
     y_hat = T_g,Fox(f) + Tree_residual(x_rdkit)
   - Sub-variant B2: Gordon-Taylor + Tuned Tree Residual
     y_hat = T_g,GordonTaylor(f, k) + Tree_residual(x_rdkit)

Uses the TUNED Tabular Baselines (XGBoost (Tuned) and Random Forest (Tuned))
and tuned hyperparameters from results/models_tuned/best_hyperparameters.json.
Also computes Option A Matched-Pair Random Control drops on Suite 3.

Exports:
- results/physics_trees_results.csv
- results/figures/fig11_physics_trees_comparison.png
- results/figures/fig12_physics_trees_trajectories.png
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import load_tabular_model, predict_tabular
from models.neural_models import load_neural_model, predict_neural
from models.physics_baselines import predict_gordon_taylor, predict_fox, predict_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phys_trees")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
})

BASE_DIR = Path("C:/Projects/copolymers")
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/Suhani  Sharma/.gemini/antigravity/brain/716101ab-b1de-4a5f-b2b4-2f12d72f1910")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}

def main():
    proc_dir = BASE_DIR / "data/processed"
    raw_path = BASE_DIR / "data/raw/experimental_dataset.csv"
    base_models_dir = RESULTS_DIR / "models"
    tuned_models_dir = RESULTS_DIR / "models_tuned"
    ctrl_models_dir = RESULTS_DIR / "models_counterfactual_control"

    # 1. Load splits & tuned hyperparameters
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    with open(proc_dir / "counterfactual_random_splits_manifest.json") as f:
        ctrl_manifest = json.load(f)
    with open(tuned_models_dir / "best_hyperparameters.json") as f:
        best_hp = json.load(f)

    xgb_params = best_hp["XGBoost"]
    rf_params = best_hp["RandomForest"]

    train_idx = manifest["train_indices"]
    s1_idx = manifest["suite1_random_indices"]
    s2_idx = manifest["suite2_core_to_tail_indices"]
    s3_idx = manifest["suite3_one_sided_indices"]
    s4_idx = manifest["suite4_gap_interpolation_indices"]

    ctrl_train_idx = ctrl_manifest["train_indices"]
    c3_idx = ctrl_manifest["control3_matched_one_sided_pairs_indices"]

    suites = {
        "Train": train_idx,
        "Suite 1 (IID)": s1_idx,
        "Suite 2 (Core-to-Tail)": s2_idx,
        "Suite 3 (One-Sided)": s3_idx,
        "Suite 4 (Equimolar Gap)": s4_idx,
    }

    # 2. Load dataset tables
    df_raw = pd.read_csv(raw_path)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    blends["pair_id"] = blends["Poly1"].astype(str) + "-" + blends["Poly2"].astype(str)
    blends["f"] = blends["comp_0"].astype(float) / 100.0

    # 3. Load RDKit features
    rdkit_data = np.load(proc_dir / "blend_rdkit_features.npz")
    X_rdkit_all = rdkit_data["features"]
    y_all = rdkit_data["targets"]

    # 4. Homopolymer lookup & Gordon-Taylor fits
    tg_lookup = manifest["homopolymer_tg_lookup"]
    with open(base_models_dir / "gordon_taylor_fits.json") as f:
        gt_fits = json.load(f)
    with open(ctrl_models_dir / "gordon_taylor_fits.json") as f:
        gt_fits_ctrl = json.load(f)

    fox_all, linear_all, gt_all, gt_ctrl_all, tg_A_all, tg_B_all = [], [], [], [], [], []
    for _, r in blends.iterrows():
        pid = r["pair_id"]
        p_A, p_B = pid.split("-")
        tg_A = tg_lookup.get(p_A, r["Tg(K)"])
        tg_B = tg_lookup.get(p_B, r["Tg(K)"])
        f_val = r["f"]
        fox_all.append(predict_fox(f_val, tg_A, tg_B))
        linear_all.append(predict_linear(f_val, tg_A, tg_B))
        gt_all.append(predict_gordon_taylor(f_val, tg_A, tg_B, gt_fits.get(pid, 1.0)))
        gt_ctrl_all.append(predict_gordon_taylor(f_val, tg_A, tg_B, gt_fits_ctrl.get(pid, 1.0)))
        tg_A_all.append(tg_A)
        tg_B_all.append(tg_B)

    fox_all = np.array(fox_all, dtype=np.float32)
    linear_all = np.array(linear_all, dtype=np.float32)
    gt_all = np.array(gt_all, dtype=np.float32)
    gt_ctrl_all = np.array(gt_ctrl_all, dtype=np.float32)
    tg_A_all = np.array(tg_A_all, dtype=np.float32)
    tg_B_all = np.array(tg_B_all, dtype=np.float32)
    f_arr = blends["f"].values.astype(np.float32)
    bilinear_f = f_arr * (1.0 - f_arr)
    tg_diff = np.abs(tg_A_all - tg_B_all)

    # Construct Method A: Augmented Features (47 features)
    X_aug_all = np.column_stack([
        X_rdkit_all,
        fox_all,
        linear_all,
        bilinear_f,
        tg_A_all,
        tg_B_all,
        tg_diff
    ])
    log.info(f"Method A Augmented Feature shape: {X_aug_all.shape} (47 features per blend)")

    # 5. Load PolyBERT for Thermodynamic Hybrid benchmark
    pb_data = np.load(proc_dir / "monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])

    ea_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_0"]]), dtype=torch.float32)
    eb_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_1"]]), dtype=torch.float32)
    f_torch = torch.tensor(blends["f"].values, dtype=torch.float32).unsqueeze(-1)

    with open(base_models_dir / "target_normalization.json") as f:
        norm_params = json.load(f)
    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    with open(ctrl_models_dir / "target_normalization.json") as f:
        norm_ctrl = json.load(f)

    SEEDS = [42, 43, 44, 45, 46]

    MODELS_TO_TEST = [
        "XGBoost (Tuned)",
        "Method A: XGBoost (Tuned) + Physical Features",
        "Method B1: Fox + XGBoost (Tuned) Residual",
        "Method B2: Gordon-Taylor + XGBoost (Tuned) Residual",
        "Random Forest (Tuned)",
        "Method A: RF (Tuned) + Physical Features",
        "Method B2: Gordon-Taylor + RF (Tuned) Residual",
        "Thermodynamic Hybrid (Gold Standard)",
    ]

    metrics_storage = {m: {s: [] for s in suites} for m in MODELS_TO_TEST}
    c3_storage = {m: [] for m in MODELS_TO_TEST}

    log.info("Running Training & Evaluation across 5 seeds using TUNED tree baselines...")

    for seed in SEEDS:
        # 1. Tuned Baseline Models
        xgb_base = load_tabular_model(tuned_models_dir / f"xgb_tuned_seed{seed}.pkl")
        rf_base = load_tabular_model(tuned_models_dir / f"rf_tuned_seed{seed}.pkl")
        m_thermo = load_neural_model("Thermodynamic", base_models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)

        # 2. Method A Models (Trained on X_aug with Tuned Hyperparameters)
        xgb_aug = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1)
        xgb_aug.fit(X_aug_all[train_idx], y_all[train_idx])

        rf_aug = RandomForestRegressor(n_estimators=120, **rf_params, random_state=seed, n_jobs=-1)
        rf_aug.fit(X_aug_all[train_idx], y_all[train_idx])

        # 3. Method B1: Fox + Tuned Tree Residual
        res_fox_tr = y_all[train_idx] - fox_all[train_idx]
        xgb_res_fox = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1)
        xgb_res_fox.fit(X_rdkit_all[train_idx], res_fox_tr)

        # 4. Method B2: Gordon-Taylor + Tuned Tree Residual
        res_gt_tr = y_all[train_idx] - gt_all[train_idx]
        xgb_res_gt = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1)
        xgb_res_gt.fit(X_rdkit_all[train_idx], res_gt_tr)

        rf_res_gt = RandomForestRegressor(n_estimators=120, **rf_params, random_state=seed, n_jobs=-1)
        rf_res_gt.fit(X_rdkit_all[train_idx], res_gt_tr)

        # Evaluate on all stress suites
        for s_name, idxs in suites.items():
            y_sub = y_all[idxs]

            p_xgb_base = predict_tabular(xgb_base, X_rdkit_all[idxs])
            p_rf_base = predict_tabular(rf_base, X_rdkit_all[idxs])
            p_thermo = predict_neural(m_thermo, ea_all[idxs], eb_all[idxs], f_torch[idxs], y_mean, y_std)

            p_xgb_aug = predict_tabular(xgb_aug, X_aug_all[idxs])
            p_rf_aug = predict_tabular(rf_aug, X_aug_all[idxs])

            p_b1_fox_xgb = fox_all[idxs] + predict_tabular(xgb_res_fox, X_rdkit_all[idxs])
            p_b2_gt_xgb = gt_all[idxs] + predict_tabular(xgb_res_gt, X_rdkit_all[idxs])
            p_b2_gt_rf = gt_all[idxs] + predict_tabular(rf_res_gt, X_rdkit_all[idxs])

            preds_dict = {
                "XGBoost (Tuned)": p_xgb_base,
                "Method A: XGBoost (Tuned) + Physical Features": p_xgb_aug,
                "Method B1: Fox + XGBoost (Tuned) Residual": p_b1_fox_xgb,
                "Method B2: Gordon-Taylor + XGBoost (Tuned) Residual": p_b2_gt_xgb,
                "Random Forest (Tuned)": p_rf_base,
                "Method A: RF (Tuned) + Physical Features": p_rf_aug,
                "Method B2: Gordon-Taylor + RF (Tuned) Residual": p_b2_gt_rf,
                "Thermodynamic Hybrid (Gold Standard)": p_thermo,
            }

            for m in MODELS_TO_TEST:
                metrics_storage[m][s_name].append(compute_metrics(y_sub, preds_dict[m]))

        # Also evaluate Option A Control 3 (Matched-Pair Random Holdout on the 20 One-Sided Pairs)
        xgb_base_c = load_tabular_model(ctrl_models_dir / f"xgb_tuned_ctrl_seed{seed}.pkl")
        rf_base_c = load_tabular_model(ctrl_models_dir / f"rf_tuned_ctrl_seed{seed}.pkl")
        m_thermo_c = load_neural_model("Thermodynamic", ctrl_models_dir / f"Thermodynamic_ctrl_seed{seed}.pt", raw_dim=raw_dim)

        xgb_aug_c = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1).fit(X_aug_all[ctrl_train_idx], y_all[ctrl_train_idx])
        rf_aug_c = RandomForestRegressor(n_estimators=120, **rf_params, random_state=seed, n_jobs=-1).fit(X_aug_all[ctrl_train_idx], y_all[ctrl_train_idx])
        xgb_res_fox_c = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1).fit(X_rdkit_all[ctrl_train_idx], y_all[ctrl_train_idx] - fox_all[ctrl_train_idx])
        xgb_res_gt_c = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1).fit(X_rdkit_all[ctrl_train_idx], y_all[ctrl_train_idx] - gt_ctrl_all[ctrl_train_idx])
        rf_res_gt_c = RandomForestRegressor(n_estimators=120, **rf_params, random_state=seed, n_jobs=-1).fit(X_rdkit_all[ctrl_train_idx], y_all[ctrl_train_idx] - gt_ctrl_all[ctrl_train_idx])

        y_c3 = y_all[c3_idx]
        preds_c3 = {
            "XGBoost (Tuned)": predict_tabular(xgb_base_c, X_rdkit_all[c3_idx]),
            "Method A: XGBoost (Tuned) + Physical Features": predict_tabular(xgb_aug_c, X_aug_all[c3_idx]),
            "Method B1: Fox + XGBoost (Tuned) Residual": fox_all[c3_idx] + predict_tabular(xgb_res_fox_c, X_rdkit_all[c3_idx]),
            "Method B2: Gordon-Taylor + XGBoost (Tuned) Residual": gt_ctrl_all[c3_idx] + predict_tabular(xgb_res_gt_c, X_rdkit_all[c3_idx]),
            "Random Forest (Tuned)": predict_tabular(rf_base_c, X_rdkit_all[c3_idx]),
            "Method A: RF (Tuned) + Physical Features": predict_tabular(rf_aug_c, X_aug_all[c3_idx]),
            "Method B2: Gordon-Taylor + RF (Tuned) Residual": gt_ctrl_all[c3_idx] + predict_tabular(rf_res_gt_c, X_rdkit_all[c3_idx]),
            "Thermodynamic Hybrid (Gold Standard)": predict_neural(m_thermo_c, ea_all[c3_idx], eb_all[c3_idx], f_torch[c3_idx], norm_ctrl["y_mean"], norm_ctrl["y_std"]),
        }
        for m in MODELS_TO_TEST:
            c3_storage[m].append(compute_metrics(y_c3, preds_c3[m]))

    # Compile Table of Results
    summary_rows = []
    for m in MODELS_TO_TEST:
        tr_rmse = np.mean([entry["rmse"] for entry in metrics_storage[m]["Train"]])
        s1_rmse = np.mean([entry["rmse"] for entry in metrics_storage[m]["Suite 1 (IID)"]])
        s2_rmse = np.mean([entry["rmse"] for entry in metrics_storage[m]["Suite 2 (Core-to-Tail)"]])
        s3_rmse = np.mean([entry["rmse"] for entry in metrics_storage[m]["Suite 3 (One-Sided)"]])
        s4_rmse = np.mean([entry["rmse"] for entry in metrics_storage[m]["Suite 4 (Equimolar Gap)"]])
        c3_rmse = np.mean([entry["rmse"] for entry in c3_storage[m]])

        s1_std = np.std([entry["rmse"] for entry in metrics_storage[m]["Suite 1 (IID)"]])
        s3_std = np.std([entry["rmse"] for entry in metrics_storage[m]["Suite 3 (One-Sided)"]])

        delta_s3_vs_s1 = ((s3_rmse - s1_rmse) / s1_rmse) * 100.0
        delta_s3_matched = ((s3_rmse - c3_rmse) / c3_rmse) * 100.0

        summary_rows.append({
            "Paradigm": m,
            "Train RMSE (K)": f"{tr_rmse:.2f}",
            "Suite 1 (IID) RMSE (K)": f"{s1_rmse:.2f} ± {s1_std:.2f}",
            "Suite 2 (Core-to-Tail) RMSE (K)": f"{s2_rmse:.2f}",
            "Suite 3 (One-Sided) RMSE (K)": f"{s3_rmse:.2f} ± {s3_std:.2f}",
            "Control 3 Rand (K)": f"{c3_rmse:.2f}",
            "Matched Drop S3 (%)": f"{delta_s3_matched:+.1f}%",
            "Raw Drop S3 vs S1 (%)": f"{delta_s3_vs_s1:+.1f}%",
            "Suite 4 (Equimolar Gap) RMSE (K)": f"{s4_rmse:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_out = RESULTS_DIR / "physics_trees_results.csv"
    summary_df.to_csv(csv_out, index=False)
    log.info(f"Saved results table to {csv_out}")

    # =========================================================================
    # FIGURE 11: BAR CHART COMPARISON (Tuned Baseline vs Method A vs Method B)
    # =========================================================================
    log.info("Generating Figure 11: Physics Trees Multi-Suite Comparison (Tuned)...")
    fig, ax = plt.subplots(figsize=(16, 7))

    models_plot = [
        "XGBoost (Tuned)",
        "Method A: XGBoost (Tuned) + Physical Features",
        "Method B1: Fox + XGBoost (Tuned) Residual",
        "Method B2: Gordon-Taylor + XGBoost (Tuned) Residual",
        "Thermodynamic Hybrid (Gold Standard)",
    ]

    suite_keys = ["Train", "Suite 1 (IID)", "Suite 2 (Core-to-Tail)", "Suite 3 (One-Sided)", "Suite 4 (Equimolar Gap)"]
    x = np.arange(len(suite_keys))
    width = 0.16

    colors = {
        "XGBoost (Tuned)": "#d95f02",
        "Method A: XGBoost (Tuned) + Physical Features": "#e6ab02",
        "Method B1: Fox + XGBoost (Tuned) Residual": "#7570b3",
        "Method B2: Gordon-Taylor + XGBoost (Tuned) Residual": "#2b5c8f",
        "Thermodynamic Hybrid (Gold Standard)": "#1b9e77",
    }

    for i, m in enumerate(models_plot):
        mean_rmses = [np.mean([entry["rmse"] for entry in metrics_storage[m][s]]) for s in suite_keys]
        std_rmses = [np.std([entry["rmse"] for entry in metrics_storage[m][s]]) for s in suite_keys]

        offset = (i - 2) * width
        rects = ax.bar(x + offset, mean_rmses, width, yerr=std_rmses, capsize=3,
                       label=m, color=colors[m], edgecolor="black", linewidth=0.8, alpha=0.9)

        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.6, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Root Mean Squared Error, RMSE (Kelvin)", fontsize=12, fontweight="bold")
    ax.set_title("Rescuing Tuned Tabular Workflows: Can Physics-Informed Priors Stabilize Decision Trees?\n(Tuned XGBoost vs. Feature Augmentation vs. Physics-Residual Trees)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(suite_keys, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    out_f11 = FIG_DIR / "fig11_physics_trees_comparison.png"
    plt.savefig(out_f11, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_f11}")

    # =========================================================================
    # FIGURE 12: CONTINUOUS TRAJECTORIES SWEEP (f in [0, 1])
    # =========================================================================
    log.info("Generating Figure 12: Continuous Trajectory Comparison (Tuned)...")
    df_desc = pd.read_csv(proc_dir / "monomer_rdkit_descriptors.csv")
    desc_dict = {r["smiles"]: r.drop(["monomer", "smiles"]).values.astype(np.float32) for _, r in df_desc.iterrows()}

    seed = 42
    xgb_base = load_tabular_model(tuned_models_dir / f"xgb_tuned_seed{seed}.pkl")
    m_thermo = load_neural_model("Thermodynamic", base_models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)

    xgb_aug = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1)
    xgb_aug.fit(X_aug_all[train_idx], y_all[train_idx])

    xgb_res_gt = XGBRegressor(n_estimators=100, **xgb_params, random_state=seed, n_jobs=-1)
    xgb_res_gt.fit(X_rdkit_all[train_idx], y_all[train_idx] - gt_all[train_idx])

    selected_pairs = ["MA-MMA", "BA-MMA"]
    f_dense = np.linspace(0.01, 0.99, 150)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for p_idx, pair_id in enumerate(selected_pairs):
        ax = axes[p_idx]
        polyA, polyB = pair_id.split("-")
        sub = blends[blends["pair_id"] == pair_id]

        smiles_A = sub.iloc[0]["SMILES_0"]
        smiles_B = sub.iloc[0]["SMILES_1"]
        tg_A = tg_lookup.get(polyA, 378.0)
        tg_B = tg_lookup.get(polyB, 295.0)
        k_val = gt_fits.get(pair_id, 1.0)

        desc_A = desc_dict[smiles_A]
        desc_B = desc_dict[smiles_B]

        X_dense_rdkit = []
        X_dense_aug = []
        gt_dense = []

        for f_val in f_dense:
            d_ideal = f_val * desc_A + (1.0 - f_val) * desc_B
            d_diff = np.abs(desc_A - desc_B)
            feat_rdkit = np.concatenate([d_ideal, d_diff, [f_val]])
            X_dense_rdkit.append(feat_rdkit)

            fox_val = predict_fox(f_val, tg_A, tg_B)
            lin_val = predict_linear(f_val, tg_A, tg_B)
            gt_val = predict_gordon_taylor(f_val, tg_A, tg_B, k_val)
            gt_dense.append(gt_val)

            feat_aug = np.concatenate([
                feat_rdkit,
                [fox_val, lin_val, f_val * (1.0 - f_val), tg_A, tg_B, abs(tg_A - tg_B)]
            ])
            X_dense_aug.append(feat_aug)

        X_dense_rdkit = np.array(X_dense_rdkit, dtype=np.float32)
        X_dense_aug = np.array(X_dense_aug, dtype=np.float32)
        gt_dense = np.array(gt_dense, dtype=np.float32)

        y_base = predict_tabular(xgb_base, X_dense_rdkit)
        y_aug = predict_tabular(xgb_aug, X_dense_aug)
        y_res_gt = gt_dense + predict_tabular(xgb_res_gt, X_dense_rdkit)

        ea = torch.tensor(pb_embs[smiles_to_pb_idx[smiles_A]], dtype=torch.float32).unsqueeze(0).repeat(len(f_dense), 1)
        eb = torch.tensor(pb_embs[smiles_to_pb_idx[smiles_B]], dtype=torch.float32).unsqueeze(0).repeat(len(f_dense), 1)
        f_t = torch.tensor(f_dense, dtype=torch.float32).unsqueeze(-1)
        y_th_dense = predict_neural(m_thermo, ea, eb, f_t, y_mean, y_std)

        ax.plot(f_dense, y_base, color="#d95f02", lw=2.0, linestyle=":", label="Tuned XGBoost (Extrapolation Plateau)")
        ax.plot(f_dense, y_aug, color="#e6ab02", lw=2.2, linestyle="--", label="Method A: Tuned XGBoost + Physical Features")
        ax.plot(f_dense, y_res_gt, color="#2b5c8f", lw=2.5, linestyle="-.", label="Method B: Gordon-Taylor + Tuned XGB Residual")
        ax.plot(f_dense, y_th_dense, color="#1b9e77", lw=3.0, linestyle="-", label="Thermodynamic Hybrid (Gold Standard)")

        train_sub = sub[sub.index.isin(train_idx)]
        test_sub = sub[~sub.index.isin(train_idx)]
        ax.scatter(train_sub["f"], train_sub["Tg(K)"], color="#2b5c8f", s=80, edgecolors="black", zorder=6, label=f"Train Exp ($N={len(train_sub)}$)")
        if len(test_sub) > 0:
            ax.scatter(test_sub["f"], test_sub["Tg(K)"], color="#e6ab02", s=90, marker="s", edgecolors="black", zorder=7, label=f"Test Exp ($N={len(test_sub)}$)")

        ax.scatter([1.0], [tg_A], color="black", marker="*", s=140, zorder=8, label=f"Homopolymer {polyA}")
        ax.scatter([0.0], [tg_B], color="black", marker="P", s=120, zorder=8, label=f"Homopolymer {polyB}")

        ax.set_title(f"Composition Trajectory: {polyA} ($f$) + {polyB} ($1-f$)", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Composition Fraction of {polyA} ($f$)", fontsize=12)
        ax.set_ylabel("Glass Transition Temperature $T_g$ (K)", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc="upper left" if p_idx == 0 else "lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Mechanistic Sweep: How Physics Priors Eliminate Extrapolation Plateaus in Tuned Decision Trees", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_f12 = FIG_DIR / "fig12_physics_trees_trajectories.png"
    plt.savefig(out_f12, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_f12}")

    for f in [out_f11, out_f12]:
        shutil.copy(f, ARTIFACT_DIR / f.name)
    shutil.copy(Path(__file__), BASE_DIR / "scripts/experiment_physics_trees.py")

    print("\n" + "=" * 145)
    print("                     PHYSICS-INFORMED TUNED DECISION TREES EXPERIMENTAL RESULTS")
    print("=" * 145)
    print(summary_df.to_string(index=False))
    print("=" * 145)

if __name__ == "__main__":
    main()
