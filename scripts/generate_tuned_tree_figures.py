"""
generate_tuned_tree_figures.py
==============================
Generates comprehensive publication-grade comparison figures specifically
evaluating the Tuned Tree Models (Tuned XGBoost, Tuned Random Forest)
against their Out-of-the-Box Baselines and the Thermodynamic Hybrid.

Figures Generated:
1. fig7_tuned_vs_baseline_master_comparison.png: Multi-suite performance (Train, S1, S2, S3, S4)
2. fig8_tuned_trees_parity_comparison.png: 4-panel parity plot (XGB Base vs Tuned, RF Base vs Tuned)
3. fig9_tuned_trees_mechanistic_trajectories.png: Continuous f in [0, 1] sweeps showing persistent plateaus
4. fig10_tuned_trees_grouped_drop.png: Grouped % drop chart comparing degradation before & after tuning
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Matplotlib configuration for headless environment
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
from models.physics_baselines import predict_gordon_taylor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tuned_tree_figs")

# Styling configuration
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

    # 1. Load splits
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    train_idx = manifest["train_indices"]
    s1_idx = manifest["suite1_random_indices"]
    s2_idx = manifest["suite2_core_to_tail_indices"]
    s3_idx = manifest["suite3_one_sided_indices"]
    s4_idx = manifest["suite4_gap_interpolation_indices"]

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

    # 3. Load Tabular RDKit features
    rdkit_data = np.load(proc_dir / "blend_rdkit_features.npz")
    X_rdkit_all = rdkit_data["features"]
    y_all = rdkit_data["targets"]

    # 4. Load PolyBERT embeddings
    pb_data = np.load(proc_dir / "monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])

    ea_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_0"]]), dtype=torch.float32)
    eb_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_1"]]), dtype=torch.float32)
    f_all = torch.tensor(blends["f"].values, dtype=torch.float32).unsqueeze(-1)

    with open(base_models_dir / "target_normalization.json") as f:
        norm_params = json.load(f)
    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    SEEDS = [42, 43, 44, 45, 46]
    MODELS_COMPARED = [
        "Random Forest (Base)",
        "Random Forest (Tuned)",
        "XGBoost (Base)",
        "XGBoost (Tuned)",
        "Thermodynamic Hybrid",
    ]

    # Metrics storage: {model: {suite: [metrics across seeds]}}
    all_metrics = {m: {s: [] for s in suites} for m in MODELS_COMPARED}
    seed42_preds = {m: {} for m in MODELS_COMPARED}

    log.info("Evaluating Baseline vs. Tuned Tree Models across 5 seeds and all suites...")
    for seed in SEEDS:
        rf_base = load_tabular_model(base_models_dir / f"rf_seed{seed}.pkl")
        rf_tuned = load_tabular_model(tuned_models_dir / f"rf_tuned_seed{seed}.pkl")
        xgb_base = load_tabular_model(base_models_dir / f"xgb_seed{seed}.pkl")
        xgb_tuned = load_tabular_model(tuned_models_dir / f"xgb_tuned_seed{seed}.pkl")
        m_thermo = load_neural_model("Thermodynamic", base_models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)

        for s_name, idxs in suites.items():
            y_sub = y_all[idxs]

            p_rf_base = predict_tabular(rf_base, X_rdkit_all[idxs])
            p_rf_tuned = predict_tabular(rf_tuned, X_rdkit_all[idxs])
            p_xgb_base = predict_tabular(xgb_base, X_rdkit_all[idxs])
            p_xgb_tuned = predict_tabular(xgb_tuned, X_rdkit_all[idxs])
            p_thermo = predict_neural(m_thermo, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)

            preds_map = {
                "Random Forest (Base)": p_rf_base,
                "Random Forest (Tuned)": p_rf_tuned,
                "XGBoost (Base)": p_xgb_base,
                "XGBoost (Tuned)": p_xgb_tuned,
                "Thermodynamic Hybrid": p_thermo,
            }

            for m in MODELS_COMPARED:
                m_dict = compute_metrics(y_sub, preds_map[m])
                all_metrics[m][s_name].append(m_dict)

            if seed == 42:
                for m in MODELS_COMPARED:
                    seed42_preds[m][s_name] = preds_map[m]

    # =========================================================================
    # FIGURE 7: MULTI-SUITE COMPARISON BAR CHART (Baseline vs. Tuned Trees)
    # =========================================================================
    log.info("Generating Figure 7: Multi-Suite Comparison...")
    fig, ax = plt.subplots(figsize=(15, 6.5))

    suite_names = list(suites.keys())
    x = np.arange(len(suite_names))
    width = 0.16

    colors = {
        "Random Forest (Base)": "#e7298a",
        "Random Forest (Tuned)": "#c51b7d",
        "XGBoost (Base)": "#d95f02",
        "XGBoost (Tuned)": "#b33b00",
        "Thermodynamic Hybrid": "#1b9e77",
    }

    for i, m in enumerate(MODELS_COMPARED):
        mean_rmses = [np.mean([entry["rmse"] for entry in all_metrics[m][s]]) for s in suite_names]
        std_rmses = [np.std([entry["rmse"] for entry in all_metrics[m][s]]) for s in suite_names]

        offset = (i - 2) * width
        rects = ax.bar(x + offset, mean_rmses, width, yerr=std_rmses, capsize=3,
                       label=m, color=colors[m], edgecolor="black", linewidth=0.8, alpha=0.9)

        # Value labels on top of bars
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.6, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Root Mean Squared Error, RMSE (Kelvin)", fontsize=12, fontweight="bold")
    ax.set_title("Impact of Hyperparameter Tuning on Tree Models Across All Benchmark Regimes\n(Comparing Baseline vs. Tuned Trees vs. Thermodynamic Hybrid)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(suite_names, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_ylim(0, 32)

    plt.tight_layout()
    out_f7 = FIG_DIR / "fig7_tuned_vs_baseline_master_comparison.png"
    plt.savefig(out_f7, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_f7}")

    # =========================================================================
    # FIGURE 8: 4-PANEL PARITY COMPARISON (Baseline Trees vs. Tuned Trees)
    # =========================================================================
    log.info("Generating Figure 8: 4-Panel Parity Comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    axes = axes.flatten()

    y_tr_true = y_all[train_idx]
    y_s1_true = y_all[s1_idx]
    y_s3_true = y_all[s3_idx]
    min_t, max_t = 180, 520

    panels = [
        ("XGBoost (Base)", "Baseline: max_depth=5, min_child=1 (Severe Memorization)", seed42_preds["XGBoost (Base)"]),
        ("XGBoost (Tuned)", "Tuned: max_depth=4, min_child=3, lambda=3.0 (Regularized)", seed42_preds["XGBoost (Tuned)"]),
        ("Random Forest (Base)", "Baseline: max_depth=12, min_leaf=1 (High Variance)", seed42_preds["Random Forest (Base)"]),
        ("Random Forest (Tuned)", "Tuned: max_depth=10, min_leaf=2, max_feat=0.8 (Pruned)", seed42_preds["Random Forest (Tuned)"]),
    ]

    for i, (title, sub, preds) in enumerate(panels):
        ax = axes[i]
        ax.plot([min_t, max_t], [min_t, max_t], "k--", lw=1.5, alpha=0.7, label=r"Ideal Parity ($y = \hat{y}$)")
        ax.fill_between([min_t, max_t], [min_t - 10, max_t - 10], [min_t + 10, max_t + 10], color="gray", alpha=0.15, label="±10 K Margin")

        p_tr = preds["Train"]
        p_s1 = preds["Suite 1 (IID)"]
        p_s3 = preds["Suite 3 (One-Sided)"]

        ax.scatter(y_tr_true, p_tr, color="#2b5c8f", alpha=0.35, s=30, label=rf"Train ($D_{{\mathrm{{train}}}}$, N={len(train_idx)})")
        ax.scatter(y_s1_true, p_s1, color="#1b9e77", alpha=0.9, s=55, marker="o", edgecolors="black", linewidth=0.8, label="Suite 1 (IID Holdout)")
        ax.scatter(y_s3_true, p_s3, color="#d95f02", alpha=0.9, s=65, marker="^", edgecolors="black", linewidth=0.8, label="Suite 3 (OOD Extrapolation)")

        rmse_tr = np.sqrt(np.mean((p_tr - y_tr_true)**2))
        rmse_s1 = np.sqrt(np.mean((p_s1 - y_s1_true)**2))
        rmse_s3 = np.sqrt(np.mean((p_s3 - y_s3_true)**2))

        ax.set_title(f"{title}\n({sub})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Experimental $T_g$ (K)", fontsize=11)
        ax.set_ylabel("Predicted $T_g$ (K)", fontsize=11)
        ax.set_xlim(min_t, max_t)
        ax.set_ylim(min_t, max_t)

        badge = f"Train RMSE: {rmse_tr:.1f} K\nSuite 1 RMSE: {rmse_s1:.1f} K\nSuite 3 RMSE: {rmse_s3:.1f} K"
        ax.text(0.04, 0.73, badge, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#666666", alpha=0.9))

        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8.5, framealpha=0.9)

    plt.suptitle("Parity Diagnostic: Baseline vs. Tuned Tree Models Under Composition Shift", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_f8 = FIG_DIR / "fig8_tuned_trees_parity_comparison.png"
    plt.savefig(out_f8, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_f8}")

    # =========================================================================
    # FIGURE 9: CONTINUOUS TRAJECTORIES SWEEP (f in [0, 1])
    # =========================================================================
    log.info("Generating Figure 9: Continuous Trajectories...")
    df_desc = pd.read_csv(proc_dir / "monomer_rdkit_descriptors.csv")
    desc_dict = {r["smiles"]: r.drop(["monomer", "smiles"]).values.astype(np.float32) for _, r in df_desc.iterrows()}
    tg_lookup = manifest["homopolymer_tg_lookup"]

    seed = 42
    xgb_base = load_tabular_model(base_models_dir / f"xgb_seed{seed}.pkl")
    xgb_tuned = load_tabular_model(tuned_models_dir / f"xgb_tuned_seed{seed}.pkl")
    rf_tuned = load_tabular_model(tuned_models_dir / f"rf_tuned_seed{seed}.pkl")
    m_thermo = load_neural_model("Thermodynamic", base_models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)

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

        desc_A = desc_dict[smiles_A]
        desc_B = desc_dict[smiles_B]
        X_dense_rdkit = []
        for f_val in f_dense:
            d_ideal = f_val * desc_A + (1.0 - f_val) * desc_B
            d_diff = np.abs(desc_A - desc_B)
            feat = np.concatenate([d_ideal, d_diff, [f_val]])
            X_dense_rdkit.append(feat)
        X_dense_rdkit = np.array(X_dense_rdkit, dtype=np.float32)

        # Predictions
        y_xgb_base = predict_tabular(xgb_base, X_dense_rdkit)
        y_xgb_tuned = predict_tabular(xgb_tuned, X_dense_rdkit)
        y_rf_tuned = predict_tabular(rf_tuned, X_dense_rdkit)

        ea = torch.tensor(pb_embs[smiles_to_pb_idx[smiles_A]], dtype=torch.float32).unsqueeze(0).repeat(len(f_dense), 1)
        eb = torch.tensor(pb_embs[smiles_to_pb_idx[smiles_B]], dtype=torch.float32).unsqueeze(0).repeat(len(f_dense), 1)
        f_t = torch.tensor(f_dense, dtype=torch.float32).unsqueeze(-1)
        y_th_dense = predict_neural(m_thermo, ea, eb, f_t, y_mean, y_std)

        # Plot curves
        ax.plot(f_dense, y_xgb_base, color="#d95f02", lw=2.0, linestyle=":", label="XGBoost Base (Sharp Steps)")
        ax.plot(f_dense, y_xgb_tuned, color="#b33b00", lw=2.5, linestyle="--", label="XGBoost Tuned (Still Plateaus)")
        ax.plot(f_dense, y_rf_tuned, color="#c51b7d", lw=2.0, linestyle="-.", label="Random Forest Tuned")
        ax.plot(f_dense, y_th_dense, color="#1b9e77", lw=3.0, linestyle="-", label="Thermodynamic Hybrid (Smooth)")

        # Ground truth points
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

    plt.suptitle("Mechanistic Sweep: Probing If Hyperparameter Tuning Can Prevent Extrapolation Plateaus", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_f9 = FIG_DIR / "fig9_tuned_trees_mechanistic_trajectories.png"
    plt.savefig(out_f9, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_f9}")

    # =========================================================================
    # FIGURE 10: GROUPED PERCENTAGE DROP (Baseline vs. Tuned Trees)
    # =========================================================================
    log.info("Generating Figure 10: Grouped % Drop for Tuned Trees...")
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.axhspan(0, 125, color="#fbb4ae", alpha=0.15, zorder=0)
    ax.axhspan(-35, 0, color="#ccebc5", alpha=0.18, zorder=0)
    ax.axhline(0, color="black", linewidth=1.4, linestyle="-", zorder=3)

    models_drop_comp = [
        "Random Forest (Base)",
        "Random Forest (Tuned)",
        "XGBoost (Base)",
        "XGBoost (Tuned)",
        "Thermodynamic Hybrid",
    ]

    s2_drops, s3_drops, s4_drops = [], [], []
    for m in models_drop_comp:
        s1_mean = np.mean([entry["rmse"] for entry in all_metrics[m]["Suite 1 (IID)"]])
        s2_mean = np.mean([entry["rmse"] for entry in all_metrics[m]["Suite 2 (Core-to-Tail)"]])
        s3_mean = np.mean([entry["rmse"] for entry in all_metrics[m]["Suite 3 (One-Sided)"]])
        s4_mean = np.mean([entry["rmse"] for entry in all_metrics[m]["Suite 4 (Equimolar Gap)"]])

        s2_drops.append(((s2_mean - s1_mean) / s1_mean) * 100.0)
        s3_drops.append(((s3_mean - s1_mean) / s1_mean) * 100.0)
        s4_drops.append(((s4_mean - s1_mean) / s1_mean) * 100.0)

    x_d = np.arange(len(models_drop_comp))
    w_d = 0.25

    c_s2 = "#e6ab02"
    c_s3 = "#d95f02"
    c_s4 = "#7570b3"

    r1 = ax.bar(x_d - w_d, s2_drops, w_d, label="Suite 2: Core-to-Tail Extrap (f < 0.25 or f > 0.75)",
                color=c_s2, edgecolor="black", linewidth=0.8, zorder=4)
    r2 = ax.bar(x_d, s3_drops, w_d, label="Suite 3: One-Sided Extrap (f > 0.50 held out)",
                color=c_s3, edgecolor="black", linewidth=0.8, zorder=4)
    r3 = ax.bar(x_d + w_d, s4_drops, w_d, label="Suite 4: Equimolar Gap Interp (0.25 <= f <= 0.75 held out)",
                color=c_s4, edgecolor="black", linewidth=0.8, zorder=4)

    for rect_group in [r1, r2, r3]:
        for rect in rect_group:
            h = rect.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 2.0 if h >= 0 else -2.5
            color = "#8c0000" if h > 50 else ("#005500" if h < -15 else "black")
            ax.text(rect.get_x() + rect.get_width()/2., h + offset, f"{h:+.1f}%",
                    ha="center", va=va, fontsize=9, fontweight="bold", color=color, zorder=5)

    ax.text(0.015, 0.93, "▲ Performance Degradation / Error Increase (%)",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color="#a50f15")
    ax.text(0.015, 0.08, "▼ Performance Improvement / Error Reduction (%)",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color="#006d2c")

    ax.set_ylabel("Relative RMSE Change Compared to Suite 1 IID (%)", fontsize=12, fontweight="bold")
    ax.set_title("Does Hyperparameter Tuning Eliminate Stress Suite Degradation in Trees?\n(Direct Comparison: Baseline vs. Tuned Models)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x_d)
    ax.set_xticklabels(models_drop_comp, rotation=20, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylim(-35, 125)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    out_f10 = FIG_DIR / "fig10_tuned_trees_grouped_drop.png"
    plt.savefig(out_f10, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_f10}")

    # Copy all to artifacts
    for f in [out_f7, out_f8, out_f9, out_f10]:
        shutil.copy(f, ARTIFACT_DIR / f.name)
        log.info(f"Copied {f.name} to artifacts.")

    log.info("All 4 Tuned Tree Model figures generated successfully!")

if __name__ == "__main__":
    main()
