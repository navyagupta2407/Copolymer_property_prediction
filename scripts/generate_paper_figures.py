"""
generate_paper_figures.py
=========================
Generates all 5 publication-ready figures for the research paper:
"The Illusion of In-Distribution Accuracy: Probing Industry-Standard Copolymer
Machine Learning Models Under Fixed-Training Stress Suites"

Figures Generated:
1. fig1_split_distributions.png: Histogram & KDE distributions of composition fraction f across splits
2. fig2_benchmark_master_comparison.png: Multi-suite performance & OOD degradation comparison
3. fig3_generalization_gap_overfitting.png: Train vs. Test Generalization Gap quadrant plot
4. fig4_parity_multi_paradigm.png: 4-panel publication parity plot (XGBoost, Concat, Gordon-Taylor, Thermodynamic)
5. fig5_mechanistic_extrapolation_curves.png: Continuous f in [0, 1] mechanistic trajectory showing why trees fail
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Matplotlib cache setup for headless / sandbox environment
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import load_tabular_model, predict_tabular
from models.neural_models import load_neural_model, predict_neural
from models.sequence_models import load_sequence_model, predict_sequence
from models.physics_baselines import predict_gordon_taylor, predict_fox, predict_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("figures")

# Publication Styling Configuration
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


def generate_figure_1_distributions(df_blends, manifest):
    """Figure 1: Composition fraction distributions across D_train and the 4 Stress Suites."""
    log.info("Generating Figure 1: Split Distributions...")
    train_f = df_blends.iloc[manifest["train_indices"]]["f"].values
    s1_f = df_blends.iloc[manifest["suite1_random_indices"]]["f"].values
    s2_f = df_blends.iloc[manifest["suite2_core_to_tail_indices"]]["f"].values
    s3_f = df_blends.iloc[manifest["suite3_one_sided_indices"]]["f"].values
    s4_f = df_blends.iloc[manifest["suite4_gap_interpolation_indices"]]["f"].values

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)

    splits_data = [
        (r"$D_{\mathrm{train}}$ (Core Blends)", train_f, "#2b5c8f", f"N = {len(train_f)}"),
        (r"Suite 1: Random Holdout", s1_f, "#1b9e77", f"N = {len(s1_f)} (IID)"),
        (r"Suite 2: Core-to-Tail", s2_f, "#e6ab02", rf"N = {len(s2_f)} ($f<0.2 \cup f>0.8$)"),
        (r"Suite 3: One-Sided Extrap", s3_f, "#d95f02", rf"N = {len(s3_f)} ($f \notin [f_{{min}}, f_{{max}}]$)"),
        (r"Suite 4: Equimolar Gap", s4_f, "#7570b3", rf"N = {len(s4_f)} ($f \in [0.4, 0.6]$)"),
    ]

    bins = np.linspace(0.0, 1.0, 15)

    for i, (title, f_arr, color, label_sub) in enumerate(splits_data):
        ax = axes[i]
        ax.hist(f_arr, bins=bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.8, density=True)
        ax.set_title(f"{title}\n({label_sub})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Composition Fraction $f$", fontsize=11)
        ax.set_xlim(0, 1)
        if i == 0:
            ax.set_ylabel("Probability Density", fontsize=12)

    plt.suptitle("Benchmark Partition Architecture: Probing Monomer Composition Regimes", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_path = FIG_DIR / "fig1_split_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_path}")


def generate_figure_2_master_comparison(matrix_df):
    """Figure 2: Grouped bar chart comparing RMSE across suites and showing OOD degradation."""
    log.info("Generating Figure 2: Master Benchmark Comparison...")
    
    # Extract clean floats
    paradigms = matrix_df["Model Paradigm"].tolist()
    
    s1_vals = [float(x.split("±")[0].strip()) for x in matrix_df["Suite 1 RMSE (K)"]]
    s2_vals = [float(x.split("±")[0].strip()) for x in matrix_df["Suite 2 RMSE (K)"]]
    s3_vals = [float(x.split("±")[0].strip()) for x in matrix_df["Suite 3 RMSE (K)"]]
    s4_vals = [float(x.split("±")[0].strip()) for x in matrix_df["Suite 4 RMSE (K)"]]

    x = np.arange(len(paradigms))
    width = 0.20

    fig, ax = plt.subplots(figsize=(15, 6))

    rects1 = ax.bar(x - 1.5 * width, s1_vals, width, label="Suite 1: Random Holdout (IID)", color="#1b9e77", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x - 0.5 * width, s2_vals, width, label="Suite 2: Core-to-Tail Extrap", color="#e6ab02", edgecolor="black", alpha=0.9)
    rects3 = ax.bar(x + 0.5 * width, s3_vals, width, label="Suite 3: One-Sided Extrap", color="#d95f02", edgecolor="black", alpha=0.9)
    rects4 = ax.bar(x + 1.5 * width, s4_vals, width, label="Suite 4: Equimolar Gap Interp", color="#7570b3", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Root Mean Squared Error, RMSE (K)", fontsize=13, fontweight="bold")
    ax.set_title("Master Performance Profile Across In-Distribution & Stress Evaluation Suites", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, shadow=True, fontsize=10)
    ax.set_ylim(0, 38)

    # Highlight tree degradation on Suite 3
    for idx in [0, 1]: # RF and XGB
        h = s3_vals[idx]
        ax.annotate(
            f"+{((h - s1_vals[idx])/s1_vals[idx]*100):.0f}%",
            xy=(x[idx] + 0.5 * width, h + 0.7),
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#d95f02"
        )

    # Highlight Thermodynamic Hybrid stability
    idx_th = paradigms.index("Thermodynamic")
    ax.annotate(
        "Rank #1 (3.55 K)\nRobust",
        xy=(x[idx_th] + 0.5 * width, s3_vals[idx_th] + 1.5),
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1b9e77",
        arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.5)
    )

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_benchmark_master_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_path}")


def generate_figure_3_generalization_quadrants(diag_df):
    """Figure 3: Train RMSE vs. Suite 1 Test RMSE quadrant plot highlighting overfitting and underfitting."""
    log.info("Generating Figure 3: Generalization Quadrants...")

    fig, ax = plt.subplots(figsize=(10, 8))

    train_rmses = [float(x.split("±")[0].strip()) for x in diag_df["Train RMSE (K)"]]
    test_rmses = [float(x.split("±")[0].strip()) for x in diag_df["Suite 1 (IID) RMSE (K)"]]
    paradigms = diag_df["Paradigm"].tolist()

    # Define color scheme based on diagnosis
    colors = {
        "Random Forest": "#e7298a",
        "XGBoost": "#d95f02",
        "Concatenation": "#7570b3",
        "BigSMILES_LM": "#e6ab02",
        "Thermodynamic": "#1b9e77",
        "LatentConvex": "#2b5c8f",
        "Gordon-Taylor": "#66a61e",
        "Fox": "#a6761d",
        "Linear": "#666666",
    }

    # Reference boundaries
    ax.axhline(10.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(10.0, color="gray", linestyle="--", alpha=0.5)

    # Shaded regions
    ax.fill_between([0, 10], 10, 30, color="#fbb4ae", alpha=0.25, label="Severe Overfitting (Low Train, High Test)")
    ax.fill_between([10, 20], 10, 30, color="#fed9a6", alpha=0.25, label="Underfitting / High Bias (High Train, High Test)")
    ax.fill_between([0, 10], 0, 10, color="#ccebc5", alpha=0.25, label="Optimal Generalization (Low Train, Low Test)")

    # 45-degree parity (Zero Generalization Gap)
    ax.plot([0, 25], [0, 25], "k:", alpha=0.6, label=r"Zero Generalization Gap ($\Delta = 0$)")

    for i, p in enumerate(paradigms):
        tr = train_rmses[i]
        te = test_rmses[i]
        c = colors.get(p, "#333333")
        ax.scatter(tr, te, color=c, s=140, edgecolors="black", linewidth=1.2, zorder=5)

        # Offsets for clean text placement
        dx, dy = 0.4, 0.4
        if p == "XGBoost":
            dx, dy = 0.5, -0.8
        elif p == "Random Forest":
            dx, dy = 0.5, 0.4
        elif p == "LatentConvex":
            dx, dy = 0.5, -0.6
        elif p == "Thermodynamic":
            dx, dy = 0.5, 0.4

        ax.annotate(
            p,
            (tr, te),
            xytext=(tr + dx, te + dy),
            fontsize=10,
            fontweight="bold",
            color=c,
            zorder=6
        )

    ax.set_xlabel("Training Set RMSE (K)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Suite 1 (IID Holdout) Test RMSE (K)", fontsize=13, fontweight="bold")
    ax.set_title("Overfitting vs. Underfitting Diagnostic Space\nMapping Model Inductive Biases", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 28)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_generalization_gap_overfitting.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_path}")


def generate_figure_4_parity_multi_paradigm(raw_df, manifest, models_dir):
    """Figure 4: 4-Panel Parity Plots for XGBoost, Concat MLP, Gordon-Taylor, Thermodynamic Hybrid."""
    log.info("Generating Figure 4: Multi-Paradigm Parity Plots...")

    # Load test indices and models for Seed 42
    seed = 42
    train_idx = manifest["train_indices"]
    s1_idx = manifest["suite1_random_indices"]
    s3_idx = manifest["suite3_one_sided_indices"]

    df_blends = raw_df[raw_df["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    df_blends["pair_id"] = df_blends["Poly1"].astype(str) + "-" + df_blends["Poly2"].astype(str)
    df_blends["f"] = df_blends["comp_0"].astype(float) / 100.0

    rdkit_data = np.load(BASE_DIR / "data/processed/blend_rdkit_features.npz")
    X_rdkit_all = rdkit_data["features"]
    y_all = rdkit_data["targets"]

    pb_data = np.load(BASE_DIR / "data/processed/monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])

    ea_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in df_blends["SMILES_0"]]), dtype=torch.float32)
    eb_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in df_blends["SMILES_1"]]), dtype=torch.float32)
    f_all = torch.tensor(df_blends["f"].values, dtype=torch.float32).unsqueeze(-1)

    with open(models_dir / "target_normalization.json") as f:
        norm_params = json.load(f)
    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    with open(models_dir / "gordon_taylor_fits.json") as f:
        gt_fits = json.load(f)
    tg_lookup = manifest["homopolymer_tg_lookup"]

    # Load the 4 models
    xgb = load_tabular_model(models_dir / f"xgb_seed{seed}.pkl")
    m_concat = load_neural_model("Concatenation", models_dir / f"Concatenation_seed{seed}.pt", raw_dim=raw_dim)
    m_thermo = load_neural_model("Thermodynamic", models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)

    # Function to get predictions
    def get_preds(idxs):
        p_xgb = predict_tabular(xgb, X_rdkit_all[idxs])
        p_cat = predict_neural(m_concat, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)
        p_th = predict_neural(m_thermo, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean, y_std)

        p_gt = []
        for _, r in df_blends.iloc[idxs].iterrows():
            pid = r["pair_id"]
            p_A, p_B = pid.split("-")
            tg_A = tg_lookup.get(p_A, r["Tg(K)"])
            tg_B = tg_lookup.get(p_B, r["Tg(K)"])
            k_val = gt_fits.get(pid, 1.0)
            p_gt.append(predict_gordon_taylor(r["f"], tg_A, tg_B, k_val))

        return {
            "XGBoost": p_xgb,
            "Concatenation": p_cat,
            "Gordon-Taylor": np.array(p_gt),
            "Thermodynamic": p_th,
        }

    preds_tr = get_preds(train_idx)
    preds_s1 = get_preds(s1_idx)
    preds_s3 = get_preds(s3_idx)

    models_to_plot = [
        ("XGBoost", "Industry Tabular Baseline (Memorization Failure)", preds_tr["XGBoost"], preds_s1["XGBoost"], preds_s3["XGBoost"]),
        ("Concatenation", "Naive Latent MLP (Underfitting Failure)", preds_tr["Concatenation"], preds_s1["Concatenation"], preds_s3["Concatenation"]),
        ("Gordon-Taylor", "Empirical 1-Param Physics Benchmark", preds_tr["Gordon-Taylor"], preds_s1["Gordon-Taylor"], preds_s3["Gordon-Taylor"]),
        ("Thermodynamic", "Physics-Informed Hybrid (Ours: Optimal Generalization)", preds_tr["Thermodynamic"], preds_s1["Thermodynamic"], preds_s3["Thermodynamic"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    axes = axes.flatten()

    y_tr_true = y_all[train_idx]
    y_s1_true = y_all[s1_idx]
    y_s3_true = y_all[s3_idx]

    min_t, max_t = 180, 520

    for i, (m_name, subtitle, p_tr, p_s1, p_s3) in enumerate(models_to_plot):
        ax = axes[i]
        ax.plot([min_t, max_t], [min_t, max_t], "k--", lw=1.5, alpha=0.7, label=r"Ideal Parity ($y = \hat{y}$)")
        ax.fill_between([min_t, max_t], [min_t - 10, max_t - 10], [min_t + 10, max_t + 10], color="gray", alpha=0.15, label="±10 K Error Margin")

        # Scatter points
        ax.scatter(y_tr_true, p_tr, color="#2b5c8f", alpha=0.35, s=35, label=rf"Train ($D_{{\mathrm{{train}}}}$, N={len(train_idx)})")
        ax.scatter(y_s1_true, p_s1, color="#1b9e77", alpha=0.9, s=55, marker="o", edgecolors="black", linewidth=0.8, label=f"Suite 1 (IID Holdout, N={len(s1_idx)})")
        ax.scatter(y_s3_true, p_s3, color="#d95f02", alpha=0.9, s=65, marker="^", edgecolors="black", linewidth=0.8, label=f"Suite 3 (OOD Extrapolation, N={len(s3_idx)})")

        rmse_tr = np.sqrt(np.mean((p_tr - y_tr_true)**2))
        rmse_s1 = np.sqrt(np.mean((p_s1 - y_s1_true)**2))
        rmse_s3 = np.sqrt(np.mean((p_s3 - y_s3_true)**2))

        ax.set_title(f"{m_name}\n({subtitle})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Experimental $T_g$ (K)", fontsize=11)
        ax.set_ylabel("Predicted $T_g$ (K)", fontsize=11)
        ax.set_xlim(min_t, max_t)
        ax.set_ylim(min_t, max_t)

        # Performance text badge
        badge = f"Train RMSE: {rmse_tr:.1f} K\nSuite 1 RMSE: {rmse_s1:.1f} K\nSuite 3 RMSE: {rmse_s3:.1f} K"
        ax.text(0.04, 0.72, badge, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#666666", alpha=0.9))

        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8.5, framealpha=0.9)

    plt.suptitle("Parity Diagnostic: Probing Model Predictions Under Out-Of-Distribution Stress", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / "fig4_parity_multi_paradigm.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_path}")


def generate_figure_5_mechanistic_curves(raw_df, manifest, models_dir):
    """Figure 5: Continuous composition sweep f in [0, 1] demonstrating tree step function failure vs smooth physics."""
    log.info("Generating Figure 5: Mechanistic Composition Sweep...")

    # We select two representative monomer pairs:
    # Pair 1: MA - MMA (Methyl Acrylate - Methyl Methacrylate)
    # Pair 2: BA - MMA (Butyl Acrylate - Methyl Methacrylate)
    selected_pairs = ["MA-MMA", "BA-MMA"]

    df_blends = raw_df[raw_df["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    df_blends["pair_id"] = df_blends["Poly1"].astype(str) + "-" + df_blends["Poly2"].astype(str)
    df_blends["f"] = df_blends["comp_0"].astype(float) / 100.0

    rdkit_data = np.load(BASE_DIR / "data/processed/blend_rdkit_features.npz")
    pb_data = np.load(BASE_DIR / "data/processed/monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])

    with open(models_dir / "target_normalization.json") as f:
        norm_params = json.load(f)
    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    with open(models_dir / "gordon_taylor_fits.json") as f:
        gt_fits = json.load(f)
    tg_lookup = manifest["homopolymer_tg_lookup"]

    # Load models
    seed = 42
    xgb = load_tabular_model(models_dir / f"xgb_seed{seed}.pkl")
    rf = load_tabular_model(models_dir / f"rf_seed{seed}.pkl")
    m_thermo = load_neural_model("Thermodynamic", models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)

    # Monomer descriptors for RDKit table synthesis
    df_desc = pd.read_csv(BASE_DIR / "data/processed/monomer_rdkit_descriptors.csv")
    desc_dict = {r["smiles"]: r.drop(["monomer", "smiles"]).values.astype(np.float32) for _, r in df_desc.iterrows()}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    f_dense = np.linspace(0.01, 0.99, 150)

    for p_idx, pair_id in enumerate(selected_pairs):
        ax = axes[p_idx]
        polyA, polyB = pair_id.split("-")
        sub = df_blends[df_blends["pair_id"] == pair_id]

        smiles_A = sub.iloc[0]["SMILES_0"]
        smiles_B = sub.iloc[0]["SMILES_1"]
        tg_A = tg_lookup.get(polyA, 378.0)
        tg_B = tg_lookup.get(polyB, 295.0)

        # 1. Synthesize RDKit features for f_dense
        # Formulation: [ ideal_mixture (20) || difference/contrast (20) || f (1) ]
        desc_A = desc_dict[smiles_A]
        desc_B = desc_dict[smiles_B]
        X_dense_rdkit = []
        for f_val in f_dense:
            d_ideal = f_val * desc_A + (1.0 - f_val) * desc_B
            d_diff = np.abs(desc_A - desc_B)
            feat = np.concatenate([d_ideal, d_diff, [f_val]])
            X_dense_rdkit.append(feat)
        X_dense_rdkit = np.array(X_dense_rdkit, dtype=np.float32)

        # Predict XGBoost and Random Forest
        y_xgb_dense = predict_tabular(xgb, X_dense_rdkit)
        y_rf_dense = predict_tabular(rf, X_dense_rdkit)

        # 2. Predict Thermodynamic Hybrid
        ea = torch.tensor(pb_embs[smiles_to_pb_idx[smiles_A]], dtype=torch.float32).unsqueeze(0).repeat(len(f_dense), 1)
        eb = torch.tensor(pb_embs[smiles_to_pb_idx[smiles_B]], dtype=torch.float32).unsqueeze(0).repeat(len(f_dense), 1)
        f_t = torch.tensor(f_dense, dtype=torch.float32).unsqueeze(-1)
        y_th_dense = predict_neural(m_thermo, ea, eb, f_t, y_mean, y_std)

        # 3. Predict Gordon-Taylor
        k_val = gt_fits.get(pair_id, 1.0)
        y_gt_dense = np.array([predict_gordon_taylor(f, tg_A, tg_B, k_val) for f in f_dense])

        # Plot curves
        ax.plot(f_dense, y_xgb_dense, color="#d95f02", lw=2.5, linestyle="--", label="XGBoost (Step-wise Flattening)")
        ax.plot(f_dense, y_rf_dense, color="#e7298a", lw=2.0, linestyle=":", label="Random Forest (Piecewise Plateaus)")
        ax.plot(f_dense, y_gt_dense, color="#666666", lw=2.0, linestyle="-.", label=f"Gordon-Taylor ($k={k_val:.2f}$)")
        ax.plot(f_dense, y_th_dense, color="#1b9e77", lw=3.0, linestyle="-", label="Thermodynamic Hybrid (Ours)")

        # Plot experimental ground truth points
        train_sub = sub[sub.index.isin(manifest["train_indices"])]
        test_sub = sub[~sub.index.isin(manifest["train_indices"])]

        ax.scatter(train_sub["f"], train_sub["Tg(K)"], color="#2b5c8f", s=80, edgecolors="black", zorder=6, label=f"Train Exp ($N={len(train_sub)}$)")
        if len(test_sub) > 0:
            ax.scatter(test_sub["f"], test_sub["Tg(K)"], color="#e6ab02", s=90, marker="s", edgecolors="black", zorder=7, label=f"Test Exp ($N={len(test_sub)}$)")

        # Endpoint homopolymer anchors
        ax.scatter([1.0], [tg_A], color="black", marker="*", s=150, zorder=8, label=f"Homopolymer {polyA} ({tg_A:.0f} K)")
        ax.scatter([0.0], [tg_B], color="black", marker="P", s=130, zorder=8, label=f"Homopolymer {polyB} ({tg_B:.0f} K)")

        ax.set_title(f"Composition Sweep: {polyA} (f) + {polyB} (1-f)", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Composition Fraction of {polyA} ($f$)", fontsize=12)
        ax.set_ylabel("Glass Transition Temperature $T_g$ (K)", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc="upper left" if p_idx == 0 else "lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Mechanistic Demonstration: Continuous Trajectory vs. Decision Tree Plateaus", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIG_DIR / "fig5_mechanistic_extrapolation_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out_path}")


def copy_figures_to_artifacts():
    """Copy all generated figures to artifact folder for user viewing."""
    log.info("Copying generated figures to artifact directory...")
    for fig_file in FIG_DIR.glob("*.png"):
        shutil.copy(fig_file, ARTIFACT_DIR / fig_file.name)
        log.info(f"  Copied {fig_file.name} to artifacts.")


def main():
    log.info("Starting Paper Figures Generation Pipeline...")

    # Load shared manifests and data
    proc_dir = BASE_DIR / "data/processed"
    raw_path = BASE_DIR / "data/raw/experimental_dataset.csv"
    results_dir = BASE_DIR / "results"
    models_dir = results_dir / "models"
    diag_dir = results_dir / "diagnostics"

    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)

    df_raw = pd.read_csv(raw_path)
    df_blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    df_blends["f"] = df_blends["comp_0"].astype(float) / 100.0

    matrix_df = pd.read_csv(results_dir / "benchmark_master_matrix.csv")
    diag_df = pd.read_csv(diag_dir / "overfitting_underfitting_summary.csv")

    # Generate Figures
    generate_figure_1_distributions(df_blends, manifest)
    generate_figure_2_master_comparison(matrix_df)
    generate_figure_3_generalization_quadrants(diag_df)
    generate_figure_4_parity_multi_paradigm(df_raw, manifest, models_dir)
    generate_figure_5_mechanistic_curves(df_raw, manifest, models_dir)

    # Copy to artifacts
    copy_figures_to_artifacts()
    log.info("All 5 publication figures successfully generated and cataloged!")


if __name__ == "__main__":
    main()
