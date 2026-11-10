"""
generate_drop_analysis.py
=========================
Calculates the exact percentage performance drop / error increase across all
model paradigms on all stress suites relative to the Suite 1 (IID) baseline.

Generates:
1. results/percentage_drop_matrix.csv: Detailed table of % RMSE change and R2 drops.
2. results/figures/fig6_stacked_drop_barchart.png: Publication-ready 2-panel stacked bar chart.
   - Panel A: Stacked Relative Performance Degradation (% Error Increase on Suites 2, 3, 4).
   - Panel B: Stacked Absolute Error Decomposition (Base Suite 1 RMSE + Added OOD Penalties in K).
"""

import json
import logging
import os
import shutil
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
log = logging.getLogger("drop_analysis")

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
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("C:/Users/Suhani  Sharma/.gemini/antigravity/brain/716101ab-b1de-4a5f-b2b4-2f12d72f1910")

    # 1. Load split manifest
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    suites = {
        "Suite 1": manifest["suite1_random_indices"],
        "Suite 2": manifest["suite2_core_to_tail_indices"],
        "Suite 3": manifest["suite3_one_sided_indices"],
        "Suite 4": manifest["suite4_gap_interpolation_indices"],
    }

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
    raw_dim = int(pb_data["dim"])

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

    # Evaluate all models across seeds and suites
    suite_metrics = {m: {s: [] for s in suites} for m in ALL_PARADIGMS}

    log.info("Computing metrics across 5 seeds and all 4 suites...")
    for seed in SEEDS:
        rf = load_tabular_model(models_dir / f"rf_seed{seed}.pkl")
        xgb = load_tabular_model(models_dir / f"xgb_seed{seed}.pkl")
        m_concat = load_neural_model("Concatenation", models_dir / f"Concatenation_seed{seed}.pt", raw_dim=raw_dim)
        m_convex = load_neural_model("LatentConvex", models_dir / f"LatentConvex_seed{seed}.pt", raw_dim=raw_dim)
        m_thermo = load_neural_model("Thermodynamic", models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)
        m_seq = load_sequence_model(models_dir / f"BigSMILES_LM_seed{seed}.pt", lm_dim=768)

        for s_key, indices in suites.items():
            y_true = y_all[indices]
            sub_blends = blends.iloc[indices]

            p_rf = predict_tabular(rf, X_rdkit_all[indices])
            p_xgb = predict_tabular(xgb, X_rdkit_all[indices])
            p_concat = predict_neural(m_concat, ea_all[indices], eb_all[indices], f_all[indices], y_mean, y_std)
            p_convex = predict_neural(m_convex, ea_all[indices], eb_all[indices], f_all[indices], y_mean, y_std)
            p_thermo = predict_neural(m_thermo, ea_all[indices], eb_all[indices], f_all[indices], y_mean, y_std)
            p_seq = predict_sequence(m_seq, zlm_all[indices], y_mean, y_std)

            p_gt, p_fox, p_lin = [], [], []
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

            preds_map = {
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
                m_res = compute_metrics(y_true, preds_map[m])
                suite_metrics[m][s_key].append(m_res)

    # Compile percentage drop matrix
    drop_rows = []
    plot_data = []

    for m in ALL_PARADIGMS:
        s1_rmse = np.mean([x["rmse"] for x in suite_metrics[m]["Suite 1"]])
        s2_rmse = np.mean([x["rmse"] for x in suite_metrics[m]["Suite 2"]])
        s3_rmse = np.mean([x["rmse"] for x in suite_metrics[m]["Suite 3"]])
        s4_rmse = np.mean([x["rmse"] for x in suite_metrics[m]["Suite 4"]])

        s1_r2 = np.mean([x["r2"] for x in suite_metrics[m]["Suite 1"]])
        s2_r2 = np.mean([x["r2"] for x in suite_metrics[m]["Suite 2"]])
        s3_r2 = np.mean([x["r2"] for x in suite_metrics[m]["Suite 3"]])
        s4_r2 = np.mean([x["r2"] for x in suite_metrics[m]["Suite 4"]])

        # Calculate percentage error increases relative to Suite 1
        pct_drop_s2 = ((s2_rmse - s1_rmse) / s1_rmse) * 100.0
        pct_drop_s3 = ((s3_rmse - s1_rmse) / s1_rmse) * 100.0
        pct_drop_s4 = ((s4_rmse - s1_rmse) / s1_rmse) * 100.0

        # R2 changes
        delta_r2_s2 = s1_r2 - s2_r2
        delta_r2_s3 = s1_r2 - s3_r2
        delta_r2_s4 = s1_r2 - s4_r2

        # Cumulative positive degradation
        cum_pct_drop = max(0.0, pct_drop_s2) + max(0.0, pct_drop_s3) + max(0.0, pct_drop_s4)

        drop_rows.append({
            "Paradigm": m,
            "Suite 1 (IID) RMSE (K)": f"{s1_rmse:.2f}",
            "Suite 1 R2": f"{s1_r2:.3f}",
            "Suite 2 RMSE (K)": f"{s2_rmse:.2f}",
            "% Drop Suite 2": f"{pct_drop_s2:+.1f}%",
            "Suite 3 RMSE (K)": f"{s3_rmse:.2f}",
            "% Drop Suite 3": f"{pct_drop_s3:+.1f}%",
            "Suite 4 RMSE (K)": f"{s4_rmse:.2f}",
            "% Drop Suite 4": f"{pct_drop_s4:+.1f}%",
            "Cumulative Positive Degradation (%)": f"{cum_pct_drop:.1f}%",
        })

        plot_data.append({
            "model": m,
            "s1_rmse": s1_rmse,
            "s2_rmse": s2_rmse,
            "s3_rmse": s3_rmse,
            "s4_rmse": s4_rmse,
            "pct_s2": pct_drop_s2,
            "pct_s3": pct_drop_s3,
            "pct_s4": pct_drop_s4,
            "pos_pct_s2": max(0.0, pct_drop_s2),
            "pos_pct_s3": max(0.0, pct_drop_s3),
            "pos_pct_s4": max(0.0, pct_drop_s4),
            "excess_s2": max(0.0, s2_rmse - s1_rmse),
            "excess_s3": max(0.0, s3_rmse - s1_rmse),
            "excess_s4": max(0.0, s4_rmse - s1_rmse),
        })

    drop_df = pd.DataFrame(drop_rows)
    drop_csv_path = results_dir / "percentage_drop_matrix.csv"
    drop_df.to_csv(drop_csv_path, index=False)
    log.info(f"Saved drop matrix to {drop_csv_path}")

    # =========================================================================
    # GENERATE STACKED BAR CHART (2 Panels)
    # =========================================================================
    log.info("Generating Figure 6: Stacked Degradation Bar Charts...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    x = np.arange(len(ALL_PARADIGMS))
    bar_width = 0.55

    # Color palette
    c_s1 = "#1b9e77"   # Teal (Base IID)
    c_s2 = "#e6ab02"   # Mustard / Orange (Suite 2 Core-to-Tail)
    c_s3 = "#d95f02"   # Deep Orange / Red (Suite 3 One-Sided Extrap)
    c_s4 = "#7570b3"   # Purple (Suite 4 Equimolar Gap)

    # -------------------------------------------------------------------------
    # PANEL A: Stacked Relative Performance Degradation (% Error Increase)
    # -------------------------------------------------------------------------
    pos_s2 = [d["pos_pct_s2"] for d in plot_data]
    pos_s3 = [d["pos_pct_s3"] for d in plot_data]
    pos_s4 = [d["pos_pct_s4"] for d in plot_data]

    p1 = ax1.bar(x, pos_s2, bar_width, label="Suite 2: Core-to-Tail Degradation", color=c_s2, edgecolor="black", linewidth=0.8)
    p2 = ax1.bar(x, pos_s3, bar_width, bottom=pos_s2, label="Suite 3: One-Sided Extrap Degradation", color=c_s3, edgecolor="black", linewidth=0.8)
    bottom_s4 = [pos_s2[i] + pos_s3[i] for i in range(len(x))]
    p3 = ax1.bar(x, pos_s4, bar_width, bottom=bottom_s4, label="Suite 4: Equimolar Gap Degradation", color=c_s4, edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("Cumulative Relative Error Increase (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Panel A: Stacked OOD Error Degradation Relative to Suite 1\n(Vulnerability to Composition Stress)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ALL_PARADIGMS, rotation=30, ha="right", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax1.set_ylim(0, 230)

    # Annotate total stacked percentage on top of bars
    for i in range(len(x)):
        total_pct = bottom_s4[i] + pos_s4[i]
        if total_pct > 5.0:
            ax1.text(x[i], total_pct + 4.0, f"+{total_pct:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        else:
            ax1.text(x[i], 4.0, "Robust\n(0%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b9e77")

    # -------------------------------------------------------------------------
    # PANEL B: Stacked Absolute Error Decomposition (Kelvin)
    # -------------------------------------------------------------------------
    base_s1 = [d["s1_rmse"] for d in plot_data]
    exc_s2 = [d["excess_s2"] for d in plot_data]
    exc_s3 = [d["excess_s3"] for d in plot_data]
    exc_s4 = [d["excess_s4"] for d in plot_data]

    b1 = ax2.bar(x, base_s1, bar_width, label="Suite 1 (IID Baseline RMSE)", color=c_s1, edgecolor="black", linewidth=0.8)
    b2 = ax2.bar(x, exc_s2, bar_width, bottom=base_s1, label="Added Error: Suite 2 (Core-to-Tail)", color=c_s2, edgecolor="black", linewidth=0.8)
    bot_exc_s3 = [base_s1[i] + exc_s2[i] for i in range(len(x))]
    b3 = ax2.bar(x, exc_s3, bar_width, bottom=bot_exc_s3, label="Added Error: Suite 3 (One-Sided Extrap)", color=c_s3, edgecolor="black", linewidth=0.8)
    bot_exc_s4 = [bot_exc_s3[i] + exc_s3[i] for i in range(len(x))]
    b4 = ax2.bar(x, exc_s4, bar_width, bottom=bot_exc_s4, label="Added Error: Suite 4 (Equimolar Gap)", color=c_s4, edgecolor="black", linewidth=0.8)

    ax2.set_ylabel("Root Mean Squared Error, RMSE (Kelvin)", fontsize=12, fontweight="bold")
    ax2.set_title("Panel B: Absolute Error Stack (Baseline + Excess OOD Penalty)\n(Total Kelvin Error Footprint)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ALL_PARADIGMS, rotation=30, ha="right", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax2.set_ylim(0, 48)

    for i in range(len(x)):
        tot_rmse = bot_exc_s4[i] + exc_s4[i]
        ax2.text(x[i], tot_rmse + 0.8, f"{tot_rmse:.1f} K", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Multi-Suite Vulnerability Profile: Exposing Fragility Under Distribution Shift", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = fig_dir / "fig6_stacked_drop_barchart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved stacked bar chart to {plot_path}")

    # Copy to artifacts
    shutil.copy(plot_path, artifact_dir / "fig6_stacked_drop_barchart.png")
    log.info(f"Copied figure to artifacts: {artifact_dir / 'fig6_stacked_drop_barchart.png'}")

    # Print summary table
    print("\n" + "=" * 125)
    print("                     PERCENTAGE DROP & DEGRADATION MATRIX (RELATIVE TO SUITE 1 IID)")
    print("=" * 125)
    print(drop_df.to_string(index=False))
    print("=" * 125)

if __name__ == "__main__":
    main()
