"""
run_option_a_counterfactual.py
==============================
Executes Option A (Matched-Pair Random Holdout Control) across all 9 paradigms,
using the TUNED Tabular Models (Random Forest & XGBoost) as the canonical
tabular baselines from the hyperparameter tuning stage onward.

Outputs:
1. data/processed/counterfactual_random_splits_manifest.json
2. results/option_a_counterfactual_matrix.csv
3. results/figures/fig6_option_a_grouped_drop.png
"""

import json
import logging
import os
import random
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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import load_tabular_model, predict_tabular, save_tabular_model
from models.neural_models import build_neural_model, train_neural_model, load_neural_model, predict_neural, save_neural_model
from models.sequence_models import build_sequence_model, train_sequence_model, load_sequence_model, predict_sequence, save_sequence_model
from models.physics_baselines import fit_all_pairs_gordon_taylor, predict_gordon_taylor, predict_fox, predict_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("option_a")

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    tuned_dir = base_dir / "results/models_tuned"
    ctrl_models_dir = base_dir / "results/models_counterfactual_control"
    ctrl_models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("C:/Users/Suhani  Sharma/.gemini/antigravity/brain/716101ab-b1de-4a5f-b2b4-2f12d72f1910")

    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    with open(tuned_dir / "best_hyperparameters.json") as f:
        best_hp = json.load(f)

    df_raw = pd.read_csv(raw_path)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    blends["blend_idx"] = blends.index
    blends["pair_id"] = blends["Poly1"].astype(str) + "-" + blends["Poly2"].astype(str)
    blends["f"] = blends["comp_0"].astype(float) / 100.0
    blends["Tg_actual"] = blends["Tg(K)"].astype(float)

    orig_train = set(manifest["train_indices"])
    s1_orig = manifest["suite1_random_indices"]
    s2_orig = manifest["suite2_core_to_tail_indices"]
    s3_orig = manifest["suite3_one_sided_indices"]
    s4_orig = manifest["suite4_gap_interpolation_indices"]
    groups = manifest["group_allocations"]

    # 1. Construct Matched-Pair Random Control Split (exact same train/test counts per pair)
    set_seed(42)
    ctrl_train = []
    ctrl_s1 = list(s1_orig)
    ctrl_s2, ctrl_s3, ctrl_s4 = [], [], []

    for pid in groups["general_pairs"]:
        p_idxs = blends[blends["pair_id"] == pid]["blend_idx"].tolist()
        for idx in p_idxs:
            if idx in orig_train:
                ctrl_train.append(idx)

    for pid in groups["core_pairs"]:
        p_idxs = blends[blends["pair_id"] == pid]["blend_idx"].tolist()
        n_test = sum(1 for i in p_idxs if i in set(s2_orig))
        random.shuffle(p_idxs)
        ctrl_s2.extend(p_idxs[:n_test])
        ctrl_train.extend(p_idxs[n_test:])

    for pid in groups["one_sided_pairs"]:
        p_idxs = blends[blends["pair_id"] == pid]["blend_idx"].tolist()
        n_test = sum(1 for i in p_idxs if i in set(s3_orig))
        random.shuffle(p_idxs)
        ctrl_s3.extend(p_idxs[:n_test])
        ctrl_train.extend(p_idxs[n_test:])

    for pid in groups["tails_pairs"]:
        p_idxs = blends[blends["pair_id"] == pid]["blend_idx"].tolist()
        n_test = sum(1 for i in p_idxs if i in set(s4_orig))
        random.shuffle(p_idxs)
        ctrl_s4.extend(p_idxs[:n_test])
        ctrl_train.extend(p_idxs[n_test:])

    ctrl_manifest = {
        "train_indices": ctrl_train,
        "control1_random_indices": ctrl_s1,
        "control2_matched_core_pairs_indices": ctrl_s2,
        "control3_matched_one_sided_pairs_indices": ctrl_s3,
        "control4_matched_tails_pairs_indices": ctrl_s4,
        "group_allocations": groups,
        "homopolymer_tg_lookup": manifest["homopolymer_tg_lookup"],
    }
    with open(proc_dir / "counterfactual_random_splits_manifest.json", "w") as f:
        json.dump(ctrl_manifest, f, indent=2)
    log.info(f"Saved counterfactual split manifest: Train={len(ctrl_train)}, C1={len(ctrl_s1)}, C2={len(ctrl_s2)}, C3={len(ctrl_s3)}, C4={len(ctrl_s4)}")

    # 2. Load features
    rdkit_data = np.load(proc_dir / "blend_rdkit_features.npz")
    X_rdkit_all = rdkit_data["features"]
    y_all = rdkit_data["targets"]

    pb_data = np.load(proc_dir / "monomer_embeddings_polybert.npz", allow_pickle=True)
    pb_embs = pb_data["embeddings"]
    pb_smiles = pb_data["smiles"].tolist()
    smiles_to_pb_idx = {s: i for i, s in enumerate(pb_smiles)}
    raw_dim = int(pb_data["dim"])

    ea_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_0"]]), dtype=torch.float32)
    eb_all = torch.tensor(np.array([pb_embs[smiles_to_pb_idx[s]] for s in blends["SMILES_1"]]), dtype=torch.float32)
    f_all = torch.tensor(blends["f"].values, dtype=torch.float32).unsqueeze(-1)

    bs_cache = np.load(proc_dir / "bigsmiles_cache.npz")
    zlm_all = torch.tensor(bs_cache["blend_embs"], dtype=torch.float32)

    # Stress D_train stats
    with open(models_dir / "target_normalization.json") as f:
        norm_stress = json.load(f)
    y_mean_stress = norm_stress["y_mean"]
    y_std_stress = norm_stress["y_std"]

    with open(models_dir / "gordon_taylor_fits.json") as f:
        gt_fits_stress = json.load(f)
    tg_lookup = manifest["homopolymer_tg_lookup"]

    # Control D_train_rand stats
    X_ctrl_tr = X_rdkit_all[ctrl_train]
    y_ctrl_tr = y_all[ctrl_train]
    y_mean_ctrl = float(y_ctrl_tr.mean())
    y_std_ctrl = float(y_ctrl_tr.std() + 1e-7)
    y_ctrl_tr_norm = torch.tensor((y_ctrl_tr - y_mean_ctrl) / y_std_ctrl, dtype=torch.float32)

    with open(ctrl_models_dir / "target_normalization.json", "w") as f:
        json.dump({"y_mean": y_mean_ctrl, "y_std": y_std_ctrl}, f, indent=2)

    gt_fits_ctrl = fit_all_pairs_gordon_taylor(blends.iloc[ctrl_train], tg_lookup)
    with open(ctrl_models_dir / "gordon_taylor_fits.json", "w") as f:
        json.dump(gt_fits_ctrl, f, indent=2)

    neural_ctrl_ds = TensorDataset(ea_all[ctrl_train], eb_all[ctrl_train], f_all[ctrl_train], y_ctrl_tr_norm)
    seq_ctrl_ds = TensorDataset(zlm_all[ctrl_train], y_ctrl_tr_norm)

    SEEDS = [42, 43, 44, 45, 46]
    ALL_PARADIGMS = [
        "Random Forest (Tuned)",
        "XGBoost (Tuned)",
        "Concatenation",
        "LatentConvex",
        "Thermodynamic",
        "BigSMILES_LM",
        "Gordon-Taylor",
        "Fox",
        "Linear",
    ]

    stress_suites = {"S1": s1_orig, "S2": s2_orig, "S3": s3_orig, "S4": s4_orig}
    ctrl_suites = {"C1": ctrl_s1, "C2": ctrl_s2, "C3": ctrl_s3, "C4": ctrl_s4}

    res_stress = {m: {k: [] for k in stress_suites} for m in ALL_PARADIGMS}
    res_ctrl = {m: {k: [] for k in ctrl_suites} for m in ALL_PARADIGMS}

    xgb_params = best_hp["XGBoost"]
    rf_params = best_hp["RandomForest"]

    for seed in SEEDS:
        log.info(f"Evaluating Stress & Training/Evaluating Matched Control for Seed {seed}...")

        # --- A. Load Stress Models (Tuned RF & XGB, Frozen Neural/Sequence) ---
        rf_s = load_tabular_model(tuned_dir / f"rf_tuned_seed{seed}.pkl")
        xgb_s = load_tabular_model(tuned_dir / f"xgb_tuned_seed{seed}.pkl")
        m_concat_s = load_neural_model("Concatenation", models_dir / f"Concatenation_seed{seed}.pt", raw_dim=raw_dim)
        m_convex_s = load_neural_model("LatentConvex", models_dir / f"LatentConvex_seed{seed}.pt", raw_dim=raw_dim)
        m_thermo_s = load_neural_model("Thermodynamic", models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)
        m_seq_s = load_sequence_model(models_dir / f"BigSMILES_LM_seed{seed}.pt", lm_dim=768)

        for k, idxs in stress_suites.items():
            y_true = y_all[idxs]
            sub_blends = blends.iloc[idxs]

            p_gt, p_fox, p_lin = [], [], []
            for _, r in sub_blends.iterrows():
                pid = r["pair_id"]
                p_A, p_B = pid.split("-")
                tg_A = tg_lookup.get(p_A, r["Tg_actual"])
                tg_B = tg_lookup.get(p_B, r["Tg_actual"])
                f_val = r["f"]
                k_val = gt_fits_stress.get(pid, 1.0)
                p_gt.append(predict_gordon_taylor(f_val, tg_A, tg_B, k_val))
                p_fox.append(predict_fox(f_val, tg_A, tg_B))
                p_lin.append(predict_linear(f_val, tg_A, tg_B))

            preds_s = {
                "Random Forest (Tuned)": predict_tabular(rf_s, X_rdkit_all[idxs]),
                "XGBoost (Tuned)": predict_tabular(xgb_s, X_rdkit_all[idxs]),
                "Concatenation": predict_neural(m_concat_s, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean_stress, y_std_stress),
                "LatentConvex": predict_neural(m_convex_s, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean_stress, y_std_stress),
                "Thermodynamic": predict_neural(m_thermo_s, ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean_stress, y_std_stress),
                "BigSMILES_LM": predict_sequence(m_seq_s, zlm_all[idxs], y_mean_stress, y_std_stress),
                "Gordon-Taylor": np.array(p_gt),
                "Fox": np.array(p_fox),
                "Linear": np.array(p_lin),
            }
            for m_name, p_arr in preds_s.items():
                res_stress[m_name][k].append(compute_metrics(y_true, p_arr))

        # --- B. Train & Evaluate Matched-Pair Control Models ---
        set_seed(seed)
        rf_c = RandomForestRegressor(
            n_estimators=120,
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            min_samples_split=rf_params["min_samples_split"],
            max_features=rf_params["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
        rf_c.fit(X_ctrl_tr, y_ctrl_tr)
        save_tabular_model(rf_c, ctrl_models_dir / f"rf_tuned_ctrl_seed{seed}.pkl")

        xgb_c = XGBRegressor(
            n_estimators=100,
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            min_child_weight=xgb_params["min_child_weight"],
            reg_lambda=xgb_params["reg_lambda"],
            subsample=xgb_params["subsample"],
            random_state=seed,
            n_jobs=-1,
        )
        xgb_c.fit(X_ctrl_tr, y_ctrl_tr)
        save_tabular_model(xgb_c, ctrl_models_dir / f"xgb_tuned_ctrl_seed{seed}.pkl")

        train_loader = DataLoader(neural_ctrl_ds, batch_size=32, shuffle=True)
        n_models_c = {}
        for m_name in ["Concatenation", "LatentConvex", "Thermodynamic"]:
            set_seed(seed)
            m_c = build_neural_model(m_name, raw_dim=raw_dim, d=128)
            train_neural_model(m_c, train_loader, epochs=60, lr=1e-3, device="cpu")
            save_neural_model(m_c, ctrl_models_dir / f"{m_name}_ctrl_seed{seed}.pt")
            n_models_c[m_name] = m_c

        set_seed(seed)
        seq_loader = DataLoader(seq_ctrl_ds, batch_size=32, shuffle=True)
        seq_c = build_sequence_model(lm_dim=768, d=128)
        train_sequence_model(seq_c, seq_loader, epochs=60, lr=1e-3, device="cpu")
        save_sequence_model(seq_c, ctrl_models_dir / f"BigSMILES_LM_ctrl_seed{seed}.pt")

        for k, idxs in ctrl_suites.items():
            y_true = y_all[idxs]
            sub_blends = blends.iloc[idxs]

            p_gt, p_fox, p_lin = [], [], []
            for _, r in sub_blends.iterrows():
                pid = r["pair_id"]
                p_A, p_B = pid.split("-")
                tg_A = tg_lookup.get(p_A, r["Tg_actual"])
                tg_B = tg_lookup.get(p_B, r["Tg_actual"])
                f_val = r["f"]
                k_val = gt_fits_ctrl.get(pid, 1.0)
                p_gt.append(predict_gordon_taylor(f_val, tg_A, tg_B, k_val))
                p_fox.append(predict_fox(f_val, tg_A, tg_B))
                p_lin.append(predict_linear(f_val, tg_A, tg_B))

            preds_c = {
                "Random Forest (Tuned)": predict_tabular(rf_c, X_rdkit_all[idxs]),
                "XGBoost (Tuned)": predict_tabular(xgb_c, X_rdkit_all[idxs]),
                "Concatenation": predict_neural(n_models_c["Concatenation"], ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean_ctrl, y_std_ctrl),
                "LatentConvex": predict_neural(n_models_c["LatentConvex"], ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean_ctrl, y_std_ctrl),
                "Thermodynamic": predict_neural(n_models_c["Thermodynamic"], ea_all[idxs], eb_all[idxs], f_all[idxs], y_mean_ctrl, y_std_ctrl),
                "BigSMILES_LM": predict_sequence(seq_c, zlm_all[idxs], y_mean_ctrl, y_std_ctrl),
                "Gordon-Taylor": np.array(p_gt),
                "Fox": np.array(p_fox),
                "Linear": np.array(p_lin),
            }
            for m_name, p_arr in preds_c.items():
                res_ctrl[m_name][k].append(compute_metrics(y_true, p_arr))

    # 3. Compile Master Option A Table
    rows = []
    plot_records = []
    for m in ALL_PARADIGMS:
        s1_m = np.mean([x["rmse"] for x in res_stress[m]["S1"]])
        s1_s = np.std([x["rmse"] for x in res_stress[m]["S1"]])
        s2_m = np.mean([x["rmse"] for x in res_stress[m]["S2"]])
        s2_s = np.std([x["rmse"] for x in res_stress[m]["S2"]])
        s3_m = np.mean([x["rmse"] for x in res_stress[m]["S3"]])
        s3_s = np.std([x["rmse"] for x in res_stress[m]["S3"]])
        s4_m = np.mean([x["rmse"] for x in res_stress[m]["S4"]])
        s4_s = np.std([x["rmse"] for x in res_stress[m]["S4"]])

        c1_m = np.mean([x["rmse"] for x in res_ctrl[m]["C1"]])
        c2_m = np.mean([x["rmse"] for x in res_ctrl[m]["C2"]])
        c3_m = np.mean([x["rmse"] for x in res_ctrl[m]["C3"]])
        c4_m = np.mean([x["rmse"] for x in res_ctrl[m]["C4"]])

        # Option A Matched-Pair % Drop: (Stress_k - Control_k) / Control_k
        drop_s2_optA = ((s2_m - c2_m) / c2_m) * 100.0
        drop_s3_optA = ((s3_m - c3_m) / c3_m) * 100.0
        drop_s4_optA = ((s4_m - c4_m) / c4_m) * 100.0

        # Raw Cross-Suite % Drop vs S1 (for reference comparison)
        drop_s2_vs_s1 = ((s2_m - s1_m) / s1_m) * 100.0
        drop_s3_vs_s1 = ((s3_m - s1_m) / s1_m) * 100.0
        drop_s4_vs_s1 = ((s4_m - s1_m) / s1_m) * 100.0

        rows.append({
            "Paradigm": m,
            "Suite 1 (IID) RMSE (K)": f"{s1_m:.2f} ± {s1_s:.2f}",
            "Suite 2 Stress (K)": f"{s2_m:.2f} ± {s2_s:.2f}",
            "Control 2 Rand (K)": f"{c2_m:.2f}",
            "Matched Drop S2 (%)": f"{drop_s2_optA:+.1f}%",
            "Suite 3 Stress (K)": f"{s3_m:.2f} ± {s3_s:.2f}",
            "Control 3 Rand (K)": f"{c3_m:.2f}",
            "Matched Drop S3 (%)": f"{drop_s3_optA:+.1f}%",
            "Suite 4 Stress (K)": f"{s4_m:.2f} ± {s4_s:.2f}",
            "Control 4 Rand (K)": f"{c4_m:.2f}",
            "Matched Drop S4 (%)": f"{drop_s4_optA:+.1f}%",
            "Raw Drop S2 vs S1 (%)": f"{drop_s2_vs_s1:+.1f}%",
            "Raw Drop S3 vs S1 (%)": f"{drop_s3_vs_s1:+.1f}%",
            "Raw Drop S4 vs S1 (%)": f"{drop_s4_vs_s1:+.1f}%",
        })

        plot_records.append({
            "model": m,
            "s2_optA": drop_s2_optA,
            "s3_optA": drop_s3_optA,
            "s4_optA": drop_s4_optA,
            "s1_rmse": s1_m,
            "s2_rmse": s2_m,
            "s3_rmse": s3_m,
            "s4_rmse": s4_m,
        })

    out_df = pd.DataFrame(rows)
    out_csv = results_dir / "option_a_counterfactual_matrix.csv"
    out_df.to_csv(out_csv, index=False)
    log.info(f"Saved Option A counterfactual matrix to {out_csv}")

    # 4. Generate Grouped Bar Chart for Option A Matched-Pair % Drop
    fig, ax = plt.subplots(figsize=(14, 6.8))
    x = np.arange(len(ALL_PARADIGMS))
    w = 0.25

    vals_s2 = [r["s2_optA"] for r in plot_records]
    vals_s3 = [r["s3_optA"] for r in plot_records]
    vals_s4 = [r["s4_optA"] for r in plot_records]

    b2 = ax.bar(x - w, vals_s2, w, label="Suite 2 vs. Control 2 (Core-to-Tail Extrap, Same 20 Pairs)", color="#e6ab02", edgecolor="black", linewidth=0.8)
    b3 = ax.bar(x, vals_s3, w, label="Suite 3 vs. Control 3 (One-Sided Extrap f > 0.50, Same 20 Pairs)", color="#d95f02", edgecolor="black", linewidth=0.8)
    b4 = ax.bar(x + w, vals_s4, w, label="Suite 4 vs. Control 4 (Equimolar Gap Interp, Same 20 Pairs)", color="#7570b3", edgecolor="black", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
    ax.set_ylabel("Matched-Pair Relative Error Change (%)\n[(Stress RMSE - Matched Random RMSE) / Matched Random RMSE]", fontsize=11, fontweight="bold")
    ax.set_title("De-Confounded Regime Vulnerability (Option A: Stress Suite vs. Matched-Pair Random Holdout)\nUsing Tuned Tabular Baselines & Identical Monomer Pairs per Suite", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_PARADIGMS, rotation=25, ha="right", fontsize=10.5, fontweight="bold")
    ax.set_ylim(-38, 58)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    for bars in [b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            y_pos = h + 1.0 if h >= 0 else h - 1.2
            ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{h:+.1f}%", ha="center", va=va, fontsize=8.0, fontweight="bold")

    plt.tight_layout()
    fig_path = fig_dir / "fig6_option_a_grouped_drop.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    shutil.copy(fig_path, artifact_dir / "fig6_option_a_grouped_drop.png")
    shutil.copy(Path(__file__), base_dir / "scripts/run_option_a_counterfactual.py")
    log.info(f"Saved Option A figure to {fig_path} and copied to artifacts.")

    print("\n" + "=" * 145)
    print("         FINALIZED OPTION A MATRIX (TUNED TABULAR MODELS + MATCHED-PAIR RANDOM CONTROLS)")
    print("=" * 145)
    print(out_df.to_string(index=False))
    print("=" * 145)

if __name__ == "__main__":
    main()
