"""
evaluate_suites.py
==================
Step 2 of the Pipeline: Standalone Evaluation Harness.
Loads the FROZEN models checkpointed by train_harness.py and probes them
across all 4 test suites:
- Suite 1: Random Holdout (IID Baseline)
- Suite 2: Core-to-Tail Extrapolation Probe
- Suite 3: One-Sided Extrapolation Probe
- Suite 4: Equimolar Gap Interpolation Probe

Computes mean ± std across the 5 independent seeds and exports:
- results/benchmark_master_matrix.csv
- results/predictions_all_suites.csv
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Configure writable matplotlib cache directory
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path("C:/Projects/copolymers/scripts")))
from models.tabular_models import load_tabular_model, predict_tabular
from models.neural_models import load_neural_model, predict_neural
from models.sequence_models import load_sequence_model, predict_sequence
from models.physics_baselines import predict_gordon_taylor, predict_fox, predict_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_suites")

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
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load split manifest
    with open(proc_dir / "benchmark_splits_manifest.json") as f:
        manifest = json.load(f)
    suites = {
        "Suite 1: Random Holdout (IID)": manifest["suite1_random_indices"],
        "Suite 2: Core-to-Tail Extrap": manifest["suite2_core_to_tail_indices"],
        "Suite 3: One-Sided Extrap": manifest["suite3_one_sided_indices"],
        "Suite 4: Equimolar Gap Interp": manifest["suite4_gap_interpolation_indices"],
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

    # Data structure: results[model_name][suite_name] = list of seed metric dicts
    suite_results = {m: {s_name: [] for s_name in suites} for m in ALL_PARADIGMS}
    pred_records = []

    log.info(f"Probing {len(ALL_PARADIGMS)} paradigms across {len(suites)} stress suites...")

    for seed in SEEDS:
        log.info(f"  Evaluating models for Seed {seed}...")

        # Load trained models for this seed
        rf = load_tabular_model(models_dir / f"rf_seed{seed}.pkl")
        xgb = load_tabular_model(models_dir / f"xgb_seed{seed}.pkl")
        m_concat = load_neural_model("Concatenation", models_dir / f"Concatenation_seed{seed}.pt", raw_dim=raw_dim)
        m_convex = load_neural_model("LatentConvex", models_dir / f"LatentConvex_seed{seed}.pt", raw_dim=raw_dim)
        m_thermo = load_neural_model("Thermodynamic", models_dir / f"Thermodynamic_seed{seed}.pt", raw_dim=raw_dim)
        m_seq = load_sequence_model(models_dir / f"BigSMILES_LM_seed{seed}.pt", lm_dim=768)

        for s_name, indices in suites.items():
            y_true = y_all[indices]
            sub_blends = blends.iloc[indices]

            # 1. Tabular predictions
            p_rf = predict_tabular(rf, X_rdkit_all[indices])
            p_xgb = predict_tabular(xgb, X_rdkit_all[indices])

            # 2. Neural predictions
            p_concat = predict_neural(m_concat, ea_all[indices], eb_all[indices], f_all[indices], y_mean, y_std)
            p_convex = predict_neural(m_convex, ea_all[indices], eb_all[indices], f_all[indices], y_mean, y_std)
            p_thermo = predict_neural(m_thermo, ea_all[indices], eb_all[indices], f_all[indices], y_mean, y_std)

            # 3. Sequence prediction
            p_seq = predict_sequence(m_seq, zlm_all[indices], y_mean, y_std)

            # 4. Physics predictions
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

            for m_name, preds in preds_map.items():
                m_dict = compute_metrics(y_true, preds)
                suite_results[m_name][s_name].append(m_dict)

                if seed == 42:
                    for i, idx in enumerate(indices):
                        pred_records.append({
                            "seed": seed, "suite": s_name, "model": m_name,
                            "actual": float(y_true[i]), "predicted": float(preds[i]),
                            "f": float(sub_blends.iloc[i]["f"]), "pair_id": sub_blends.iloc[i]["pair_id"]
                        })

    # Save detailed prediction points
    pd.DataFrame(pred_records).to_csv(results_dir / "predictions_all_suites.csv", index=False)

    # 7. Compile Master Comparison Matrix
    summary_rows = []
    s1_key = "Suite 1: Random Holdout (IID)"
    s2_key = "Suite 2: Core-to-Tail Extrap"
    s3_key = "Suite 3: One-Sided Extrap"
    s4_key = "Suite 4: Equimolar Gap Interp"

    for m in ALL_PARADIGMS:
        row = {"Model Paradigm": m}

        # Suite 1
        s1_rmses = [x["rmse"] for x in suite_results[m][s1_key]]
        s1_maes = [x["mae"] for x in suite_results[m][s1_key]]
        s1_r2s = [x["r2"] for x in suite_results[m][s1_key]]
        s1_mean_rmse = np.mean(s1_rmses)
        row["Suite 1 RMSE (K)"] = f"{s1_mean_rmse:.2f} ± {np.std(s1_rmses):.2f}"
        row["Suite 1 MAE (K)"] = f"{np.mean(s1_maes):.2f}"
        row["Suite 1 R2"] = f"{np.mean(s1_r2s):.3f}"

        # Suite 2
        s2_rmses = [x["rmse"] for x in suite_results[m][s2_key]]
        s2_mean_rmse = np.mean(s2_rmses)
        row["Suite 2 RMSE (K)"] = f"{s2_mean_rmse:.2f} ± {np.std(s2_rmses):.2f}"
        delta_s2 = ((s2_mean_rmse - s1_mean_rmse) / max(1e-4, s1_mean_rmse)) * 100.0
        row["Delta_OOD Suite 2 (%)"] = f"{delta_s2:+.1f}%"

        # Suite 3
        s3_rmses = [x["rmse"] for x in suite_results[m][s3_key]]
        s3_mean_rmse = np.mean(s3_rmses)
        row["Suite 3 RMSE (K)"] = f"{s3_mean_rmse:.2f} ± {np.std(s3_rmses):.2f}"
        delta_s3 = ((s3_mean_rmse - s1_mean_rmse) / max(1e-4, s1_mean_rmse)) * 100.0
        row["Delta_OOD Suite 3 (%)"] = f"{delta_s3:+.1f}%"

        # Suite 4
        s4_rmses = [x["rmse"] for x in suite_results[m][s4_key]]
        row["Suite 4 RMSE (K)"] = f"{np.mean(s4_rmses):.2f} ± {np.std(s4_rmses):.2f}"

        summary_rows.append(row)

    matrix_df = pd.DataFrame(summary_rows)
    matrix_path = results_dir / "benchmark_master_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False)

    log.info("\n" + "=" * 95)
    log.info("                    MASTER COMPARISON BENCHMARK MATRIX")
    log.info("=" * 95)
    log.info(f"\n{matrix_df.to_string(index=False)}")
    log.info("=" * 95)

if __name__ == "__main__":
    main()
