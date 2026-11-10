"""
models/physics_baselines.py
===========================
Classical Thermodynamic and Mixing Rule Models:
- Gordon-Taylor equation (1-parameter nonlinear fit per pair)
- Fox equation (0-parameter harmonic reciprocal rule)
- Linear rule of mixtures (0-parameter additive rule)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def predict_linear(f: float, tg_A: float, tg_B: float) -> float:
    return float(f * tg_A + (1.0 - f) * tg_B)

def predict_fox(f: float, tg_A: float, tg_B: float) -> float:
    return float(1.0 / (f / tg_A + (1.0 - f) / tg_B))

def predict_gordon_taylor(f: float, tg_A: float, tg_B: float, k: float) -> float:
    denom = f + k * (1.0 - f)
    if abs(denom) < 1e-7:
        return float(tg_B)
    return float((f * tg_A + k * (1.0 - f) * tg_B) / denom)

def fit_gordon_taylor_k(f_arr: np.ndarray, y_arr: np.ndarray, tg_A: float, tg_B: float) -> float:
    if len(f_arr) < 2:
        return 1.0

    def mse_obj(k_val):
        preds = np.array([predict_gordon_taylor(f, tg_A, tg_B, k_val) for f in f_arr])
        return float(np.mean((preds - y_arr) ** 2))

    res = minimize_scalar(mse_obj, bounds=(0.05, 10.0), method="bounded")
    return float(res.x if res.success else 1.0)

def fit_all_pairs_gordon_taylor(train_df: pd.DataFrame, tg_lookup: dict) -> dict:
    fits = {}
    for pid in train_df["pair_id"].unique():
        sub = train_df[train_df["pair_id"] == pid]
        p_A, p_B = pid.split("-")
        tg_A = tg_lookup.get(p_A, sub["Tg_actual"].max())
        tg_B = tg_lookup.get(p_B, sub["Tg_actual"].min())

        f_vals = sub["f"].values
        y_vals = sub["Tg_actual"].values
        k_opt = fit_gordon_taylor_k(f_vals, y_vals, tg_A, tg_B)
        fits[pid] = k_opt
    return fits
