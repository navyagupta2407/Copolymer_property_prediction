"""
models/tabular_models.py
========================
Standalone training and inference functions for Tabular Tree Models:
- Random Forest Regressor
- XGBoost Regressor

Operates on RDKit physical and thermodynamic descriptors.
"""

import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42, n_estimators: int = 100, max_depth: int = 12):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.08):
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    return xgb

def save_tabular_model(model, filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_tabular_model(filepath: Path):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def predict_tabular(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
