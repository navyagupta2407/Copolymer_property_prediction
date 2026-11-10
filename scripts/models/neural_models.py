"""
models/neural_models.py
=======================
Standalone training and inference functions for Deep Learning Latent Architectures:
- S1: Thermodynamic Free-Volume Hybrid (Gordon-Taylor core + Margules envelope)
- S2: Latent Convex Combination (Linear manifold geodesic)
- S3: Unconstrained Feature Concatenation MLP (Naive baseline)

Operates on PolyBERT 600-dimensional monomer embeddings.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class S1_ThermodynamicHybrid(nn.Module):
    def __init__(self, raw_dim: int = 600, d: int = 128):
        super().__init__()
        self.proj = nn.Linear(raw_dim, d)
        self.k_estimator = nn.Sequential(
            nn.Linear(2 * raw_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.excess_net = nn.Sequential(
            nn.Linear(raw_dim, 64),
            nn.Tanh(),
            nn.Linear(64, d, bias=False),
        )

    def forward(self, e_A, e_B, f, z_lm=None):
        z_A = self.proj(e_A)
        z_B = self.proj(e_B)
        k_hat = torch.nn.functional.softplus(self.k_estimator(torch.cat([e_A, e_B], dim=-1))) + 0.1
        denom = f + k_hat * (1.0 - f)
        z_gt = (f * z_A + k_hat * (1.0 - f) * z_B) / denom
        excess = f * (1.0 - f) * self.excess_net(torch.abs(e_A - e_B))
        return z_gt + excess

class S2_LatentConvex(nn.Module):
    def __init__(self, raw_dim: int = 600, d: int = 128):
        super().__init__()
        self.proj = nn.Linear(raw_dim, d)

    def forward(self, e_A, e_B, f, z_lm=None):
        z_A = self.proj(e_A)
        z_B = self.proj(e_B)
        return f * z_A + (1.0 - f) * z_B

class S3_UnconstrainedConcat(nn.Module):
    def __init__(self, raw_dim: int = 600, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * raw_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, e_A, e_B, f, z_lm=None):
        x = torch.cat([e_A, e_B, f], dim=-1)
        return self.net(x)

class FixedDownstreamRegressor(nn.Module):
    def __init__(self, d: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d)
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        h = self.drop(self.act(self.ln1(self.fc1(z))))
        z_res = self.act(self.ln2(self.fc2(h)) + z)
        return self.head(z_res).squeeze(-1)

class FullCopolymerModel(nn.Module):
    def __init__(self, rep_module: nn.Module, head_module: nn.Module):
        super().__init__()
        self.rep = rep_module
        self.head = head_module

    def forward(self, e_A, e_B, f, z_lm=None):
        z = self.rep(e_A, e_B, f, z_lm=z_lm)
        return self.head(z)

def build_neural_model(model_type: str, raw_dim: int = 600, d: int = 128) -> FullCopolymerModel:
    if model_type == "Thermodynamic":
        rep = S1_ThermodynamicHybrid(raw_dim=raw_dim, d=d)
    elif model_type == "LatentConvex":
        rep = S2_LatentConvex(raw_dim=raw_dim, d=d)
    elif model_type == "Concatenation":
        rep = S3_UnconstrainedConcat(raw_dim=raw_dim, d=d)
    else:
        raise ValueError(f"Unknown neural model type: {model_type}")
    
    head = FixedDownstreamRegressor(d=d)
    return FullCopolymerModel(rep, head)

def train_neural_model(
    model: FullCopolymerModel,
    train_loader: DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> float:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        n_samples = 0
        for b_ea, b_eb, b_f, b_y in train_loader:
            b_ea, b_eb, b_f, b_y = b_ea.to(device), b_eb.to(device), b_f.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_ea, b_eb, b_f)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(b_y)
            n_samples += len(b_y)
        final_loss = running_loss / max(1, n_samples)
        scheduler.step(final_loss)
    return final_loss

def save_neural_model(model: FullCopolymerModel, filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)

def load_neural_model(model_type: str, filepath: Path, raw_dim: int = 600, d: int = 128, device: str = "cpu") -> FullCopolymerModel:
    model = build_neural_model(model_type, raw_dim=raw_dim, d=d)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_neural(model: FullCopolymerModel, e_A: torch.Tensor, e_B: torch.Tensor, f: torch.Tensor, y_mean: float, y_std: float, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds_norm = model(e_A.to(device), e_B.to(device), f.to(device))
        preds = preds_norm * y_std + y_mean
    return preds.cpu().numpy()
