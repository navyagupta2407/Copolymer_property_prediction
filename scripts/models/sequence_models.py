"""
models/sequence_models.py
=========================
Standalone training and inference functions for Polymer Language Models:
- S6: BigSMILES Sequence Token Transformer Regressor

Operates on 768-dimensional ChemBERTa BigSMILES sequence embeddings.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class S6_BigSMILES_LM(nn.Module):
    def __init__(self, lm_dim: int = 768, d: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(lm_dim, 256),
            nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, z_lm):
        return self.proj(z_lm)

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

class FullSequenceModel(nn.Module):
    def __init__(self, rep_module: nn.Module, head_module: nn.Module):
        super().__init__()
        self.rep = rep_module
        self.head = head_module

    def forward(self, z_lm):
        z = self.rep(z_lm)
        return self.head(z)

def build_sequence_model(lm_dim: int = 768, d: int = 128) -> FullSequenceModel:
    rep = S6_BigSMILES_LM(lm_dim=lm_dim, d=d)
    head = FixedDownstreamRegressor(d=d)
    return FullSequenceModel(rep, head)

def train_sequence_model(
    model: FullSequenceModel,
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
        for b_zlm, b_y in train_loader:
            b_zlm, b_y = b_zlm.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_zlm)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(b_y)
            n_samples += len(b_y)
        final_loss = running_loss / max(1, n_samples)
        scheduler.step(final_loss)
    return final_loss

def save_sequence_model(model: FullSequenceModel, filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)

def load_sequence_model(filepath: Path, lm_dim: int = 768, d: int = 128, device: str = "cpu") -> FullSequenceModel:
    model = build_sequence_model(lm_dim=lm_dim, d=d)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_sequence(model: FullSequenceModel, z_lm: torch.Tensor, y_mean: float, y_std: float, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds_norm = model(z_lm.to(device))
        preds = preds_norm * y_std + y_mean
    return preds.cpu().numpy()
