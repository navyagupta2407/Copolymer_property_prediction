"""
generate_polybert_embeddings.py
===============================
Generates 600-dimensional canonical monomer embeddings using the official
PolyBERT macromolecular foundation model (Kuenneth & Ramprasad, Nature Communications 2023).

Converts raw monomer SMILES containing [Th]/[Lr] chain tags into standard PSMILES (with '*' connections)
and performs attention-mask-weighted mean pooling over the DeBERTa-v2 hidden states.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("polybert")

def main():
    base_dir = Path("C:/Projects/copolymers")
    raw_path = base_dir / "data/raw/experimental_dataset.csv"
    proc_dir = base_dir / "data/processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy()

    # Identify all 14 unique monomers and extract canonical names
    p1 = blends[["Poly1", "SMILES_0"]].rename(columns={"Poly1": "name", "SMILES_0": "smiles"})
    p2 = blends[["Poly2", "SMILES_1"]].rename(columns={"Poly2": "name", "SMILES_1": "smiles"})
    mono_df = pd.concat([p1, p2]).drop_duplicates(subset=["smiles"]).sort_values("smiles").reset_index(drop=True)

    # Convert to PSMILES
    mono_df["psmiles"] = mono_df["smiles"].str.replace("[Th]", "*", regex=False).str.replace("[Lr]", "*", regex=False)
    log.info(f"Identified {len(mono_df)} unique monomers across 85 copolymer pairs.")
    for _, r in mono_df.iterrows():
        log.info(f"  {r['name']:8s} | Raw: {r['smiles']:32s} -> PSMILES: {r['psmiles']}")

    # Load PolyBERT model and tokenizer
    model_id = "xushijie/polyBERT"
    log.info(f"Loading PolyBERT from cached weights ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True)
    model.eval()

    # Tokenize and compute 600-d embeddings
    psmiles_list = mono_df["psmiles"].tolist()
    inputs = tokenizer(psmiles_list, padding=True, truncation=True, return_tensors="pt")

    log.info("Computing PolyBERT embeddings via masked mean pooling...")
    with torch.no_grad():
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    embs_np = embs.cpu().numpy()
    log.info(f"Generated embedding matrix of shape: {embs_np.shape} (dim = {model.config.hidden_size})")

    # Save to data/processed/monomer_embeddings_polybert.npz with full metadata
    out_path = proc_dir / "monomer_embeddings_polybert.npz"
    np.savez_compressed(
        out_path,
        embeddings=embs_np,
        dim=model.config.hidden_size,
        smiles=mono_df["smiles"].values,
        psmiles=mono_df["psmiles"].values,
        monomer_names=mono_df["name"].values,
        model_id=model_id,
        pooling="masked_mean_pooling",
    )
    log.info(f"Successfully saved PolyBERT embeddings to {out_path} ({out_path.stat().st_size / 1024:.1f} KB).")

    # Quick pairwise cosine similarity check to inspect chemical differentiation
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
    cos_sim = (embs_np @ embs_np.T) / (norms @ norms.T)
    min_sim = cos_sim[np.triu_indices_from(cos_sim, k=1)].min()
    mean_sim = cos_sim[np.triu_indices_from(cos_sim, k=1)].mean()
    max_sim = cos_sim[np.triu_indices_from(cos_sim, k=1)].max()
    log.info(f"PolyBERT chemical differentiation metrics across 14 monomers:")
    log.info(f"  Min Cosine Sim: {min_sim:.4f} | Mean: {mean_sim:.4f} | Max: {max_sim:.4f}")

if __name__ == "__main__":
    main()
