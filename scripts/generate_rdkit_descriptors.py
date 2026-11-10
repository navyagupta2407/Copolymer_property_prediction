"""
generate_rdkit_descriptors.py
=============================
Generates 20 physical and thermodynamic RDKit descriptors for the 14 constituent monomers,
and compiles the blended feature representations for all 351 experimental copolymer blends:

    x_blend = [ f * d_A  ||  (1 - f) * d_B  ||  |d_A - d_B|  ||  f ]

Features capture:
- Mass and volume (MolWt, HeavyAtomMolWt, HeavyAtomCount)
- Polarizability and cohesion (MolMR, TPSA, MolLogP)
- Chain flexibility and sterics (NumRotatableBonds, FractionCSP3, RingCount)
- Hydrogen bonding capability (NumHDonors, NumHAcceptors, NOCount)
- Molecular shape and branching (BertzCT, BalabanJ, Kappa1, Kappa2)
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("descriptors")

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "FractionCSP3": Lipinski.FractionCSP3,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "NOCount": Lipinski.NOCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "RingCount": Lipinski.RingCount,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "BertzCT": Descriptors.BertzCT,
    "BalabanJ": Descriptors.BalabanJ,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
}

def clean_smiles(s: str) -> str:
    return str(s).replace("[Th]", "").replace("[Lr]", "").strip()

def compute_monomer_descriptors(smiles: str) -> dict:
    s_clean = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(s_clean)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles} (clean: {s_clean})")
    
    desc = {}
    for name, fn in DESCRIPTOR_FUNCS.items():
        try:
            desc[name] = float(fn(mol))
        except Exception:
            desc[name] = 0.0
    return desc

def main():
    base_dir = Path("C:/Projects/copolymers")
    raw_path = base_dir / "data/raw/experimental_dataset.csv"
    proc_dir = base_dir / "data/processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)

    # 1. Unique monomers table
    p1 = blends[["Poly1", "SMILES_0"]].rename(columns={"Poly1": "monomer", "SMILES_0": "smiles"})
    p2 = blends[["Poly2", "SMILES_1"]].rename(columns={"Poly2": "monomer", "SMILES_1": "smiles"})
    mono_df = pd.concat([p1, p2]).drop_duplicates(subset=["smiles"]).sort_values("smiles").reset_index(drop=True)

    log.info(f"Extracting {len(DESCRIPTOR_FUNCS)} physical descriptors for {len(mono_df)} unique monomers...")
    desc_rows = []
    for _, r in mono_df.iterrows():
        d = compute_monomer_descriptors(r["smiles"])
        d["monomer"] = r["monomer"]
        d["smiles"] = r["smiles"]
        desc_rows.append(d)

    mono_desc_df = pd.DataFrame(desc_rows)
    # Reorder columns
    cols = ["monomer", "smiles"] + list(DESCRIPTOR_FUNCS.keys())
    mono_desc_df = mono_desc_df[cols]
    mono_desc_path = proc_dir / "monomer_rdkit_descriptors.csv"
    mono_desc_df.to_csv(mono_desc_path, index=False)
    log.info(f"Saved monomer descriptors to {mono_desc_path}")

    # Build descriptor lookup dictionary
    desc_lookup = {}
    feature_names = list(DESCRIPTOR_FUNCS.keys())
    for _, r in mono_desc_df.iterrows():
        desc_lookup[r["smiles"]] = r[feature_names].values.astype(np.float32)

    # 2. Build blend descriptor representations
    log.info(f"Building blend descriptor representations for {len(blends)} copolymer blends...")
    blend_features = []
    blend_targets = []
    blend_ratios = []
    blend_pairs = []

    for _, row in blends.iterrows():
        s_A = row["SMILES_0"]
        s_B = row["SMILES_1"]
        f_A = float(row["comp_0"]) / 100.0
        f_B = 1.0 - f_A
        tg = float(row["Tg(K)"])

        d_A = desc_lookup[s_A]
        d_B = desc_lookup[s_B]

        # Physics-informed blending formulation:
        # [ ideal_mixture (20) || difference/contrast (20) || f (1) ]
        d_ideal = f_A * d_A + f_B * d_B
        d_diff = np.abs(d_A - d_B)
        
        feat = np.concatenate([d_ideal, d_diff, [f_A]])
        blend_features.append(feat)
        blend_targets.append(tg)
        blend_ratios.append(f_A)
        blend_pairs.append(f"{row['Poly1']}-{row['Poly2']}")

    blend_features_np = np.array(blend_features, dtype=np.float32)
    blend_targets_np = np.array(blend_targets, dtype=np.float32)

    log.info(f"Compiled blend feature matrix: shape {blend_features_np.shape} (41 features per blend)")

    # Feature names
    full_feature_names = [f"ideal_{name}" for name in feature_names] + \
                         [f"contrast_{name}" for name in feature_names] + \
                         ["blend_ratio_f"]

    # Save to data/processed/blend_rdkit_features.npz
    out_blend_path = proc_dir / "blend_rdkit_features.npz"
    np.savez_compressed(
        out_blend_path,
        features=blend_features_np,
        targets=blend_targets_np,
        ratios=np.array(blend_ratios, dtype=np.float32),
        pair_ids=np.array(blend_pairs),
        feature_names=full_feature_names,
    )
    log.info(f"Successfully saved blend features to {out_blend_path} ({out_blend_path.stat().st_size / 1024:.1f} KB).")

if __name__ == "__main__":
    main()
