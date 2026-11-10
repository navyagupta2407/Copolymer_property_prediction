"""
create_unified_benchmark_splits.py
==================================
Creates the definitive 'Train Once, Probe Multiple Suites' experimental partitions:
- Unified Training Set (D_train)
- Suite 1: IID Random Test Holdout
- Suite 2: Core-to-Tail Extrapolation Probe (f < 0.25 or f > 0.75 held out)
- Suite 3: One-Sided Extrapolation Probe (f > 0.50 held out)
- Suite 4: Equimolar Gap Interpolation Probe (0.25 <= f <= 0.75 held out)
"""

import json
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("splits")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def main():
    set_seed(42)
    base_dir = Path("C:/Projects/copolymers")
    raw_path = base_dir / "data/raw/experimental_dataset.csv"
    proc_dir = base_dir / "data/processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path)
    log.info(f"Loaded raw dataset from {raw_path} with {len(df_raw)} records.")

    # 1. Homopolymer reference lookup
    p1 = df_raw[df_raw["comp_0"] == 100.0][["Poly1", "Tg(K)"]].rename(columns={"Poly1": "monomer", "Tg(K)": "Tg"})
    p2 = df_raw[df_raw["comp_0"] == 0.0][["Poly2", "Tg(K)"]].rename(columns={"Poly2": "monomer", "Tg(K)": "Tg"})
    tg_lookup = pd.concat([p1, p2]).groupby("monomer")["Tg"].mean().to_dict()

    # 2. Extract blend rows (0 < comp_0 < 100)
    blends = df_raw[df_raw["comp_0"].between(0, 100, inclusive="neither")].copy().reset_index(drop=True)
    blends["blend_idx"] = blends.index
    blends["pair_id"] = blends["Poly1"].astype(str) + "-" + blends["Poly2"].astype(str)
    blends["f"] = blends["comp_0"].astype(float) / 100.0
    blends["Tg_actual"] = blends["Tg(K)"].astype(float)
    blends["Tg_A"] = blends["Poly1"].map(tg_lookup)
    blends["Tg_B"] = blends["Poly2"].map(tg_lookup)

    unique_pairs = sorted(blends["pair_id"].unique())
    n_pairs = len(unique_pairs)
    log.info(f"Identified {len(blends)} blends across {n_pairs} unique monomer pairs.")

    # 3. Categorize pairs based on composition trajectory richness
    # To test Core-to-Tail cleanly, a pair must span into tails (<25% and >75%)
    pair_stats = blends.groupby("pair_id")["comp_0"].agg(min_c="min", max_c="max", count="count")
    
    wide_pairs = pair_stats[(pair_stats["min_c"] <= 28) & (pair_stats["max_c"] >= 75) & (pair_stats["count"] >= 4)].index.tolist()
    other_pairs = [p for p in unique_pairs if p not in wide_pairs]
    
    random.shuffle(wide_pairs)
    random.shuffle(other_pairs)

    # We partition the 85 pairs into 4 balanced groups:
    # Group 1 (Core-Only): 20 pairs -> Train sees [0.25, 0.75], Tails in Suite 2
    # Group 2 (One-Sided): 20 pairs -> Train sees f <= 0.50, Upper in Suite 3
    # Group 3 (Tails-Only): 20 pairs -> Train sees f < 0.25 | f > 0.75, Core in Suite 4
    # Group 4 (General Background): 25 pairs -> Contributes to Train + Suite 1 (Random Holdout)
    group_core_pairs = wide_pairs[:20]
    group_one_sided_pairs = wide_pairs[20:35] + other_pairs[:5]  # 20 pairs
    group_tails_pairs = other_pairs[5:25]                        # 20 pairs
    group_general_pairs = wide_pairs[35:] + other_pairs[25:]     # 25 pairs

    log.info(f"Pair allocation: Core={len(group_core_pairs)}, OneSided={len(group_one_sided_pairs)}, Tails={len(group_tails_pairs)}, General={len(group_general_pairs)}")

    train_indices = []
    suite1_random_indices = []
    suite2_core_to_tail_indices = []
    suite3_one_sided_indices = []
    suite4_gap_interpolation_indices = []

    # Process Group 1: Core-to-Tail Extrapolation
    for pid in group_core_pairs:
        p_blends = blends[blends["pair_id"] == pid]
        core = p_blends[(p_blends["f"] >= 0.25) & (p_blends["f"] <= 0.75)]
        tails = p_blends[(p_blends["f"] < 0.25) | (p_blends["f"] > 0.75)]
        train_indices.extend(core["blend_idx"].tolist())
        suite2_core_to_tail_indices.extend(tails["blend_idx"].tolist())

    # Process Group 2: One-Sided Asymmetric Extrapolation
    for pid in group_one_sided_pairs:
        p_blends = blends[blends["pair_id"] == pid]
        lower = p_blends[p_blends["f"] <= 0.50]
        upper = p_blends[p_blends["f"] > 0.50]
        train_indices.extend(lower["blend_idx"].tolist())
        suite3_one_sided_indices.extend(upper["blend_idx"].tolist())

    # Process Group 3: Equimolar Gap Interpolation
    for pid in group_tails_pairs:
        p_blends = blends[blends["pair_id"] == pid]
        tails = p_blends[(p_blends["f"] < 0.30) | (p_blends["f"] > 0.70)]
        core = p_blends[(p_blends["f"] >= 0.30) & (p_blends["f"] <= 0.70)]
        train_indices.extend(tails["blend_idx"].tolist())
        suite4_gap_interpolation_indices.extend(core["blend_idx"].tolist())

    # Process Group 4: General Background (Random Holdout)
    for pid in group_general_pairs:
        p_blends = blends[blends["pair_id"] == pid]
        # Reserve 1 point per pair for random holdout, rest in train
        indices = p_blends["blend_idx"].tolist()
        random.shuffle(indices)
        suite1_random_indices.append(indices[0])
        train_indices.extend(indices[1:])

    # Summary checks
    log.info(f"Total Unified Training Set (D_train): {len(train_indices)} samples")
    log.info(f"Suite 1 (IID Random Holdout):        {len(suite1_random_indices)} samples")
    log.info(f"Suite 2 (Core-to-Tail Extrap):       {len(suite2_core_to_tail_indices)} samples")
    log.info(f"Suite 3 (One-Sided Extrap):          {len(suite3_one_sided_indices)} samples")
    log.info(f"Suite 4 (Equimolar Gap Interp):      {len(suite4_gap_interpolation_indices)} samples")

    # Verify no overlap between training set and test suites
    assert len(set(train_indices) & set(suite1_random_indices)) == 0, "Leakage in Suite 1!"
    assert len(set(train_indices) & set(suite2_core_to_tail_indices)) == 0, "Leakage in Suite 2!"
    assert len(set(train_indices) & set(suite3_one_sided_indices)) == 0, "Leakage in Suite 3!"
    assert len(set(train_indices) & set(suite4_gap_interpolation_indices)) == 0, "Leakage in Suite 4!"

    # Export manifest
    manifest = {
        "train_indices": train_indices,
        "suite1_random_indices": suite1_random_indices,
        "suite2_core_to_tail_indices": suite2_core_to_tail_indices,
        "suite3_one_sided_indices": suite3_one_sided_indices,
        "suite4_gap_interpolation_indices": suite4_gap_interpolation_indices,
        "group_allocations": {
            "core_pairs": group_core_pairs,
            "one_sided_pairs": group_one_sided_pairs,
            "tails_pairs": group_tails_pairs,
            "general_pairs": group_general_pairs,
        },
        "homopolymer_tg_lookup": tg_lookup,
    }

    manifest_path = proc_dir / "benchmark_splits_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Saved split manifest to {manifest_path}")

    # Export split statistics table
    stats_rows = [
        {"Split / Test Suite": "Unified Training Set (D_train)", "Sample Count": len(train_indices), "Pct of Dataset": f"{100*len(train_indices)/len(blends):.1f}%", "Description": "Single training partition for all candidate models"},
        {"Split / Test Suite": "Suite 1: Random Holdout", "Sample Count": len(suite1_random_indices), "Pct of Dataset": f"{100*len(suite1_random_indices)/len(blends):.1f}%", "Description": "IID control sampled across general background pairs"},
        {"Split / Test Suite": "Suite 2: Core-to-Tail Extrap", "Sample Count": len(suite2_core_to_tail_indices), "Pct of Dataset": f"{100*len(suite2_core_to_tail_indices)/len(blends):.1f}%", "Description": "Tails (f < 0.25 or f > 0.75) of pairs trained on core only"},
        {"Split / Test Suite": "Suite 3: One-Sided Extrap", "Sample Count": len(suite3_one_sided_indices), "Pct of Dataset": f"{100*len(suite3_one_sided_indices)/len(blends):.1f}%", "Description": "Upper compositions (f > 0.50) of pairs trained on lower side"},
        {"Split / Test Suite": "Suite 4: Equimolar Gap Interp", "Sample Count": len(suite4_gap_interpolation_indices), "Pct of Dataset": f"{100*len(suite4_gap_interpolation_indices)/len(blends):.1f}%", "Description": "Core (0.30 <= f <= 0.70) of pairs trained on tails only"},
    ]
    summary_df = pd.DataFrame(stats_rows)
    summary_path = proc_dir / "splits_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"\n{summary_df.to_string(index=False)}")

if __name__ == "__main__":
    main()
