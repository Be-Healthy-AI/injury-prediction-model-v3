#!/usr/bin/env python3
"""
Create a 10% target ratio timelines file for season 2025-2026
from the natural (unbalanced) v4 muscular timelines.

Input:
  models_production/lgbm_muscular_v1/data/timelines/train/
    - timelines_35day_season_2025_2026_v4_muscular.csv

Output:
  Same folder:
    - timelines_35day_season_2025_2026_10pc_v4_muscular.csv

Logic:
  - Keep ALL positives (target == 1)
  - Sample negatives (target == 0) so that:
        TARGET_RATIO ≈ 0.10 = n_pos / (n_pos + n_neg_sampled)
"""

import io
import os
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datetime import datetime

import numpy as np
import pandas as pd


def main() -> None:
    TARGET_RATIO = 0.10
    RANDOM_STATE = 42

    base_dir = os.path.join(
        os.getcwd(),
        "models_production",
        "lgbm_muscular_v1",
        "data",
        "timelines",
        "train",
    )

    natural_filename = "timelines_35day_season_2025_2026_v4_muscular.csv"
    natural_path = os.path.join(base_dir, natural_filename)

    output_filename = "timelines_35day_season_2025_2026_10pc_v4_muscular.csv"
    output_path = os.path.join(base_dir, output_filename)

    print("=" * 80)
    print("CREATE 10% TARGET RATIO TIMELINES FOR SEASON 2025-2026 (V4 MUSCULAR)")
    print("=" * 80)
    print(f"📂 Base timelines directory : {base_dir}")
    print(f"📥 Input (natural) file     : {natural_filename}")
    print(f"📤 Output (10% ratio) file  : {output_filename}")

    if not os.path.exists(natural_path):
        print(f"\n❌ ERROR: Input file not found: {natural_path}")
        sys.exit(1)

    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n📖 Loading natural 2025-2026 timelines...")
    df = pd.read_csv(natural_path, encoding="utf-8-sig", low_memory=False)

    if "target" not in df.columns:
        print("\n❌ ERROR: 'target' column not found in input file.")
        sys.exit(1)

    n_total = len(df)
    n_pos = int(df["target"].sum())
    n_neg = n_total - n_pos

    print(
        f"   Total rows: {n_total:,} | Positives: {n_pos:,} | Negatives: {n_neg:,} "
        f"(natural ratio: {df['target'].mean():.4%})"
    )

    if n_pos == 0:
        print("\n❌ ERROR: No positives found in 2025-2026 file; cannot build 10% set.")
        sys.exit(1)

    # Desired number of negatives for TARGET_RATIO
    # R = n_pos / (n_pos + n_neg_desired)  =>  n_neg_desired = n_pos*(1-R)/R
    n_neg_desired = int(n_pos * (1 - TARGET_RATIO) / TARGET_RATIO)
    n_neg_desired = min(n_neg_desired, n_neg)

    print(
        f"\n🎯 Target ratio: {TARGET_RATIO:.0%} → aiming for approximately "
        f"{n_neg_desired:,} negatives."
    )

    df_pos = df[df["target"] == 1]
    df_neg = df[df["target"] == 0]

    print(
        f"   Available negatives: {len(df_neg):,} "
        f"({'enough' if len(df_neg) >= n_neg_desired else 'limited'})"
    )

    if n_neg_desired <= 0:
        print("\n⚠️  Computed desired negatives <= 0; using all negatives (degenerate case).")
        df_balanced = df.copy()
    else:
        df_neg_sample = df_neg.sample(
            n=n_neg_desired, random_state=RANDOM_STATE, replace=False
        )
        df_balanced = pd.concat([df_pos, df_neg_sample], ignore_index=True)

    # Shuffle rows to avoid any ordering artefacts
    df_balanced = df_balanced.sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    n_total_bal = len(df_balanced)
    n_pos_bal = int(df_balanced["target"].sum())
    ratio_bal = n_pos_bal / n_total_bal if n_total_bal > 0 else 0.0

    print(
        f"\n✅ Balanced 2025-2026 subset: {n_total_bal:,} rows "
        f"({n_pos_bal:,} positives, ratio: {ratio_bal:.4%})"
    )

    # Save output
    os.makedirs(base_dir, exist_ok=True)
    df_balanced.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved 10% timelines file to:\n   {output_path}")

    total_time = datetime.now() - start_time
    print(f"\n✅ Finished in {total_time}")


if __name__ == "__main__":
    main()













