#!/usr/bin/env python3
"""
Phase 2: Apply labeling to unlabeled timeline CSVs.

Reads *_v4_unlabeled.csv from data/timelines/train/ and test/, loads injury data,
and writes *_v4_labeled.csv (or a custom suffix) with target1 and target2.

Labeling rules (default):
- Target1 (muscular): Positive if reference_date in [D-10, D-1] with D = onset of a
  muscular or unknown injury. Negative otherwise.
- Target2 (skeletal): Positive if reference_date in [D-21, D-1] with D = onset of a
  skeletal or unknown injury. Negative otherwise.

Variant (Option A): use --muscular-only --skeletal-only --muscular-days 7 to get
- Target1: muscular only, window [D-7, D-1].
- Target2: skeletal only, window [D-21, D-1].
Output is written to _v4_labeled_muscle_skeletal_only_d7.csv unless --labeled-suffix is set.

Variant (MSU): use --msu-d7 to get
- target_msu: positive if reference_date in [D-7, D-1] for any muscular, skeletal, or unknown injury.
- target1 and target2 are also set (same as muscular-only D-7 and skeletal-only D-21) for compatibility.
Output is written to _v4_labeled_msu_d7.csv unless --labeled-suffix is set.

Usage:
  python apply_labeling_v4.py [--injuries-file PATH] [--timelines-dir PATH] [--dry-run]
  python apply_labeling_v4.py --muscular-only --skeletal-only --muscular-days 7 [--labeled-suffix SUFFIX]
  python apply_labeling_v4.py --msu-d7 [--labeled-suffix SUFFIX]
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
V4_ROOT = SCRIPT_DIR.parent.parent
V4_DATA_DIR = V4_ROOT / "data"
V4_RAW_DATA = V4_DATA_DIR / "raw_data"
V4_TIMELINES = V4_DATA_DIR / "timelines"

# Labeling parameters
MUSCULAR_DAYS_BEFORE = 10   # target1: ref_date in [D-10, D-1]
SKELETAL_DAYS_BEFORE = 21   # target2: ref_date in [D-21, D-1]
MSU_DAYS_BEFORE = 7         # target_msu: ref_date in [D-7, D-1] for muscular/skeletal/unknown
TARGET1_CLASSES = {"muscular", "unknown"}   # injury classes that define target1 positives
TARGET2_CLASSES = {"skeletal", "unknown"}  # injury classes that define target2 positives
MSU_CLASSES = {"muscular", "skeletal", "unknown"}  # injury classes for combined MSU target [D-7, D-1]

UNLABELED_SUFFIX = "_v4_unlabeled.csv"
LABELED_SUFFIX = "_v4_labeled.csv"


def load_injury_class_map(injuries_file: str):
    """Load (player_id, onset_date) -> injury_class from the enhanced module."""
    from create_35day_timelines_v4_enhanced import load_injuries_data
    return load_injuries_data(injuries_file)


def build_positive_reference_sets(
    injury_class_map: dict,
    target1_classes=None,
    target2_classes=None,
    muscular_days_before=None,
    skeletal_days_before=None,
):
    """
    Build per-player sets of reference dates that are target1=1 and target2=1.

    Optional args override module defaults when provided.
    Returns:
        positive_refs_t1: Dict[player_id, Set[ref_date]]
        positive_refs_t2: Dict[player_id, Set[ref_date]]
    """
    t1_classes = target1_classes if target1_classes is not None else TARGET1_CLASSES
    t2_classes = target2_classes if target2_classes is not None else TARGET2_CLASSES
    mus_days = muscular_days_before if muscular_days_before is not None else MUSCULAR_DAYS_BEFORE
    skel_days = skeletal_days_before if skeletal_days_before is not None else SKELETAL_DAYS_BEFORE

    positive_refs_t1 = defaultdict(set)
    positive_refs_t2 = defaultdict(set)

    for (player_id, onset_d), cls in injury_class_map.items():
        player_id = int(player_id)
        onset_n = pd.Timestamp(onset_d).normalize()
        if getattr(onset_n, "tz", None) is not None:
            onset_n = onset_n.tz_localize(None)
        cls = (cls or "").strip().lower()

        if cls in t1_classes:
            for k in range(1, mus_days + 1):
                ref = onset_n - timedelta(days=k)
                positive_refs_t1[player_id].add(ref)

        if cls in t2_classes:
            for k in range(1, skel_days + 1):
                ref = onset_n - timedelta(days=k)
                positive_refs_t2[player_id].add(ref)

    return dict(positive_refs_t1), dict(positive_refs_t2)


def build_positive_reference_set_msu(
    injury_class_map: dict,
    classes=None,
    days_before=None,
):
    """
    Build per-player set of reference dates that are target_msu=1:
    ref in [D-N, D-1] for injury class in {muscular, skeletal, unknown}.

    Returns:
        positive_refs_msu: Dict[player_id, Set[ref_date]]
    """
    cls_set = classes if classes is not None else MSU_CLASSES
    n_days = days_before if days_before is not None else MSU_DAYS_BEFORE
    positive_refs_msu = defaultdict(set)
    for (player_id, onset_d), cls in injury_class_map.items():
        player_id = int(player_id)
        onset_n = pd.Timestamp(onset_d).normalize()
        if getattr(onset_n, "tz", None) is not None:
            onset_n = onset_n.tz_localize(None)
        cls = (cls or "").strip().lower()
        if cls not in cls_set:
            continue
        for k in range(1, n_days + 1):
            ref = onset_n - timedelta(days=k)
            positive_refs_msu[player_id].add(ref)
    return dict(positive_refs_msu)

def apply_labels_to_df(
    df: pd.DataFrame,
    positive_refs_t1: dict,
    positive_refs_t2: dict,
    positive_refs_msu: dict = None,
) -> pd.DataFrame:
    """Add target1 and target2 columns; optionally target_msu if positive_refs_msu is provided.
    reference_date must be parseable to datetime."""
    df = df.copy()
    ref_dates = pd.to_datetime(df["reference_date"], errors="coerce")
    player_ids = df["player_id"].astype(int)

    t1 = []
    t2 = []
    t_msu = [] if positive_refs_msu is not None else None
    for i in range(len(df)):
        pid = player_ids.iloc[i]
        ref_n = ref_dates.iloc[i]
        if pd.isna(ref_n):
            t1.append(0)
            t2.append(0)
            if t_msu is not None:
                t_msu.append(0)
            continue
        ref_n = pd.Timestamp(ref_n).normalize()
        if getattr(ref_n, "tz", None) is not None:
            ref_n = ref_n.tz_localize(None)
        t1.append(1 if ref_n in positive_refs_t1.get(pid, set()) else 0)
        t2.append(1 if ref_n in positive_refs_t2.get(pid, set()) else 0)
        if t_msu is not None:
            t_msu.append(1 if ref_n in positive_refs_msu.get(pid, set()) else 0)

    df["target1"] = t1
    df["target2"] = t2
    if t_msu is not None:
        df["target_msu"] = t_msu
    return df


def main():
    parser = argparse.ArgumentParser(description="Apply target1/target2 labeling to unlabeled timeline CSVs")
    parser.add_argument("--injuries-file", type=str, default=None,
                        help=f"Injuries CSV (default: {V4_RAW_DATA}/injuries_data.csv)")
    parser.add_argument("--timelines-dir", type=str, default=None,
                        help=f"Base timelines directory (default: {V4_TIMELINES})")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, do not write files")
    parser.add_argument("--muscular-only", action="store_true",
                        help="Target1 positives: muscular injury only (exclude unknown)")
    parser.add_argument("--skeletal-only", action="store_true",
                        help="Target2 positives: skeletal injury only (exclude unknown)")
    parser.add_argument("--muscular-days", type=int, default=10,
                        help="Target1 positive window: [D-N, D-1] (default: 10)")
    parser.add_argument("--labeled-suffix", type=str, default=None,
                        help="Output file suffix (e.g. _v4_labeled_muscle_skeletal_only_d7.csv). Default when using variant: _v4_labeled_muscle_skeletal_only_d7.csv")
    parser.add_argument("--msu-d7", action="store_true",
                        help="Add target_msu: positive if ref in [D-7, D-1] for muscular/skeletal/unknown. Also set target1 (muscular D-7), target2 (skeletal D-21). Output: _v4_labeled_msu_d7.csv")
    args = parser.parse_args()

    injuries_file = args.injuries_file or str(V4_RAW_DATA / "injuries_data.csv")
    timelines_dir = Path(args.timelines_dir or str(V4_TIMELINES))

    use_msu_d7 = args.msu_d7
    if use_msu_d7:
        t1_classes = {"muscular"}
        t2_classes = {"skeletal"}
        mus_days = 7
        if args.labeled_suffix:
            labeled_suffix = args.labeled_suffix
        else:
            labeled_suffix = "_v4_labeled_msu_d7.csv"
    else:
        t1_classes = {"muscular"} if args.muscular_only else TARGET1_CLASSES
        t2_classes = {"skeletal"} if args.skeletal_only else TARGET2_CLASSES
        mus_days = args.muscular_days
        if args.labeled_suffix:
            labeled_suffix = args.labeled_suffix
        elif args.muscular_only and args.skeletal_only and args.muscular_days == 7:
            labeled_suffix = "_v4_labeled_muscle_skeletal_only_d7.csv"
        else:
            labeled_suffix = LABELED_SUFFIX

    if not os.path.exists(injuries_file):
        print(f"ERROR: Injuries file not found: {injuries_file}")
        return 1

    print("Loading injury data...")
    injury_class_map = load_injury_class_map(injuries_file)
    print(f"Building positive reference sets (target1: D-{mus_days}..D-1 classes={t1_classes}; target2: D-{SKELETAL_DAYS_BEFORE}..D-1 classes={t2_classes})...")
    positive_refs_t1, positive_refs_t2 = build_positive_reference_sets(
        injury_class_map,
        target1_classes=t1_classes,
        target2_classes=t2_classes,
        muscular_days_before=mus_days,
        skeletal_days_before=SKELETAL_DAYS_BEFORE,
    )
    print(f"  Players with target1 positives: {len(positive_refs_t1):,}")
    print(f"  Players with target2 positives: {len(positive_refs_t2):,}")

    positive_refs_msu = None
    if use_msu_d7:
        print(f"Building MSU positive reference set (target_msu: D-{MSU_DAYS_BEFORE}..D-1 classes={MSU_CLASSES})...")
        positive_refs_msu = build_positive_reference_set_msu(injury_class_map)
        print(f"  Players with target_msu positives: {len(positive_refs_msu):,}")

    print("Applying labels to unlabeled timeline files...")
    for subdir in ("train", "test"):
        in_dir = timelines_dir / subdir
        if not in_dir.exists():
            print(f"  Skip {subdir}: directory not found")
            continue
        for path in sorted(in_dir.glob(f"*{UNLABELED_SUFFIX}")):
            out_path = in_dir / path.name.replace(UNLABELED_SUFFIX, labeled_suffix)
            print(f"  {path.name} -> {out_path.name}")
            if args.dry_run:
                continue
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            if "reference_date" not in df.columns or "player_id" not in df.columns:
                print(f"    ERROR: missing reference_date or player_id")
                continue
            df = apply_labels_to_df(df, positive_refs_t1, positive_refs_t2, positive_refs_msu=positive_refs_msu)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            n1 = df["target1"].sum()
            n2 = df["target2"].sum()
            msg = f"    Rows: {len(df):,}  target1=1: {int(n1):,}  target2=1: {int(n2):,}"
            if positive_refs_msu is not None and "target_msu" in df.columns:
                n_msu = df["target_msu"].sum()
                msg += f"  target_msu=1: {int(n_msu):,}"
            print(msg)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
