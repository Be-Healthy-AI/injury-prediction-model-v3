#!/usr/bin/env python3
"""
Run labeling experiments (target1 only): load labeled timelines once, apply
per-experiment ruling-out rules to train and test, train 500-feature LGBM, report test Gini.

Experiments:
  1: Baseline (no ruling out).
  2: Rule out ref_date in [D-35, D-11].
  3: Rule out [D-35, D-6] and positives in [D-10, D-6].
  4: Rule out [D-35, D-11] and [D, R] (injury period).
  5: Rule out [D-35, D-6], [D, R], and positives in [D-10, D-6].
  6: Rule out negatives only if any injury in next 35 days [ref, ref+34].
  7: Same as 6 for negatives; keep only positives in [D-10, D-5].
  8: Exp 3 + activity: require >=90 min OR >=1 match in past 35 days; rehab exception (within 60 days after injury end).
  9: Positives unchanged (D-7). Negatives only if no muscular/skeletal/unknown injury in [ref, ref+35].
  10: Same as 9 but use only onset of muscular/skeletal/unknown (not full period). Negatives dropped only if [ref, ref+35] contains an injury start date.

Usage:
  python run_labeling_experiments_500.py [--no-cache] [--verbose]
  python run_labeling_experiments_500.py --exp 8   # run only experiment 8
  python run_labeling_experiments_500.py --labeled-suffix _v4_labeled_muscle_skeletal_only_d7.csv --exp 3 --exp 7
"""

import sys
import glob
import os
import json
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import joblib

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
TRAIN_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "data" / "timelines" / "train"
TEST_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "data" / "timelines" / "test"
V4_RAW_DATA = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "data" / "raw_data"
INJURIES_FILE = V4_RAW_DATA / "injuries_data.csv"
BEST_FEATURES_PATH = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "models" / "lgbm_muscular_best_iteration_features.json"
MODEL_OUTPUT_DIR = BEST_FEATURES_PATH.parent
EXCLUDE_SEASON = "2025_2026"
DEFAULT_LABELED_SUFFIX = "_v4_labeled.csv"

# Add code paths for imports
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Muscular/unknown positive windows
TARGET1_CLASSES = {"muscular", "unknown"}
MUSCULAR_DAYS_BEFORE = 10  # positives [D-10, D-1]
NEXT_35_DAYS = 34          # [ref_date, ref_date+34] inclusive
MIN_ACTIVITY_MINUTES = 90  # activity: >=90 min in past 35 days
REHAB_DAYS = 60            # within 60 days after injury end, activity check bypassed


def _norm(ts):
    t = pd.Timestamp(ts).normalize()
    if getattr(t, "tz", None) is not None:
        t = t.tz_localize(None)
    return t


def load_labeled_train_test(labeled_suffix=None, exclude_train_seasons=None):
    """Load labeled train (all seasons except test) and test (2025_2026) once.
    labeled_suffix: e.g. _v4_labeled.csv or _v4_labeled_muscle_skeletal_only_d7.csv (default: DEFAULT_LABELED_SUFFIX).
    exclude_train_seasons: optional set of season strings to exclude from train (e.g. {"2020_2021", "2021_2022"}).
    """
    suffix = labeled_suffix if labeled_suffix is not None else DEFAULT_LABELED_SUFFIX
    exclude = set(exclude_train_seasons) if exclude_train_seasons else set()
    pattern = str(TRAIN_DIR / ("timelines_35day_season_*" + suffix))
    files = glob.glob(pattern)
    train_dfs = []
    for f in sorted(files):
        base = os.path.basename(f)
        if EXCLUDE_SEASON in base or "_10pc_" in base or "_25pc_" in base or "_50pc_" in base:
            continue
        if "season_" in base:
            parts = base.split("season_")
            if len(parts) > 1:
                # e.g. "2020_2021_v4_labeled.csv" or "2020_2021_v4_labeled_muscle_skeletal_only_d7.csv"
                season = parts[1].split(suffix)[0].strip("_")
                if season == EXCLUDE_SEASON or season in exclude:
                    continue
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        train_dfs.append(df)
    if not train_dfs:
        raise FileNotFoundError(f"No train files found for pattern {pattern}")
    df_train = pd.concat(train_dfs, ignore_index=True)

    test_file = TEST_DIR / ("timelines_35day_season_2025_2026" + suffix)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    df_test = pd.read_csv(test_file, encoding="utf-8-sig", low_memory=False)

    df_train["reference_date"] = pd.to_datetime(df_train["reference_date"], errors="coerce")
    df_test["reference_date"] = pd.to_datetime(df_test["reference_date"], errors="coerce")
    return df_train, df_test


def build_exclusion_sets(injury_class_map, muscular_unknown_periods, all_injury_periods, muscular_skeletal_unknown_periods=None):
    """Build sets of (player_id, ref_date) for ruling out. All dates normalized.
    muscular_skeletal_unknown_periods: optional, for Exp 9 (negatives with no muscle/skeletal/unknown in [ref, ref+35]).
    """
    excl_D35_D11 = set()
    excl_D35_D6 = set()
    excl_D_R = set()
    positive_refs_D10_D6 = set()
    positive_refs_D5_D1 = set()
    positive_refs_D10_D5 = set()

    for (player_id, onset_d), cls in injury_class_map.items():
        pid = int(player_id)
        onset = _norm(onset_d)
        cls = (cls or "").strip().lower()
        if cls not in TARGET1_CLASSES:
            continue
        for k in range(11, 36):  # D-35 .. D-11
            excl_D35_D11.add((pid, onset - timedelta(days=k)))
        for k in range(6, 36):   # D-35 .. D-6
            excl_D35_D6.add((pid, onset - timedelta(days=k)))
        for k in range(6, 11):   # D-10 .. D-6
            positive_refs_D10_D6.add((pid, onset - timedelta(days=k)))
        for k in range(1, 6):    # D-5 .. D-1
            positive_refs_D5_D1.add((pid, onset - timedelta(days=k)))
        for k in range(5, 11):   # D-10 .. D-5
            positive_refs_D10_D5.add((pid, onset - timedelta(days=k)))

    for pid, periods in muscular_unknown_periods.items():
        for onset, end in periods:
            onset = _norm(onset)
            end = _norm(end)
            d = onset
            while d <= end:
                excl_D_R.add((pid, d))
                d += timedelta(days=1)

    # Negatives with any injury in [ref, ref+34]: for each player, ref_dates to exclude
    negative_excl_next_35 = set()
    for pid, periods in all_injury_periods.items():
        for onset, end in periods:
            onset = _norm(onset)
            end = _norm(end)
            # ref_date such that [ref_date, ref_date+34] overlaps [onset, end]
            # ref_date <= end and ref_date + 34 >= onset  =>  ref_date in [onset-34, end]
            ref_start = onset - timedelta(days=NEXT_35_DAYS)
            ref_end = end
            d = ref_start
            while d <= ref_end:
                negative_excl_next_35.add((pid, d))
                d += timedelta(days=1)

    # Exp 9: negatives with muscular/skeletal/unknown injury in [ref, ref+34]
    negative_excl_next_35_muscle_skel_unknown = set()
    if muscular_skeletal_unknown_periods:
        for pid, periods in muscular_skeletal_unknown_periods.items():
            for onset, end in periods:
                onset = _norm(onset)
                end = _norm(end)
                ref_start = onset - timedelta(days=NEXT_35_DAYS)
                ref_end = end
                d = ref_start
                while d <= ref_end:
                    negative_excl_next_35_muscle_skel_unknown.add((pid, d))
                    d += timedelta(days=1)

    # Exp 10: negatives when [ref, ref+34] contains onset of muscular/skeletal/unknown (onset only)
    from timelines.create_35day_timelines_v4_enhanced import build_exp10_onset_only_exclusion_set
    negative_excl_next_35_onset_only = build_exp10_onset_only_exclusion_set(injury_class_map)

    return {
        "excl_D35_D11": excl_D35_D11,
        "excl_D35_D6": excl_D35_D6,
        "excl_D_R": excl_D_R,
        "positive_refs_D10_D6": positive_refs_D10_D6,
        "positive_refs_D5_D1": positive_refs_D5_D1,
        "positive_refs_D10_D5": positive_refs_D10_D5,
        "negative_excl_next_35": negative_excl_next_35,
        "negative_excl_next_35_muscle_skel_unknown": negative_excl_next_35_muscle_skel_unknown,
        "negative_excl_next_35_onset_only": negative_excl_next_35_onset_only,
    }


def build_rehab_bypass_set(all_injury_periods):
    """Set of (player_id, ref_date) within 60 days after any injury end (rehab exception)."""
    out = set()
    for pid, periods in all_injury_periods.items():
        for onset, end in periods:
            end_n = _norm(end)
            for d in range(1, REHAB_DAYS + 1):
                ref = end_n + timedelta(days=d)
                out.add((int(pid), ref))
    return out


def activity_drop_mask(df, rehab_bypass_set):
    """True for rows to drop due to activity: no minimum activity and not in rehab bypass.
    Activity pass: has_minimum_activity==1 OR total matches in past 35 days >= 1.
    """
    player_ids = df["player_id"].astype(int)
    ref_dates = df["reference_date"]
    n = len(df)
    drop = np.zeros(n, dtype=bool)

    has_activity_col = "has_minimum_activity" in df.columns
    match_cols = [c for c in df.columns if c.startswith("matches_played_week_") and c.endswith(("_1", "_2", "_3", "_4", "_5"))]
    match_cols = sorted(match_cols)[:5]

    for i in range(n):
        pid = player_ids.iloc[i]
        ref = ref_dates.iloc[i]
        if pd.isna(ref):
            drop[i] = True
            continue
        ref = _norm(ref)
        if (pid, ref) in rehab_bypass_set:
            continue
        activity_ok = False
        if has_activity_col:
            val = df["has_minimum_activity"].iloc[i]
            if val == 1 or (pd.notna(val) and int(val) >= 1):
                activity_ok = True
        if not activity_ok and match_cols:
            total_matches = 0
            for c in match_cols:
                v = df[c].iloc[i]
                if pd.notna(v):
                    total_matches += int(v)
            if total_matches >= 1:
                activity_ok = True
        if not activity_ok:
            drop[i] = True
    return pd.Series(drop, index=df.index)


def mask_drop_for_experiment(df, exclusions, exp_num):
    """Return a boolean Series True for rows to drop (rule out)."""
    player_ids = df["player_id"].astype(int)
    ref_dates = df["reference_date"]
    target1 = df["target1"].values
    n = len(df)
    drop = np.zeros(n, dtype=bool)

    for i in range(n):
        pid = player_ids.iloc[i]
        ref = ref_dates.iloc[i]
        if pd.isna(ref):
            drop[i] = True
            continue
        ref = _norm(ref)
        key = (pid, ref)
        is_pos = target1[i] == 1

        if exp_num == 1:
            pass
        elif exp_num == 2:
            drop[i] = key in exclusions["excl_D35_D11"]
        elif exp_num == 3:
            drop[i] = key in exclusions["excl_D35_D6"] or (is_pos and key in exclusions["positive_refs_D10_D6"])
        elif exp_num == 4:
            drop[i] = key in exclusions["excl_D35_D11"] or key in exclusions["excl_D_R"]
        elif exp_num == 5:
            drop[i] = (
                key in exclusions["excl_D35_D6"]
                or key in exclusions["excl_D_R"]
                or (is_pos and key in exclusions["positive_refs_D10_D6"])
            )
        elif exp_num == 6:
            drop[i] = not is_pos and key in exclusions["negative_excl_next_35"]
        elif exp_num == 7:
            if is_pos:
                drop[i] = key not in exclusions["positive_refs_D10_D5"]  # keep only [D-10, D-5]
            else:
                drop[i] = key in exclusions["negative_excl_next_35"]
        elif exp_num == 8:
            drop[i] = key in exclusions["excl_D35_D6"] or (is_pos and key in exclusions["positive_refs_D10_D6"])
        elif exp_num == 9:
            drop[i] = not is_pos and key in exclusions["negative_excl_next_35_muscle_skel_unknown"]
        elif exp_num == 10:
            drop[i] = not is_pos and key in exclusions["negative_excl_next_35_onset_only"]
        else:
            raise ValueError(f"Unknown experiment {exp_num}")
    return pd.Series(drop, index=df.index)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run labeling experiments, report test Gini")
    parser.add_argument("--no-cache", action="store_true", help="Disable prepare_data cache")
    parser.add_argument("--verbose", action="store_true", help="Verbose training logs")
    parser.add_argument("--exp", type=int, nargs="*", default=None, help="Run only these experiments (e.g. --exp 3 7). Default: all 1-8")
    parser.add_argument("--labeled-suffix", type=str, default=None,
                        help="Labeled timeline file suffix (e.g. _v4_labeled_muscle_skeletal_only_d7.csv). Default: _v4_labeled.csv")
    parser.add_argument("--test-negatives-before", type=str, default=None,
                        help="Eval only: keep all positives; drop negatives with reference_date >= this date (YYYY-MM-DD, e.g. 2025-11-01)")
    parser.add_argument("--exclude-train-seasons", type=str, nargs="*", default=None,
                        help="Exclude these seasons from training only (e.g. --exclude-train-seasons 2020_2021 2021_2022)")
    parser.add_argument(
        "--algorithm",
        choices=["lgbm", "gb"],
        default="lgbm",
        help="Algorithm to use for training (default: lgbm).",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    verbose = args.verbose
    labeled_suffix = args.labeled_suffix if args.labeled_suffix else DEFAULT_LABELED_SUFFIX
    test_negatives_before = args.test_negatives_before  # None or date string
    exclude_train_seasons = args.exclude_train_seasons  # None or list of "YYYY_YYYY"
    experiments_to_run = args.exp if args.exp else list(range(1, 11))
    if test_negatives_before:
        try:
            pd.Timestamp(test_negatives_before)
        except Exception as e:
            print(f"ERROR: Invalid --test-negatives-before date: {test_negatives_before}: {e}")
            return 1

    print("Loading labeled train and test once...")
    if labeled_suffix != DEFAULT_LABELED_SUFFIX:
        print(f"  Using labeled suffix: {labeled_suffix}")
    if exclude_train_seasons:
        print(f"  Excluding train seasons: {exclude_train_seasons}")
    df_train_full, df_test_full = load_labeled_train_test(
        labeled_suffix=labeled_suffix,
        exclude_train_seasons=exclude_train_seasons,
    )
    print(f"  Train: {len(df_train_full):,} rows, target1=1: {(df_train_full['target1']==1).sum():,}")
    print(f"  Test:  {len(df_test_full):,} rows, target1=1: {(df_test_full['target1']==1).sum():,}")

    print("Loading injury data...")
    from timelines.create_35day_timelines_v4_enhanced import (
        load_injuries_data,
        load_muscular_unknown_injury_periods,
        load_all_injury_periods,
        load_muscular_skeletal_unknown_injury_periods,
    )
    injury_class_map = load_injuries_data(str(INJURIES_FILE))
    muscular_unknown_periods = load_muscular_unknown_injury_periods(str(INJURIES_FILE))
    all_injury_periods = load_all_injury_periods(str(INJURIES_FILE))
    muscular_skeletal_unknown_periods = load_muscular_skeletal_unknown_injury_periods(str(INJURIES_FILE))
    print("Building exclusion sets...")
    exclusions = build_exclusion_sets(
        injury_class_map,
        muscular_unknown_periods,
        all_injury_periods,
        muscular_skeletal_unknown_periods=muscular_skeletal_unknown_periods,
    )
    rehab_bypass_set = build_rehab_bypass_set(all_injury_periods)
    print(f"  Rehab bypass (ref within {REHAB_DAYS} days after injury end): {len(rehab_bypass_set):,} (player, ref) pairs")

    if not BEST_FEATURES_PATH.exists():
        print(f"ERROR: Best features not found: {BEST_FEATURES_PATH}")
        return 1
    with open(BEST_FEATURES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_list = meta["features"]
    print(f"Loaded {len(feature_list)} features from {BEST_FEATURES_PATH.name}")

    from train_iterative_feature_selection_muscular_standalone import (
        train_and_evaluate_on_dataframes,
    )

    results = []
    for exp in experiments_to_run:
        print(f"\n{'='*60}\nExperiment {exp}\n{'='*60}")
        if exp == 8:
            mask_train = mask_drop_for_experiment(df_train_full, exclusions, 8) | activity_drop_mask(df_train_full, rehab_bypass_set)
            mask_test = mask_drop_for_experiment(df_test_full, exclusions, 8) | activity_drop_mask(df_test_full, rehab_bypass_set)
        else:
            mask_train = mask_drop_for_experiment(df_train_full, exclusions, exp)
            mask_test = mask_drop_for_experiment(df_test_full, exclusions, exp)
        df_train_e = df_train_full.loc[~mask_train].reset_index(drop=True)
        df_test_e = df_test_full.loc[~mask_test].reset_index(drop=True)
        if test_negatives_before:
            cutoff = pd.Timestamp(test_negatives_before)
            ref = pd.to_datetime(df_test_e["reference_date"], errors="coerce")
            keep = (df_test_e["target1"] == 1) | (ref < cutoff)
            n_before = len(df_test_e)
            n_neg_dropped = ((df_test_e["target1"] == 0) & (ref >= cutoff)).sum()
            df_test_e = df_test_e.loc[keep].reset_index(drop=True)
            print(f"  Eval filter: negatives with ref_date >= {test_negatives_before} dropped ({int(n_neg_dropped):,}); test rows {n_before:,} -> {len(df_test_e):,}")
        n_pos_train = (df_train_e["target1"] == 1).sum()
        n_pos_test = (df_test_e["target1"] == 1).sum()
        print(f"  Train: {len(df_train_e):,} rows, target1=1: {int(n_pos_train):,}")
        print(f"  Test:  {len(df_test_e):,} rows, target1=1: {int(n_pos_test):,}")
        if n_pos_train == 0 or n_pos_test == 0:
            print("  Skipping: no positives in train or test")
            results.append({"exp": exp, "test_gini": None, "skip": True})
            continue
        data_suffix = "_d7" if labeled_suffix != DEFAULT_LABELED_SUFFIX else ""
        if test_negatives_before:
            data_suffix += "_eval_preNov01"
        if exclude_train_seasons:
            data_suffix += "_no" + "_".join(sorted(exclude_train_seasons))
        out = train_and_evaluate_on_dataframes(
            df_train_e,
            df_test_e,
            feature_list,
            cache_suffix=f"exp{exp}{data_suffix}",
            use_cache=use_cache,
            verbose=verbose,
            algorithm=args.algorithm,
            hyperparameter_set="below",
        )
        gini = out["test_gini"]
        results.append({"exp": exp, "test_gini": gini, "skip": False})
        print(f"  Test Gini: {gini:.4f}")
        if exp == 10:
            MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_OUTPUT_DIR / "lgbm_muscular_exp10_labeling.joblib"
            joblib.dump(out["model"], model_path)
            meta_path = MODEL_OUTPUT_DIR / "lgbm_muscular_exp10_labeling_metadata.json"
            meta = {
                "experiment": 10,
                "test_gini": gini,
                "test_metrics": out["test_metrics"],
                "features": out["feature_names_used"],
                "description": "Exp 10: negatives dropped only if [ref, ref+35] contains onset of muscular/skeletal/unknown.",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"  Saved: {model_path.name}, {meta_path.name}")
        if exp == 8:
            print(f"  (Exp 3 baseline: 0.4398)")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Exp':<6} {'Test Gini':<12}")
    print("-"*20)
    for r in results:
        if r.get("skip"):
            print(f"{r['exp']:<6} (skipped)")
        else:
            print(f"{r['exp']:<6} {r['test_gini']:.4f}")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
