#!/usr/bin/env python3
"""
Build TruthSeeker train/valid/test splits (4-label, strict stratification) + label-map JSON.

Adds:
- Writes a label-map JSON for `5_label_majority_answer`. By default:
    {
      "Agree": 0,
      "Mostly Agree": 1,
      "Mostly Disagree": 2,
      "Disagree": 3
    }

Still does:
- Removes rows where 5_label_majority_answer == "NO MAJORITY"
- Allowed labels: {"Agree", "Disagree", "Mostly Agree", "Mostly Disagree"}
- Drops columns: 'Unnamed: 0_x', '3_label_majority_answer', 'Unnamed: 0_y', 'BinaryNumTarget' (+ 'target', 'manual_keywords')
- Appends ONLY numeric columns from Features_For_Traditional_ML_Techniques.csv (join on ["statement","tweet"])
- Reports numeric feature count (+ optional list)
- Drops blank targets; drops tweets with <= 3 words
- Target column last
- Strict stratified 80/10/10; exits with helpful error if impossible
- Prints class-wise counts and final column titles
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ALLOWED_4LABELS = {"Agree", "Disagree", "Mostly Agree", "Mostly Disagree"}
# Consistent label order for mapping & printing
LABEL_ORDER = ["Agree", "Mostly Agree", "Mostly Disagree", "Disagree"]

# Columns to remove from the Truth Seeker file up front
EXCLUDE_TS_COLS = {
    "target",
    "manual_keywords",
    "BinaryNumTarget",
    "3_label_majority_answer",
    "Unnamed: 0_x",
    "Unnamed: 0_y",  # if present in TS
}

# Columns to force-drop after merge (if any sneak in)
FORCE_DROP_AFTER_MERGE = {
    "BinaryNumTarget",
    "3_label_majority_answer",
    "Unnamed: 0_x",
    "Unnamed: 0_y",
}

TARGET_COL = "5_label_majority_answer"
JOIN_KEYS = ["statement", "tweet"]


def read_csv_strict(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        sys.exit(f"ERROR: Failed to read CSV '{path}': {e}")


def tweet_word_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.split().map(len)


def prepare_features(feats: pd.DataFrame):
    # Ensure join keys exist
    for col in JOIN_KEYS:
        if col not in feats.columns:
            sys.exit(f"ERROR: Column '{col}' not found in Features CSV.")
    # Numeric-only feature columns (excluding keys)
    numeric_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in JOIN_KEYS]
    if not numeric_cols:
        return feats[JOIN_KEYS].drop_duplicates().copy(), []

    # Aggregate duplicates by mean to get one row per (statement, tweet)
    grouped = feats.groupby(JOIN_KEYS, as_index=False)[numeric_cols].mean()
    return grouped, numeric_cols


def ensure_target_last(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target] + [target]
    return df[cols]


def stratified_three_way(df: pd.DataFrame, target: str, seed: int):
    """Strict stratified 80/10/10 (via 80/20 then 50/50 on temp). Exit on failure with helpful counts."""
    y = df[target].values

    # 80/20
    try:
        splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        (train_idx, temp_idx), = splitter1.split(df, y)
    except ValueError as e:
        counts = df[target].value_counts().sort_index()
        sys.exit(
            "ERROR: Stratified 80/20 split failed. "
            f"Reason: {e}\nPer-class counts (after cleaning):\n{counts.to_string()}"
        )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # 50/50 on temp -> valid/test
    try:
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        (valid_idx, test_idx), = splitter2.split(temp_df, temp_df[target].values)
    except ValueError as e:
        counts_temp = temp_df[target].value_counts().sort_index()
        sys.exit(
            "ERROR: Stratified 50/50 split on the 20% temp set failed "
            f"(this corresponds to 10%/10% valid/test). Reason: {e}\n"
            f"Per-class counts in temp set:\n{counts_temp.to_string()}"
        )

    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df  = temp_df.iloc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df


def print_class_counts(df: pd.DataFrame, name: str):
    counts = df[TARGET_COL].value_counts().to_dict()
    print(f"{name} class counts:")
    for lab in LABEL_ORDER:
        print(f"  {lab}: {counts.get(lab, 0)}")
    print(f"  TOTAL: {len(df)}")


def print_column_titles(df: pd.DataFrame):
    print(f"\nColumn titles (in order) — total {len(df.columns)} columns:")
    for i, c in enumerate(df.columns, start=1):
        print(f"  {i:>3}. {c}")


def write_label_map_json(out_path: Path, present_labels: pd.Index | list[str]):
    """
    Write label mapping JSON using LABEL_ORDER for stable indices.
    If some labels are missing (shouldn't happen with stratification), they are still included for consistency.
    """
    label_to_id = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)
    print(f"\nWrote label map JSON -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truthseeker-csv", required=True,
                    help="Path to Truth_Seeker_Model_Dataset.csv")
    ap.add_argument("--features-csv", required=True,
                    help="Path to Features_For_Traditional_ML_Techniques.csv")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for outputs (CSV splits + label map)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-numeric-names", action="store_true",
                    help="Also print the list of numeric feature column names.")
    ap.add_argument("--label-map-json", default=None,
                    help="Optional explicit path for the label-map JSON. "
                         "Default: <output-dir>/label_mapping_4label.json")
    ap.add_argument("--train-subsample", action="store_true",
                    help="If set, stratify-subsample 20% of the TRAIN split and use that as train.csv.")

    args = ap.parse_args()

    ts_path = Path(args.truthseeker_csv)
    feat_path = Path(args.features_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    ts = read_csv_strict(ts_path)
    feats = read_csv_strict(feat_path)

    # 2) Keep TS columns (minus excluded) & basic checks
    ts_cols = [c for c in ts.columns if c not in EXCLUDE_TS_COLS]
    missing = [c for c in (JOIN_KEYS + [TARGET_COL]) if c not in ts_cols]
    if missing:
        sys.exit(f"ERROR: Missing required columns in Truth Seeker CSV (after exclusions): {missing}")
    ts = ts[ts_cols].copy()

    # 3) Clean target: trim and drop blank
    before = len(ts)
    # ts[TARGET_COL] = ts[TARGET_COL].astype(str).str.strip()
    # ts = ts[ts[TARGET_COL].ne("") & ts[TARGET_COL].ne("nan") & ts[TARGET_COL].ne("None")]
    # dropped_empty_target = before - len(ts)

    nan_count_true = ts[TARGET_COL].isna().sum()
    if nan_count_true:
        print(f"WARNING: Dropping {nan_count_true} rows with NaN in '{TARGET_COL}'")
    ts = ts.dropna(subset=[TARGET_COL])

    ts[TARGET_COL] = ts[TARGET_COL].astype(str).str.strip()
    bad_placeholders = {"", "nan", "NaN", "None", "NULL"}
    mask_bad = ts[TARGET_COL].isin(bad_placeholders)
    dropped_empty_target = int(mask_bad.sum())
    ts = ts.loc[~mask_bad].copy()

    # Final sanity check on cleaned label column (pre-merge)
    if ts[TARGET_COL].isna().any():
        n = int(ts[TARGET_COL].isna().sum())
        sample_idx = ts[ts[TARGET_COL].isna()].index[:5].tolist()
        sys.exit(
            f"ERROR: Found {n} NaN values in '{TARGET_COL}' after initial cleaning; sample indices: {sample_idx}"
        )

    # 4) Remove NO MAJORITY rows
    before = len(ts)
    ts = ts[ts[TARGET_COL] != "NO MAJORITY"].copy()
    dropped_no_majority = before - len(ts)

    # 5) Validate labels (keep only 4 allowed; report any unknowns)
    before = len(ts)
    unique_labels = sorted(ts[TARGET_COL].unique().tolist())
    unknown = [lab for lab in unique_labels if lab not in ALLOWED_4LABELS]
    if unknown:
        for bad in unknown:
            print(f"ERROR: Unknown value in {TARGET_COL}: '{bad}'", file=sys.stderr)
        ts = ts[ts[TARGET_COL].isin(ALLOWED_4LABELS)].copy()
    dropped_invalid_target = before - len(ts)

    # 6) Drop short tweets (<= 3 words)
    before = len(ts)
    wc = tweet_word_count(ts["tweet"])
    ts = ts[wc > 3].copy()
    dropped_short_tweets = before - len(ts)

    if ts.empty:
        sys.exit("ERROR: No data left after cleaning (empty dataset).")

    # 7) Prepare numeric features and merge
    feats_prepped, numeric_cols = prepare_features(feats)
    merged = ts.merge(feats_prepped, on=JOIN_KEYS, how="left")

    # 8) Force-drop unwanted columns (if any slipped in)
    drop_now = [c for c in FORCE_DROP_AFTER_MERGE if c in merged.columns and c != TARGET_COL]
    if drop_now:
        merged = merged.drop(columns=drop_now)

    # 9) Ensure target last
    merged = ensure_target_last(merged, TARGET_COL)

    # 10) STRICT stratified split
    train_df, valid_df, test_df = stratified_three_way(merged, TARGET_COL, args.seed)

    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if df[TARGET_COL].isna().any():
            n = int(df[TARGET_COL].isna().sum())
            sys.exit(f"ERROR: {name}.csv contains {n} NaN values in '{TARGET_COL}' after splitting.")


    # Optional: subsample 20% of the TRAIN split (stratified) for train.csv
    if args.train_subsample:
        from sklearn.model_selection import StratifiedShuffleSplit
        y_train = train_df[TARGET_COL].values
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.20, random_state=args.seed)
            (sub_idx, _), = sss.split(train_df, y_train)
            original_train_n = len(train_df)
            train_df = train_df.iloc[sub_idx].reset_index(drop=True)
            print(f"[train-subsample] Reduced train split from {original_train_n} -> {len(train_df)} "
                  f"(20% stratified of the 80% train split).")
            # Show per-class counts of the new (smaller) train
            print_class_counts(train_df, "Train (subsampled)")
        except ValueError as e:
            # If a tiny class prevents stratification, fail with helpful counts
            counts = train_df[TARGET_COL].value_counts().sort_index()
            sys.exit(
                "ERROR: Stratified 20% subsample of the TRAIN split failed. "
                f"Reason: {e}\nPer-class counts in original train split:\n{counts.to_string()}"
            )


    # 11) Save splits
    train_p = out_dir / "train.csv"
    valid_p = out_dir / "valid.csv"
    test_p  = out_dir / "test.csv"
    train_df.to_csv(train_p, index=False)
    valid_df.to_csv(valid_p, index=False)
    test_df.to_csv(test_p, index=False)

    if merged[TARGET_COL].isna().any():
        n = int(merged[TARGET_COL].isna().sum())
        sys.exit(f"ERROR: Found {n} NaN values in '{TARGET_COL}' in merged dataset before writing label map.")

    # 12) Write label-map JSON
    label_map_path = Path(args.label_map_json) if args.label_map_json else (out_dir / "label_mapping_4label.json")
    write_label_map_json(label_map_path, merged[TARGET_COL].unique())

    # 13) Summary
    print("=== Summary (4-label) ===")
    print(f"Numeric feature columns appended: {len(numeric_cols)}")
    if args.print_numeric_names and numeric_cols:
        print("Numeric feature column names:")
        for c in numeric_cols:
            print(f"  - {c}")
    print(f"Dropped rows with empty target: {dropped_empty_target}")
    print(f"Dropped rows with target == 'NO MAJORITY': {dropped_no_majority}")
    print(f"Dropped rows with invalid target (outside 4 labels): {dropped_invalid_target}")
    print(f"Dropped rows with tweet <= 3 words: {dropped_short_tweets}")
    print(f"Final counts -> train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}\n")

    print_class_counts(train_df, "Train")
    print_class_counts(valid_df, "Validation")
    print_class_counts(test_df, "Test")

    # Column titles of the final dataset schema (same for all splits)
    print_column_titles(merged)

    print(f"\nWrote:\n  {train_p}\n  {valid_p}\n  {test_p}\n  {label_map_path}")


if __name__ == "__main__":
    main()
