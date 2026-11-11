"""Build an all-variable normalized mutual-information (NMI) matrix for Brookshire's.

Input is the merged one-row-per-store-per-day CSV made by combine_and_evaluate.py.
Every non-date original variable is included. Date is represented as year/month/weekday
because using 730 unique date labels would generate a misleading identifier MI.

Usage (from folder containing both this script and the CSV):
    python make_all_variables_mi_matrix.py

Other usage:
    python make_all_variables_mi_matrix.py --input "M2/Models/results/merged_all_stores.csv"
    python make_all_variables_mi_matrix.py --all-data
    python make_all_variables_mi_matrix.py --train-end 2025-03-12

Default calculates MI on the earliest 60% of *whole seven-day ad weeks*.
Use --train-end to match your actual modeling split, or --all-data only for
exploration, not predictive feature selection. These are ASSOCIATIONS, not causal tests.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


SALE_TYPE_NAMES = {
    "0": "Low-promotion", "0.0": "Low-promotion",
    "1": "Other ad week", "1.0": "Other ad week",
    "2": "Candidate big-sale", "2.0": "Candidate big-sale",
    "low-promotion": "Low-promotion", "other ad week": "Other ad week",
    "candidate big-sale": "Candidate big-sale",
}


def resolve_input(raw: str) -> Path:
    """Look in working directory, script folder, and common project output folders."""
    arg = Path(raw).expanduser()
    here = Path(__file__).resolve().parent
    candidates = [
        arg,
        here / arg,
        here / arg.name,
        here / "results" / arg.name,
        here / "M2" / "Models" / "results" / arg.name,
        Path.cwd() / "M2" / "Models" / "results" / arg.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find CSV {raw!r}. Current terminal directory: {Path.cwd()}. "
        'Use --input "full/path/to/merged_all_stores.csv".'
    )


def categorize_numeric(series: pd.Series, bins: int, keep_zero_separate: bool) -> pd.Series:
    """Convert continuous data into categories for stable, interpretable NMI.

    Stockout ratios are zero-heavy. Put exactly-zero values in their own bin, then
    use up to bins-1 quantile bins for positive values. Quantile ties are honored.
    """
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    finite = vals.notna() & np.isfinite(vals)
    valid = vals.loc[finite]
    if valid.empty:
        return out
    if valid.nunique() == 1:
        out.loc[valid.index] = "constant"
        return out
    if keep_zero_separate and (valid == 0).any():
        out.loc[valid.index[valid == 0]] = "zero"
        positive = valid[valid > 0]
        negative = valid[valid < 0]
        if not negative.empty:
            out.loc[negative.index] = "negative"
        if positive.empty:
            return out
        if positive.nunique() == 1:
            out.loc[positive.index] = "positive"
        else:
            q = pd.qcut(positive, q=min(bins - 1, positive.nunique()), duplicates="drop")
            out.loc[positive.index] = "positive " + q.astype(str)
        return out
    q = pd.qcut(valid, q=min(bins, valid.nunique()), duplicates="drop")
    out.loc[valid.index] = q.astype(str)
    return out


def make_heatmap(matrix: pd.DataFrame, target: Path, title: str, large: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed; CSV matrices are available, no PNG generated.")
        return
    nrows, ncols = matrix.shape
    size = max(9, 0.48 * max(nrows, ncols) + 5) if large else 10
    fig, ax = plt.subplots(figsize=(size, size * (nrows / max(ncols, 1) * .8 + .3)))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(ncols), matrix.columns, rotation=65, ha="right", fontsize=7 if large else 9)
    ax.set_yticks(range(nrows), matrix.index, fontsize=7 if large else 9)

    # Put MI values in every heatmap cell.
    if large:
        cell_fontsize = max(4, min(6, int(180 / max(nrows, ncols))))
    else:
        cell_fontsize = 8
    for i in range(nrows):
        for j in range(ncols):
            score = matrix.iat[i, j]
            if pd.notna(score):
                ax.text(
                    j,
                    i,
                    f"{score:.2f}",
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="white" if score > .55 else "black",
                )

    ax.set_title(title, pad=12)
    fig.colorbar(image, ax=ax, label="Normalized MI (0–1)", shrink=.75)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="merged_all_stores.csv", help="Location of your merged CSV")
    parser.add_argument("--output-dir", help="Default: mi_all_variables_results beside the CSV")
    parser.add_argument("--train-end", help="Last date of the training set, YYYY-MM-DD")
    parser.add_argument("--all-data", action="store_true", help="Use all dates (exploratory use only)")
    parser.add_argument("--bins", type=int, default=5, help="Quantile categories for numeric columns (default 5)")
    parser.add_argument("--exclude-christmas", action="store_true",
                        help="Exclude December 25 (closed stores in this Brookshire's dataset)")
    args = parser.parse_args()
    if args.bins < 3:
        parser.error("--bins must be at least 3")
    if args.all_data and args.train_end:
        parser.error("Use either --train-end or --all-data, not both")

    path = resolve_input(args.input)
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else path.parent / "mi_all_variables_results"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names in CSV")
    if not {"date", "store_id", "sale_week_type"}.issubset(df.columns):
        raise ValueError("Need date, store_id, sale_week_type columns in the CSV")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some dates could not be parsed")
    if df.duplicated(["store_id", "date"]).any():
        raise ValueError("Duplicate (store_id,date) rows: aggregate department data to one store-day first")
    source_rows, source_columns = df.shape
    if args.exclude_christmas:
        df = df.loc[~((df["date"].dt.month == 12) & (df["date"].dt.day == 25))].copy()
    profit_cols = [c for c in df.columns if c.endswith("_profit")]
    zero_cols = [c for c in df.columns if c.startswith("zero_ratio_")]
    if not profit_cols or not zero_cols:
        raise ValueError("Expected department columns ending _profit and beginning zero_ratio_")
    for col in profit_cols + zero_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[profit_cols + zero_cols].isna().any().any():
        missing = df[profit_cols + zero_cols].isna().sum()
        raise ValueError(f"Missing or nonnumeric department values: {missing[missing > 0].to_dict()}")
    if not df[zero_cols].apply(lambda s: s.between(0, 1).all()).all():
        raise ValueError("The zero_ratio_* columns must be ratios in the range 0–1")

    sale_labels = df["sale_week_type"].astype("string").str.strip().str.lower()
    df["sale_week_type"] = sale_labels.map(SALE_TYPE_NAMES)
    if df["sale_week_type"].isna().any():
        raise ValueError("Unrecognized sale_week_type: expected codes 0/1/2 or promotion names")
    df["store_id"] = df["store_id"].astype("string").str.strip()

    # Raw date is intentionally represented by meaningful calendar features.
    df["year"] = df["date"].dt.year.astype("string")
    df["month"] = df["date"].dt.month.astype("string")
    df["weekday"] = df["date"].dt.dayofweek.astype("string")

    # Two derived summary variables to connect the department-level results.
    df["total_department_profit"] = df[profit_cols].sum(axis=1)
    df["mean_department_zero_ratio"] = df[zero_cols].mean(axis=1)

    if args.all_data:
        used = df.copy()
        split_note = "ALL DATES, EXPLORATORY ONLY — not suitable for feature selection before testing"
    elif args.train_end:
        end = pd.to_datetime(args.train_end)
        used = df.loc[df["date"] <= end].copy()
        split_note = f"Training rows through {end.date()} inclusive"
    else:
        # Brookshire's calendar begins on an ad-week boundary; keep seven-day groups together.
        first_date = df["date"].min()
        week_number = ((df["date"] - first_date).dt.days // 7).astype(int)
        all_weeks = sorted(week_number.unique())
        count = max(1, int(len(all_weeks) * 0.60))
        used = df.loc[week_number.isin(all_weeks[:count])].copy()
        split_note = f"Earliest {count}/{len(all_weeks)} whole 7-day periods (~60% of ad weeks)"
    if len(used) < 10:
        raise ValueError("Too few rows in training subset; check --train-end")

    # All 20 non-date input variables plus date-derived and useful summaries.
    categorical = ["sale_week_type", "store_id", "year", "month", "weekday"]
    continuous = profit_cols + zero_cols + ["total_department_profit", "mean_department_zero_ratio"]
    all_features = categorical + continuous
    if used[all_features].isna().any().any():
        raise ValueError("Missing feature values in selected rows")

    labels = {}
    feature_records = []
    for col in all_features:
        if col in categorical:
            converted = used[col].astype("string")
            strategy = "categorical"
        else:
            converted = categorize_numeric(used[col], args.bins,
                                           keep_zero_separate=(col in zero_cols or col == "mean_department_zero_ratio"))
            strategy = f"zero + positive quantiles" if col in zero_cols or col == "mean_department_zero_ratio" else f"up to {args.bins} quantiles"
        if converted.isna().any():
            raise ValueError(f"Could not discretize {col}")
        labels[col] = converted
        feature_records.append({"feature": col,
                                "origin": "derived" if col in ["year", "month", "weekday", "total_department_profit", "mean_department_zero_ratio"] else "original CSV",
                                "categories": converted.nunique(),
                                "encoding": strategy})

    matrix = pd.DataFrame(np.eye(len(all_features)), index=all_features, columns=all_features)
    pairs = []
    for i, left in enumerate(all_features):
        for j in range(i + 1, len(all_features)):
            right = all_features[j]
            score = float(normalized_mutual_info_score(labels[left], labels[right], average_method="arithmetic"))
            matrix.iat[i, j] = score
            matrix.iat[j, i] = score
            pairs.append((left, right, score))
    matrix.round(5).to_csv(outdir / "mi_all_features_matrix.csv", index_label="variable")
    pd.DataFrame(feature_records).to_csv(outdir / "features_used.csv", index=False)
    make_heatmap(matrix, outdir / "mi_all_features_matrix.png", "All variables: normalized mutual information", large=True)

    pairs_frame = pd.DataFrame(pairs, columns=["variable_1", "variable_2", "normalized_MI"])
    pairs_frame.sort_values("normalized_MI", ascending=False).to_csv(outdir / "mi_all_pairs_ranked.csv", index=False)

    # Matched department names tolerate & and spaces differences.
    def key(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())
    profit_lookup = {key(c[:-len("_profit")]): c for c in profit_cols}
    interesting = []
    for col in zero_cols:
        dept_name = col[len("zero_ratio_"):]
        matched = profit_lookup.get(key(dept_name))
        interesting.append(("sale_week_type", col, f"Promotion type vs. {dept_name} stockouts"))
        if matched:
            interesting.append((col, matched, f"{dept_name} stockouts vs. {dept_name} profit"))
    for col in profit_cols:
        interesting.append(("sale_week_type", col, f"Promotion type vs. {col.replace('_profit', '')} profit"))
    interesting += [
        ("sale_week_type", "mean_department_zero_ratio", "Promotion type vs. average department stockouts"),
        ("mean_department_zero_ratio", "total_department_profit", "Average stockouts vs. total profit"),
        ("sale_week_type", "total_department_profit", "Promotion type vs. total profit"),
    ]
    selected = pd.DataFrame([
        {"variable_1": a, "variable_2": b, "question": description,
         "normalized_MI": round(float(matrix.loc[a, b]), 6)}
        for a, b, description in interesting
    ])
    selected.to_csv(outdir / "mi_hypothesis_relationships.csv", index=False)

    # Direction requires actual mean comparisons; MI values alone cannot show it.
    summary = (used.groupby("sale_week_type", observed=True)
               .agg(store_days=("date", "size"),
                    avg_total_department_profit=("total_department_profit", "mean"),
                    avg_department_zero_ratio=("mean_department_zero_ratio", "mean"))
               .reset_index())
    summary["avg_department_zero_pct"] = summary["avg_department_zero_ratio"] * 100
    summary = summary.drop(columns="avg_department_zero_ratio")
    summary.to_csv(outdir / "comparison_by_sale_type.csv", index=False)

    print(f"Input: {path}\nRows: {source_rows:,} total; {len(used):,} analyzed")
    print(f"Variables: {len(all_features)} in full MI matrix")
    print(f"Date split: {split_note}")
    print("\nSome hypothesis-related MI scores:")
    print(selected.tail(3).to_string(index=False))
    print(f"\nFiles saved to: {outdir}")
    for name in ["mi_all_features_matrix.csv", "mi_all_features_matrix.png",
                 "mi_hypothesis_relationships.csv", "mi_all_pairs_ranked.csv",
                 "features_used.csv", "comparison_by_sale_type.csv"]:
        print(" -", name)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
