"""Compare promotion types, daily stockouts and profit within Brookshire's stores.

Reads one store-day CSV (date, store_id, sale_week_type, department *_profit,
zero_ratio_*). Each row must represent ONE store/date. No external sales or
inventory files are required.

Usage, from the directory containing this script and merged_all_stores.csv:
    python compare_stores_stockouts_and_profit.py

From another directory:
    python compare_stores_stockouts_and_profit.py --input "path/to/merged_all_stores.csv"

Optional: --train-end 2025-03-14 to restrict analysis to a training subset.

These are DESCRIPTIVE comparisons, not causal experiments or model validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

SALE_NAMES = {
    "0": "Low-promotion", "0.0": "Low-promotion", "low-promotion": "Low-promotion",
    "1": "Other ad week", "1.0": "Other ad week", "other ad week": "Other ad week",
    "2": "Candidate big-sale", "2.0": "Candidate big-sale",
    "candidate big-sale": "Candidate big-sale", "candidate big sale": "Candidate big-sale",
}
ORDER = ["Low-promotion", "Other ad week", "Candidate big-sale"]


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    here = Path(__file__).resolve().parent
    guesses = [p, here / p, here / p.name, here / "results" / p.name,
               Path.cwd() / "M2" / "Models" / "results" / p.name,
               here / "M2" / "Models" / "results" / p.name]
    for guess in guesses:
        if guess.is_file():
            return guess.resolve()
    raise FileNotFoundError(
        f"Cannot locate {raw!r}. Put it beside this script or provide --input with the CSV's full path."
    )


def save_heatmap(frame: pd.DataFrame, target: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; exported comparison CSVs without chart.")
        return
    pivot = frame.pivot(index="store_id", columns="sale_week_type", values="avg_zero_stock_pct")
    pivot = pivot.reindex(columns=ORDER)
    fig, ax = plt.subplots(figsize=(9, 6))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=18, ha="right")
    ax.set_yticks(range(len(pivot.index)), [f"Store {x}" for x in pivot.index])
    ax.set_title("Average opening zero-stock rate by store and promotion type")
    ax.set_xlabel("Promotion type")
    ax.set_ylabel("Store")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}%", ha="center", va="center", fontsize=9,
                        color="white" if data[i, j] > (np.nanmax(data) * .6) else "black")
    fig.colorbar(im, ax=ax, label="Mean department zero-stock ratio (%)")
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)


def save_profit_difference_chart(frame: pd.DataFrame, target: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ordered = frame.set_index("sale_week_type").reindex(ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    y = ordered["equal_stratum_mean_profit_difference_high_minus_low"].to_numpy(dtype=float)
    bars = ax.bar(ordered["sale_week_type"], y)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("High-stockout minus low-stockout profit ($ / store-day)")
    ax.set_title("Profit difference within the same store and promotion type")
    for bar, val in zip(bars, y):
        if np.isfinite(val):
            ax.annotate(f"${val:,.0f}", (bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 4 if val >= 0 else -15), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="merged_all_stores.csv", help="Merged CSV path")
    parser.add_argument("--output-dir", default=None, help="Where to save the output; default: comparison_results beside input CSV")
    parser.add_argument("--train-end", default=None, help="Optional final date to analyze (YYYY-MM-DD); excludes later dates")
    parser.add_argument("--include-christmas", action="store_true", help="Keep Dec 25 closed days (excluded by default)")
    args = parser.parse_args()

    path = resolve_path(args.input)
    out = Path(args.output_dir).expanduser() if args.output_dir else path.parent / "comparison_results"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"store_id", "date", "sale_week_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    profit_cols = [c for c in df if c.endswith("_profit")]
    zero_cols = [c for c in df if c.startswith("zero_ratio_")]
    if not profit_cols or not zero_cols:
        raise ValueError("Expected department profit columns (*_profit) and stockout columns (zero_ratio_*)")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some dates cannot be parsed")
    if df.duplicated(["store_id", "date"]).any():
        raise ValueError("CSV has duplicate store_id/date rows; one store-day must appear only once")
    for col in profit_cols + zero_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[profit_cols + zero_cols].isna().any().any():
        raise ValueError("Some department profit/stockout fields are missing or nonnumeric")
    if not df[zero_cols].apply(lambda c: c.between(0, 1).all()).all():
        raise ValueError("Expected zero_ratio_* values between 0 and 1")

    df["sale_week_type"] = df["sale_week_type"].astype("string").str.strip().str.lower().map(SALE_NAMES)
    if df["sale_week_type"].isna().any():
        raise ValueError("Unknown sale_week_type: expected 0, 1, 2 or the promotion names")

    all_rows = len(df)
    closed_rows = ((df["date"].dt.month == 12) & (df["date"].dt.day == 25)).sum()
    if not args.include_christmas:
        df = df.loc[~((df["date"].dt.month == 12) & (df["date"].dt.day == 25))].copy()
    if args.train_end:
        cutoff = pd.to_datetime(args.train_end)
        df = df.loc[df["date"] <= cutoff].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering")

    # NOTE: the rate is an UNWEIGHTED average of nine department ratios,
    # not the exact store-wide proportion of stockouts.
    df["average_department_zero_stock_pct"] = 100 * df[zero_cols].mean(axis=1)
    # This is a sum of columns labeled 'profit', not a margin percentage.
    df["total_department_profit_dollars"] = df[profit_cols].sum(axis=1)

    # Comparison 1: SAME store compared across three promotion week categories.
    by_store_type = (df.groupby(["store_id", "sale_week_type"], observed=True)
        .agg(days=("date", "size"),
             avg_zero_stock_pct=("average_department_zero_stock_pct", "mean"),
             median_zero_stock_pct=("average_department_zero_stock_pct", "median"),
             avg_profit_dollars=("total_department_profit_dollars", "mean"),
             median_profit_dollars=("total_department_profit_dollars", "median"))
        .reset_index().sort_values(["store_id", "sale_week_type"]))
    by_store_type.to_csv(out / "comparison_1_each_store_by_sale_type.csv", index=False)

    low = (by_store_type.loc[by_store_type["sale_week_type"] == "Low-promotion"]
           .set_index("store_id"))
    big = (by_store_type.loc[by_store_type["sale_week_type"] == "Candidate big-sale"]
           .set_index("store_id"))
    common = low.index.intersection(big.index)
    big_vs_low = pd.DataFrame({
        "store_id": common,
        "low_promotion_days": low.loc[common, "days"].values,
        "big_sale_days": big.loc[common, "days"].values,
        "low_promotion_avg_zero_stock_pct": low.loc[common, "avg_zero_stock_pct"].values,
        "big_sale_avg_zero_stock_pct": big.loc[common, "avg_zero_stock_pct"].values,
        "difference_zero_stock_percentage_points_big_minus_low": (
            big.loc[common, "avg_zero_stock_pct"].values - low.loc[common, "avg_zero_stock_pct"].values),
        "low_promotion_avg_profit_dollars": low.loc[common, "avg_profit_dollars"].values,
        "big_sale_avg_profit_dollars": big.loc[common, "avg_profit_dollars"].values,
        "difference_profit_dollars_big_minus_low": (
            big.loc[common, "avg_profit_dollars"].values - low.loc[common, "avg_profit_dollars"].values),
    })
    big_vs_low.to_csv(out / "comparison_1_big_sale_minus_low_by_store.csv", index=False)
    save_heatmap(by_store_type, out / "comparison_1_stockout_heatmap.png")

    # Comparison 2: within EACH (store, promotion type), separate higher from lower
    # stockout days based on the median for that SAME store/promotion category.
    # Low: <= median; high: > median. This handles the many zero-stockout days
    # without pretending that percentile cutoffs produce nonzero groups.
    summaries = []
    for (store, sale), piece in df.groupby(["store_id", "sale_week_type"], observed=True):
        mid = float(piece["average_department_zero_stock_pct"].median())
        low_days = piece[piece["average_department_zero_stock_pct"] <= mid]
        high_days = piece[piece["average_department_zero_stock_pct"] > mid]
        summaries.append({
            "store_id": store, "sale_week_type": sale,
            "median_cutoff_zero_stock_pct": mid,
            "low_stockout_days": len(low_days), "high_stockout_days": len(high_days),
            "low_mean_zero_stock_pct": low_days["average_department_zero_stock_pct"].mean(),
            "high_mean_zero_stock_pct": high_days["average_department_zero_stock_pct"].mean(),
            "low_mean_profit_dollars": low_days["total_department_profit_dollars"].mean(),
            "high_mean_profit_dollars": high_days["total_department_profit_dollars"].mean(),
            "profit_difference_high_minus_low_dollars": (
                high_days["total_department_profit_dollars"].mean() -
                low_days["total_department_profit_dollars"].mean()) if len(high_days) else np.nan,
        })
    within = pd.DataFrame(summaries).sort_values(["store_id", "sale_week_type"])
    within.to_csv(out / "comparison_2_high_vs_low_stockout_by_store_sale.csv", index=False)

    # Equal weight per eligible store×promotion stratum, rather than allowing
    # a large store group to dominate. Omit strata with no high-stockout days.
    eligible = within[(within["low_stockout_days"] > 0) & (within["high_stockout_days"] > 0)]
    overall = (eligible.groupby("sale_week_type", observed=True)
        .agg(store_groups_compared=("store_id", "nunique"),
             equal_stratum_mean_profit_low=("low_mean_profit_dollars", "mean"),
             equal_stratum_mean_profit_high=("high_mean_profit_dollars", "mean"),
             equal_stratum_mean_profit_difference_high_minus_low=("profit_difference_high_minus_low_dollars", "mean"),
             equal_stratum_mean_zero_stock_low_pct=("low_mean_zero_stock_pct", "mean"),
             equal_stratum_mean_zero_stock_high_pct=("high_mean_zero_stock_pct", "mean"))
        .reset_index())
    overall.to_csv(out / "comparison_2_summary_by_sale_type.csv", index=False)
    if not overall.empty:
        save_profit_difference_chart(overall, out / "comparison_2_profit_differences.png")

    # Save daily calculation inputs for inspection/teammates, not prediction features.
    df[["date", "store_id", "sale_week_type", "average_department_zero_stock_pct",
        "total_department_profit_dollars"]].sort_values(["store_id", "date"]).to_csv(
            out / "daily_store_features_for_comparison.csv", index=False)

    print(f"Read {all_rows:,} rows; analyzed {len(df):,} (excluded Christmas by default).")
    print(f"Stores: {df.store_id.nunique()}, department profit columns: {len(profit_cols)}, stockout columns: {len(zero_cols)}")
    print(f"\nComparison 1: {len(big_vs_low)} stores have both low-promotion and candidate big-sale periods.")
    if len(big_vs_low):
        print("Mean within-store big-sale minus low-promotion stockout rate:",
              round(big_vs_low["difference_zero_stock_percentage_points_big_minus_low"].mean(), 3), "percentage points")
    print("\nComparison 2: high-stockout minus low-stockout mean profit within store & sale type:")
    print(overall[["sale_week_type", "store_groups_compared", "equal_stratum_mean_profit_difference_high_minus_low"]].to_string(index=False))
    print("\nResults folder:", out.resolve())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
