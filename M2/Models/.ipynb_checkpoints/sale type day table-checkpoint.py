r"""Create one 3-column store-by-day promotion lookup for Brookshire's.

Output columns: date,store_id,sale_week_type
Each store appears once for every calendar date (including Christmas closures).

Inputs needed next to this script:
  - calendar.csv
  - stores.csv
  - brookshires_q4_q5_results/inferred_promotion_weeks.csv

Alternatively, put inferred_promotion_weeks.csv alongside the script, or supply
its location with --classification. The week types were inferred previously
from prices; this script reuses those classifications rather than guessing.

Windows PowerShell:
    python .\make_three_column_sale_table.py
    python .\make_three_column_sale_table.py --classification .\inferred_promotion_weeks.csv

Uses only Python's standard library; does not modify source files.
"""

import argparse
import csv
from datetime import date
from pathlib import Path


def load_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {', '.join(sorted(missing))}")
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Make one CSV containing date, store ID, and sale week type.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder holding calendar.csv and stores.csv (default: script folder)")
    parser.add_argument("--classification", type=Path,
                        help="Location of inferred_promotion_weeks.csv (if not in the default location)")
    parser.add_argument("--output", type=Path,
                        help="Output CSV path (default: DATA_DIR/store_sale_day_table.csv)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    candidates = [
        data_dir / "brookshires_q4_q5_results" / "inferred_promotion_weeks.csv",
        data_dir / "inferred_promotion_weeks.csv",
    ]
    if args.classification is not None:
        classification_path = args.classification.resolve()
    else:
        classification_path = next((p for p in candidates if p.is_file()), candidates[0])

    if not classification_path.is_file():
        raise FileNotFoundError(
            "Missing inferred_promotion_weeks.csv. Place it in either:\n"
            f"  {candidates[0]}\n  {candidates[1]}\n"
            "or run with --classification PATH_TO_FILE."
        )

    calendar = load_csv(data_dir / "calendar.csv", ["date", "ad_week"])
    stores = load_csv(data_dir / "stores.csv", ["store_id", "ad_zone"])
    weeks = load_csv(classification_path, ["ad_zone", "ad_week", "week_type"])

    # A week's promotion classification is specific to its advertising zone.
    sale_type_by_zone_week = {}
    for row in weeks:
        key = (row["ad_zone"].strip(), row["ad_week"].strip())
        week_type = row["week_type"].strip()
        if key in sale_type_by_zone_week:
            raise ValueError(f"Duplicate promotion classification for {key}")
        if week_type not in ("Low-promotion", "Other ad week", "Candidate big-sale"):
            raise ValueError(f"Unexpected week_type {week_type!r} for {key}")
        sale_type_by_zone_week[key] = week_type

    if len({r["date"].strip() for r in calendar}) != len(calendar):
        raise ValueError("calendar.csv contains duplicate dates")
    if len({r["store_id"].strip() for r in stores}) != len(stores):
        raise ValueError("stores.csv contains duplicate store IDs")

    output_path = (args.output or data_dir / "store_sale_day_table.csv").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "store_id", "sale_week_type"])
        for day in sorted(calendar, key=lambda d: d["date"]):
            day_date = day["date"].strip()
            date.fromisoformat(day_date)  # Verify YYYY-MM-DD dates.
            ad_week = day["ad_week"].strip()
            for store in sorted(stores, key=lambda s: int(s["store_id"])):
                key = (store["ad_zone"].strip(), ad_week)
                if key not in sale_type_by_zone_week:
                    raise ValueError(f"No classification for zone {key[0]!r} in week {key[1]!r}")
                writer.writerow([day_date, store["store_id"].strip(), sale_type_by_zone_week[key]])
                row_count += 1

    print(f"Created: {output_path}")
    print(f"Columns: date, store_id, sale_week_type")
    print(f"Rows: {row_count:,} ({len(stores)} stores x {len(calendar)} dates)")
    print("Merge with teammates' results using date + store_id.")
    print("Note: sale_week_type is inferred from prices; it is not an official promotion label.")


if __name__ == "__main__":
    main()
