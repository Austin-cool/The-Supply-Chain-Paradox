"""Build and query sale-week types keyed by store number.

Source: store_sale_day_table.csv with columns date, store_id, sale_week_type.
Output: sale_types_by_store.json, a mapping of store number -> 730 daily rows.

PowerShell:
    python store_sale_type_lookup.py
    python store_sale_type_lookup.py --store 104
    python store_sale_type_lookup.py --store 104 --all

Python use:
    from store_sale_type_lookup import load_store_tables
    stores = load_store_tables('store_sale_day_table.csv')
    rows = stores[104]   # 730 records, each {'date': ..., 'sale_week_type': ...}

Uses Python's standard library only. Week labels are inferred classifications.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

SOURCE_COLUMNS = ('date', 'store_id', 'sale_week_type')
SALE_TYPES = {'Low-promotion', 'Other ad week', 'Candidate big-sale'}


def load_store_tables(csv_path, expected_days=730):
    """Return {104: [{'date': 'YYYY-MM-DD', 'sale_week_type': '...'}, ...], ...}."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing {csv_path}. Run make_three_column_sale_table.py first.')

    grouped = defaultdict(dict)
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        absent = set(SOURCE_COLUMNS) - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f'Missing columns: {", ".join(sorted(absent))}')
        for line, row in enumerate(reader, start=2):
            day_str = row['date'].strip()
            store_text = row['store_id'].strip()
            sale_type = row['sale_week_type'].strip()
            try:
                day = date.fromisoformat(day_str)
            except ValueError as exc:
                raise ValueError(f'Row {line} has an invalid date: {day_str}') from exc
            if not store_text.isdecimal():
                raise ValueError(f'Row {line} has invalid store ID: {store_text!r}')
            store = int(store_text)
            if sale_type not in SALE_TYPES:
                raise ValueError(f'Row {line} has invalid sale week type: {sale_type!r}')
            if day in grouped[store]:
                raise ValueError(f'Duplicate store/date: {store}, {day_str}')
            grouped[store][day] = sale_type

    if not grouped:
        raise ValueError('No rows found in input CSV')

    result = {}
    common_dates = None
    for store in sorted(grouped):
        dates = sorted(grouped[store])
        if expected_days is not None and len(dates) != expected_days:
            raise ValueError(f'Store {store} has {len(dates)} days, expected {expected_days}')
        if any(b - a != timedelta(days=1) for a, b in zip(dates, dates[1:])):
            raise ValueError(f'Store {store} has missing dates')
        if common_dates is None:
            common_dates = dates
        elif dates != common_dates:
            raise ValueError(f'Store {store} does not have the same dates as the other stores')
        result[store] = [{'date': day.isoformat(), 'sale_week_type': grouped[store][day]}
                         for day in dates]
    return result


def get_store_table(store_id, tables):
    """Fetch the full list of daily rows for a store, e.g., get_store_table(104, tables)."""
    key = int(store_id)
    if key not in tables:
        raise KeyError(f'Store {key} not found. Available stores: {", ".join(map(str, tables))}')
    return tables[key]


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Make a store-number-keyed table of daily sale types')
    parser.add_argument('--input', type=Path, default=base / 'store_sale_day_table.csv',
                        help='Three-column CSV to read')
    parser.add_argument('--output', type=Path, default=base / 'sale_types_by_store.json',
                        help='JSON dictionary to create')
    parser.add_argument('--store', type=int, help='Display the entries for this store')
    parser.add_argument('--all', action='store_true', help='Display all 730 rows for the selected store')
    args = parser.parse_args()

    tables = load_store_tables(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2)
        f.write('\n')
    print(f'Saved: {args.output}')
    print(f'Stores: {len(tables)}. Dates per store: {len(next(iter(tables.values())))}.')
    print('Store IDs:', ', '.join(map(str, tables)))

    if args.store is not None:
        rows = get_store_table(args.store, tables)
        display_rows = rows if args.all else rows[:10]
        print(f'\nStore {args.store} | {len(rows)} daily records')
        print(f'{"date":<14} sale_week_type')
        print('-' * 38)
        for row in display_rows:
            print(f'{row["date"]:<14} {row["sale_week_type"]}')
        if len(display_rows) < len(rows):
            print(f'... showing first {len(display_rows)} of {len(rows)}; use --all to show all')


if __name__ == '__main__':
    main()
