# To run: make sure your terminal is in the "The-Supply-Chain-Paradox". Type this command: python M2/Models/combine_and_evaluate.py 


import profit_per_day_per_department
import stock_per_day_per_department
import pandas as pd
from pathlib import Path

store_table = pd.read_csv("M2/Models/store_sale_day_table.csv")
store_table["date"] = profit_per_day_per_department.parse_dates(store_table["date"])
store_table.groupby(["date", "store_id"], observed=True)["sale_week_type"]
store_table["sale_week_type"] = store_table["sale_week_type"].map({"Low-promotion": 0, "Other ad week": 1, "Candidate big-sale": 2})

merged_dict = {}

for store in stock_per_day_per_department.store_ids:
    st = store_table[store_table["store_id"] == store].drop(columns="store_id")
    zr = stock_per_day_per_department.zero_ratio_per_store[store].copy()
    zr["date"] = profit_per_day_per_department.parse_dates(zr["date"])

    merged = (
        profit_per_day_per_department.store_dict[store]
        .reset_index()                      # turn the date index into a column
        .merge(zr, on="date", how="left")
        .merge(st, on="date", how="left")
    )
    merged_dict[store] = merged

for store in stock_per_day_per_department.store_ids:
    print(merged_dict[store])

    # Create an output folder
output_folder = Path("M2/Models/results")
output_folder.mkdir(parents=True, exist_ok=True)

# Combine all store DataFrames into one table
all_stores = pd.concat(
    [
        df.assign(store_id=store)
        for store, df in merged_dict.items()
    ],
    ignore_index=True
)

# Save all data to one CSV
all_stores.to_csv(
    output_folder / "merged_all_stores.csv",
    index=False
)

print("Successfully saved all store data!")
print(f"Total rows: {len(all_stores)}")