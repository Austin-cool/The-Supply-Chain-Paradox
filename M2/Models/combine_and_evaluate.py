import profit_per_day_per_department
import stock_per_day_per_department
import pandas as pd

store_table = pd.read_csv("M2/Models/store_sale_day_table.csv")
store_table["date"] = profit_per_day_per_department.parse_dates(store_table["date"])
store_table.groupby(["date", "store_id"], observed=True)["sale_week_type"]

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