# Important: When you run this file, make sure you run it from the The-Supply-Chain-Paradox


import pandas as pd
import matplotlib.pyplot as plt


prod  = pd.read_csv("Project data/products.csv", usecols=["item_id", "department", "unit_cost"])
sales = pd.read_csv("Project data/sales_train.csv.gz", dtype={"store_id": "int16", "item_id": "int32", "unit_sales": "int32"})
prcs  = pd.read_csv("Project data/prices.csv", dtype={"store_id": "int16", "item_id": "int32", "ad_week": "int32"})
clndr = pd.read_csv("Project data/calendar.csv", usecols=["date", "ad_week"])

# take out trailing spaces from products.csv
prod["department"] = prod["department"].str.strip()

# Some sales have outlandishly high values, so they need to be taken out
SENTINELS = [99999, 12000, 9999, 4800, 999]

bad = sales["unit_sales"].isin(SENTINELS)
print(f"Removing {bad.sum()} sentinel rows")   # expect 18
sales = sales[~bad].copy()



# fix date formatting
def parse_dates(s):
    s = s.str.strip()
    uniq = pd.Series(s.unique())
    iso = pd.to_datetime(uniq, format="%Y-%m-%d", errors="coerce")
    us  = pd.to_datetime(uniq, format="%m/%d/%Y", errors="coerce")
    mapping = dict(zip(uniq, iso.fillna(us)))
    out = s.map(mapping)
    assert out.notna().all(), f"unparsed: {s[out.isna()].unique()[:10]}"
    return out

sales["date"] = parse_dates(sales["date"])
clndr["date"] = parse_dates(clndr["date"])

# store 131 and 214 report their prices in cents, so that needs to be converted
prcs.loc[prcs["store_id"].isin([131, 214]), "sell_price"] /= 100





# merge the data
print("begining merge")

merged = (
    sales
    .merge(clndr, on="date", how="left", validate="m:1")
    .merge(prod, on="item_id", how="left", validate="m:1")
    .merge(prcs, on=["store_id", "item_id", "ad_week"], how="left", validate="m:1")
    [["date", "ad_week", "store_id", "item_id",
      "department", "unit_cost", "unit_sales", "sell_price"]]
)

# take out all rows without a unit price
rows_before = len(merged)
merged = merged.dropna(subset=["unit_cost"]).copy()
print(f"Dropped {rows_before - len(merged)} sales rows with no unit_cost")

print("finished merging")

merged["profit"] = (merged["sell_price"] - merged["unit_cost"]) * merged["unit_sales"]

profit_by_dept_day = merged.groupby(["date", "department"], observed=True)["profit"].sum().reset_index()

# print(profit_by_dept_day.head(50))

# # 1. overall count of missing values
# n_missing = merged["sell_price"].isna().sum()
# print(f"Missing sell_price: {n_missing} of {len(merged)} rows ({n_missing / len(merged):.4%})")

# # 2. also check the source file, in case the blanks are already in prices.csv
# print(f"Missing in prices.csv itself: {prcs['sell_price'].isna().sum()}")
# print(f"Zero or negative prices in prices.csv: {(prcs['sell_price'] <= 0).sum()}")



daily = merged.groupby("date")["profit"].sum()
daily.plot(figsize=(12, 5), title="Total profit per day")
plt.ylabel("Profit ($)")
plt.show()


# profit per day, one column per department
dept_daily = (
    merged.groupby(["date", "department"], observed=True)["profit"]
    .sum()
    .unstack("department")
)

# 1. all departments on one chart (7-day average so the lines are readable)
dept_daily.rolling(7, center=True).mean().plot(figsize=(13, 6), title="Profit per day by department (7-day average)")
plt.ylabel("Profit ($)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# compare each store's changes in profit
# def compare(day, by, k=(-3, -2, -1, 1, 2, 3)):
#     day = pd.Timestamp(day)
#     base_days = [day + pd.Timedelta(days=7 * i) for i in k]
#     on   = merged[merged["date"] == day].groupby(by, observed=True)["profit"].sum()
#     base = merged[merged["date"].isin(base_days)].groupby(by, observed=True)["profit"].sum() / len(base_days)
#     out = pd.DataFrame({"that_day": on, "typical": base})
#     out["diff"]  = out["that_day"] - out["typical"]
#     out["ratio"] = out["that_day"] / out["typical"]
#     return out.sort_values("diff")

# for d in ["2025-03-08", "2025-07-06", "2024-11-28", "2025-11-27"]:
#     print(f"\n=== {d} ===")
#     print(compare(d, "store_id"))
#     print(compare(d, "department"))
