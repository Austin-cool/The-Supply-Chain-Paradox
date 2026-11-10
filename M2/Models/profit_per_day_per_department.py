# Important: When you run this file, make sure you run it from the The-Supply-Chain-Paradox directory


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










# profit per day, per store, one column per department
store_dept_daily = (
    merged.groupby(["store_id", "date", "department"], observed=True)["profit"]
    .sum()
    .unstack("department")
)

stores = sorted(merged["store_id"].unique())
print(f"Plotting {len(stores)} stores: {stores}")


store_dict = {}

for store in stores:
    df = store_dept_daily.loc[store].add_suffix("_profit")          # index = date, columns = departments
    store_dict[store] = df


    ### this is just to plot each table
    # smooth = df.rolling(7, center=True).mean()

    # ax = smooth.plot(figsize=(13, 6), title=f"Store {store}: profit per day by department (7-day average)")
    # # smooth.sum(axis=1, min_count=1).plot(ax=ax, color="black", linewidth=2.5, label="Store total")

    # ax.set_ylabel("Profit ($)")
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    # plt.show()

















