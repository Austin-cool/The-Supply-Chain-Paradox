
# Description: This program will do a few things:
# 1. Access the data to give statistics on the profit margins of each zone during big sale events
# 2. Access the data to give statistics on the profit margins of each store's products when the product is and is not on sale

# disclaimer: make sure you are in the "The-Supply-Chain-Paradox" directory when you run this file

import pandas as pd


files = [
    "Project data/calendar.csv",   # events
    "Project data/products.csv",   # unit cost and item id
    "Project data/prices.csv",     # og sell price and marked-down sell price
    "Project data/stores.csv"      # zones
]

# open files
cldr = pd.read_csv(files[0])
prod = pd.read_csv(files[1])
prcs = pd.read_csv(files[2])
strs = pd.read_csv(files[3])

### get data

# stores 131 and 214 report their prices in cents. It needs to be converted
prcs.loc[prcs["store_id"].isin([131, 214]), "sell_price"] /= 100
prcs["sell_price"] = prcs["sell_price"].round(2)


event_weeks = cldr.loc[cldr["event_name_1"].notna(), "ad_week"].unique()

# merge cost and zone into prices
df = (
    prcs.merge(prod[["item_id", "unit_cost"]], on="item_id", how="left")
    .merge(strs[["store_id", "store_name", "ad_zone"]], on="store_id", how="left")
)

# see which items have no cost, then drop them so profit and revenue cover the same rows
missing = df.loc[df["unit_cost"].isna(), "item_id"].unique()
print(len(missing), "items missing cost:", missing)
df = df.dropna(subset=["unit_cost"]).copy()

keys = ["store_id", "item_id"]
df["reg_price"] = df.groupby(keys)["sell_price"].transform("max")
df["on_sale"]   = df["sell_price"] < df["reg_price"]
df["is_event"]  = df["ad_week"].isin(event_weeks)

df["profit"] = df["sell_price"] - df["unit_cost"]
df["margin"] = df["profit"] / df["sell_price"]

def summarize(g):
    return pd.Series({
        "profit": g["profit"].sum(),
        "revenue": g["sell_price"].sum(),
        "profit_margin_pct (total_profit/total_revenue)": 100 * g["profit"].sum() / g["sell_price"].sum(),
        "avg_margin_pct (mean of each row's margin)": 100 * g["margin"].mean(),
    })

pd.set_option("display.width", 200)

# Q2: lowest-margin zone during event weeks
q1 = (df[df["is_event"]].groupby("ad_zone")
        .apply(summarize, include_groups=False)
        .sort_values("profit_margin_pct (total_profit/total_revenue)"))
print(q1.to_string())
print("Lowest-margin zone:", q1.index[0])

# Q3: on sale vs not, per store and overall
q2_store = df.groupby(["store_id", "store_name", "on_sale"]).apply(summarize, include_groups=False)
q2_all = df.groupby("on_sale").apply(summarize, include_groups=False)
print(q2_store.to_string())
print(q2_all.to_string())