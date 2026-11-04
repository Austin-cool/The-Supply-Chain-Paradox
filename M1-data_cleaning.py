import pandas as pd

# We will not need to work with all of the data, so not all of the data needs to be cleaned. 

prcs = pd.read_csv("prices.csv")
prod = pd.read_csv("products.csv")



# stores 131 and 214 report their prices in cents. It needs to be converted
prcs.loc[prcs["store_id"].isin([131, 214]), "sell_price"] /= 100
prcs["sell_price"] = prcs["sell_price"].round(2)



# see which items have no cost, then drop them so profit and revenue cover the same rows
missing = prod.loc[prod["unit_cost"].isna(), "item_id"].unique()
print(len(missing), "items missing cost:", missing)
prod = prod.dropna(subset=["unit_cost"]).copy()
