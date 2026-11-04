import pandas as pd

# We will not need to work with all of the data, so not all of the data needs to be cleaned. 

prcs = pd.read_csv("Project data/prices.csv")
prod = pd.read_csv("Project data/products.csv")



# stores 131 and 214 report their prices in cents. It needs to be converted. 
# This will not change the number of rows, so it is not printed
prcs.loc[prcs["store_id"].isin([131, 214]), "sell_price"] /= 100
prcs["sell_price"] = prcs["sell_price"].round(2)



# see which items have no unit_cost, then drop them so profit and revenue cover the same rows
print(f"# of rows in products.csv before cleaning: {len(prod['unit_cost'])}")
missing = prod.loc[prod["unit_cost"].isna(), "item_id"].unique()
prod = prod.dropna(subset=["unit_cost"]).copy()
print(f"# of rows in products.csv after cleaning: {len(prod["unit_cost"])}")