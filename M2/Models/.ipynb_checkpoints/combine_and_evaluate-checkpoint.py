# To run: make sure your terminal is in the "The-Supply-Chain-Paradox". Type this command: python M2/Models/combine_and_evaluate.py 

import matplotlib.pyplot as plt
import pandas as pd
import profit_per_day_per_department
import stock_per_day_per_department

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
    df = merged_dict[store].copy()
    df["day_number"] = (df["date"] - df["date"].min()).dt.days   # lets date take part in the correlation
    corr = df.drop(columns="date").corr()                        # 20 x 20

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=ax)
    ax.set_title(f"Store {store}: correlation matrix")
    plt.tight_layout()
    plt.show()

# --- Save as SVG ---
    fig.savefig(
        f"Profit vs sale type per department in {store}.svg",
        bbox_inches='tight'
    )
    
    plt.close(fig)