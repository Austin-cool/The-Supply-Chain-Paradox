# Obtain necessary datasets
import pandas as pd

Calendar = pd.read_csv('calendar.csv')
Inventory = pd.read_csv('inventory.csv.gz')
Products = pd.read_csv('products.csv')
Stores = pd.read_csv('stores.csv')

# Create a list with the item ids
item_ids = [i for i in Products['item_id']]

# Create a list with the store ids\
store_ids = [i for i in Stores['store_id']]

# Create a list with the available dates
dates = [i for i in Calendar['date']]

# Create a list with the available departments
Departments = list({i.replace(" ", "") for i in Products['department']})

# Create a dictionary where the keys are the item ids and the values are the corresponding departments
item_department_dict = {row['item_id']: row['department'].replace(" ", "") for index, row in Products.iterrows()}

# Create a list of items for each department
department_items_dict = {department: [item for item in item_ids if item_department_dict[item] == department] for department in Departments}


# Create a new dataset that combines that adds the departments feature to the inventory dataset
Inventory_with_departments = Inventory.copy()
Inventory_with_departments['department'] = Inventory_with_departments['item_id'].map(item_department_dict)
# Inventory_with_departments

features_per_store_per_department = {}

for store in store_ids:
    features_per_store_per_department[store] = {}
    
    for department in Departments:
        Inventory_with_departments_per_store_per_department = Inventory_with_departments.loc[
            (Inventory_with_departments["store_id"] == store) &
            (Inventory_with_departments["department"] == department)
        ]
        
        New_table = (
            Inventory_with_departments_per_store_per_department.groupby("date", as_index=False)
              .agg(
                  total_on_hand=("on_hand", "sum"),
                  num_zero=("on_hand", lambda s: (s == 0).sum()),
                  num_items=("on_hand", "size"),
              )
        )
        
        New_table["zero_ratio_" + department] = New_table["num_zero"] / New_table["num_items"]
        
        # Keep only the requested columns
        # New_table = New_table[["date", "total_on_hand", "num_zero", "zero_ratio"]]
        New_table = New_table[["date", "zero_ratio_" + department]]

        features_per_store_per_department[store][department] = New_table

zero_ratio_per_store = {}

for store in store_ids:
    df1 = features_per_store_per_department[store]['Beverages']
    df2 = features_per_store_per_department[store]['Grocery']
    df3 = features_per_store_per_department[store]['Meat&Seafood']
    df4 = features_per_store_per_department[store]['Health&Beauty']
    df5 = features_per_store_per_department[store]['Produce']
    df6 = features_per_store_per_department[store]['Frozen']
    df7 = features_per_store_per_department[store]['Bakery']
    df8 = features_per_store_per_department[store]['Household']
    df9 = features_per_store_per_department[store]['Dairy']

    zero_rate_per_department = pd.DataFrame({
    'date': df1["date"],
    'zero_ratio_Beverages': df1["zero_ratio_Beverages"],
    'zero_ratio_Grocery': df2["zero_ratio_Grocery"],
    'zero_ratio_Meat&Seafood': df3["zero_ratio_Meat&Seafood"],
    'zero_ratio_Health&Beauty': df4["zero_ratio_Health&Beauty"],
    'zero_ratio_Produce': df5["zero_ratio_Produce"],
    'zero_ratio_Frozen': df6["zero_ratio_Frozen"],
    'zero_ratio_Bakery': df7["zero_ratio_Bakery"],
    'zero_ratio_Household': df8["zero_ratio_Household"],
    'zero_ratio_Dairy': df9["zero_ratio_Dairy"],
    })

    zero_ratio_per_store[store] = zero_rate_per_department

zero_ratio_per_store[311]