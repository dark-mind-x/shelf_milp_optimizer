import pandas as pd
from pathlib import Path

# Path to the expanded 50-product dataset
excel_path = Path("data/Second_Paper_MILP_Dataset_Full.xlsx")

# Load data
products = pd.read_excel(excel_path, sheet_name="Products", engine="openpyxl")
shelves = pd.read_excel(excel_path, sheet_name="Shelf_Rack_Locations", engine="openpyxl")

# 1. Relax the minimum width constraints so all 50 items physically fit
products["Min_Facing"] = 1

# 2. Fix the broken products (P039-P046)
# They require 60-65cm height but are labeled 'Folded' (folded shelves only have 28cm).
# They need to be Hanging to go on the Lower racks which have 75cm clearance.
problem_items = ["P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046"]
products.loc[products["Product_ID"].isin(problem_items), "Display_Mode"] = "Hanging"

# 3. Recalculate the Feasibility Matrix (Placement_Options)
rows = []
for _, p in products.iterrows():
    for _, s in shelves.iterrows():
        # Using the same 4 physical logic rules
        mode_f   = 1 if p["Display_Mode"] == s["Zone_Type"] else 0
        heavy_f  = 0 if (p["Bulky_Item_0_1"] == 1 and s["Shelf_Level"] == "Top") else 1
        depth_f  = 1 if p["Facing_Depth_cm"] <= s["Depth_cm"] else 0
        height_f = 1 if p["Max_Stack_Height_cm"] <= s["Height_Clearance_cm"] else 0
        
        overall_f = 1 if (mode_f and heavy_f and depth_f and height_f) else 0
        
        rows.append({
            "Product_ID": p["Product_ID"],
            "Location_ID": s["Location_ID"],
            "Mode_Feasible_0_1": mode_f,
            "Heavy_Feasible_0_1": heavy_f,
            "Depth_Feasible_0_1": depth_f,
            "Height_Feasible_0_1": height_f,
            "Overall_Feasible_0_1": overall_f,
            "Reason": "feasible" if overall_f == 1 else "infeasible"
        })

full_feas_df = pd.DataFrame(rows)

# Save everything back to the file
xls = pd.ExcelFile(excel_path, engine="openpyxl")
sheet_dict = {}

for sheet_name in xls.sheet_names:
    if sheet_name == "Products":
        sheet_dict[sheet_name] = products
    elif sheet_name == "Placement_Options":
        sheet_dict[sheet_name] = full_feas_df
    else:
        sheet_dict[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    for sheet_name, df in sheet_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Dataset patched successfully! You can now run your Streamlit UI.")
