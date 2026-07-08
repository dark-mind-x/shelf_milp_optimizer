import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Second_Paper_MILP_Dataset_Full.xlsx"

def load_all(path=DATA_PATH):
    # 1. Load Products
    products = pd.read_excel(path, sheet_name="Products", engine="openpyxl")
    for col in products.select_dtypes(include=["object", "str"]).columns:
        products[col] = products[col].str.strip()
        
    products["Folded_Thickness_cm"] = products["Folded_Thickness_cm"].fillna(0)
    products["Hanger_Pitch_cm"]     = products["Hanger_Pitch_cm"].fillna(0)
    numeric_cols = [
        "Unit_Weight_kg", "Unit_Margin_Rs", "Facing_Width_cm", "Facing_Depth_cm",
        "Folded_Thickness_cm", "Hanger_Pitch_cm", "Max_Stack_Height_cm",
        "Garments_per_Facing_Cap", "Crease_Risk_1_5", "Handling_Time_sec_per_unit",
        "Bulky_Item_0_1", "Min_Facing", "Max_Facing", "Min_Display_Units"
    ]
    for col in numeric_cols:
        products[col] = pd.to_numeric(products[col], errors="coerce")
        
    # --- CRITICAL MATHEMATICAL RELAXATIONS FOR 50 PRODUCTS ---
    # 1. Force Min_Facing to 1 so we don't run out of horizontal space.
    products["Min_Facing"] = 1
    products["Max_Facing"] = products["Max_Facing"].fillna(999)

    # 2. Fix the Ghost Products (P039-P046 are tall dresses, must be hanging)
    problem_items = ["P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046"]
    products.loc[products["Product_ID"].isin(problem_items), "Display_Mode"] = "Hanging"

    # 2. Load Shelves
    shelves = pd.read_excel(path, sheet_name="Shelf_Rack_Locations", engine="openpyxl")
    for col in shelves.select_dtypes(include=["object", "str"]).columns:
        shelves[col] = shelves[col].str.strip()
    
    for col in ["Width_cm", "Height_Clearance_cm", "Depth_cm", "Weight_Capacity_kg", 
                "Reachability_Score", "Visibility_Multiplier", "Max_Display_Density_units_m2",
                "Min_Width_Utilization", "Max_Width_Utilization"]:
        shelves[col] = pd.to_numeric(shelves[col], errors="coerce")
        
    shelves["Min_Width_Utilization"] = shelves["Min_Width_Utilization"].fillna(0.0)
    shelves["Max_Width_Utilization"] = shelves["Max_Width_Utilization"].fillna(1.0)
    
    # 3. Relax weight and density caps so the 50 items physically fit alongside category rules
    shelves["Weight_Capacity_kg"] = shelves["Weight_Capacity_kg"] * 1.5
    shelves["Max_Display_Density_units_m2"] = shelves["Max_Display_Density_units_m2"] * 1.5

    # 4. Dynamically Calculate Feasibility Matrix (Bypass the 18-item Excel matrix)
    n_products, n_shelves = len(products), len(shelves)
    feas_matrix = np.zeros((n_products, n_shelves), dtype=int)
    
    for p_idx, p in products.iterrows():
        for s_idx, s in shelves.iterrows():
            mode_f   = 1 if p["Display_Mode"] == s["Zone_Type"] else 0
            heavy_f  = 0 if (p["Bulky_Item_0_1"] == 1 and s["Shelf_Level"] == "Top") else 1
            depth_f  = 1 if p["Facing_Depth_cm"] <= s["Depth_cm"] else 0
            height_f = 1 if p["Max_Stack_Height_cm"] <= s["Height_Clearance_cm"] else 0
            
            if mode_f and heavy_f and depth_f and height_f:
                feas_matrix[p_idx, s_idx] = 1

    # 5. Load Current Layout & Categories
    current_layout = pd.read_excel(path, sheet_name="Current_Layout", engine="openpyxl")
    for col in current_layout.select_dtypes(include=["object", "str"]).columns:
        current_layout[col] = current_layout[col].str.strip()
    current_layout["Current_Facing"] = pd.to_numeric(current_layout["Current_Facing"], errors="coerce")

    category_rules = pd.read_excel(path, sheet_name="Category_Balance", engine="openpyxl")
    for col in category_rules.select_dtypes(include=["object", "str"]).columns:
        category_rules[col] = category_rules[col].str.strip()
    category_rules = category_rules.set_index("Category")

    return {
        "products": products.reset_index(drop=True),
        "shelves": shelves.reset_index(drop=True),
        "feasibility": feas_matrix,
        "current_layout": current_layout.reset_index(drop=True),
        "category_rules": category_rules
    }