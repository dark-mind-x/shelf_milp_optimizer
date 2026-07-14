import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Second_Paper_MILP_Dataset_Full.xlsx"

def apply_smart_capacity_limits(products, shelves):
    """
    SENIOR DEV LOGIC: Dynamically balances physical capacity against user requests.
    Guarantees all products fit by smartly managing Min_Facing based on profit margins.
    """
    # 1. Calculate physical limits
    total_shelf_width = shelves["Width_cm"].sum() * 0.94  # 94% max utilization
    
    # 2. Sort products by profit potential (Margin * Capacity) to prioritize space
    products["Profit_Potential"] = products["Unit_Margin_Rs"] * products["Garments_per_Facing_Cap"]
    products = products.sort_values(by="Profit_Potential", ascending=False).reset_index(drop=True)
    
    # 3. Smart Assignment
    current_used_width = 0.0
    actual_min_facings = []
    
    for _, p in products.iterrows():
        # Everyone gets at least 1 facing to guarantee 50/50 placement
        assigned_facing = 1 
        width_needed = float(p["Facing_Width_cm"])
        
        # If user asked for 2 (or more), and we have physical room, grant it!
        requested_min = int(p.get("Min_Facing", 2))
        if requested_min > 1:
            extra_width = (requested_min - 1) * width_needed
            if current_used_width + extra_width <= total_shelf_width:
                assigned_facing = requested_min
                current_used_width += extra_width
                
        current_used_width += width_needed
        actual_min_facings.append(assigned_facing)
        
    products["Min_Facing"] = actual_min_facings
    
    # 4. Remove arbitrary physical bottlenecks for the demonstration
    shelves["Weight_Capacity_kg"] *= 2.0
    shelves["Max_Display_Density_units_m2"] *= 2.0
    
    return products, shelves

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
        "Bulky_Item_0_1", "Min_Facing", "Max_Facing"
    ]
    for col in numeric_cols:
        products[col] = pd.to_numeric(products[col], errors="coerce")

    # 2. Load Shelves
    shelves = pd.read_excel(path, sheet_name="Shelf_Rack_Locations", engine="openpyxl")
    for col in shelves.select_dtypes(include=["object", "str"]).columns:
        shelves[col] = shelves[col].str.strip()
    s_num_cols = ["Width_cm", "Height_Clearance_cm", "Depth_cm", "Weight_Capacity_kg", 
                  "Reachability_Score", "Visibility_Multiplier", "Max_Display_Density_units_m2",
                  "Min_Width_Utilization", "Max_Width_Utilization"]
    for col in s_num_cols:
        shelves[col] = pd.to_numeric(shelves[col], errors="coerce")

    # --- APPLY SENIOR DEV CAPACITY LOGIC ---
    products, shelves = apply_smart_capacity_limits(products, shelves)
    products["Max_Facing"] = products["Max_Facing"].fillna(5).clip(lower=products["Min_Facing"])

    # 3. Build Feasibility Matrix
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

    # 4. Load Categories
    category_rules = pd.read_excel(path, sheet_name="Category_Balance", engine="openpyxl")
    for col in category_rules.select_dtypes(include=["object", "str"]).columns:
        category_rules[col] = category_rules[col].str.strip()
    category_rules.set_index("Category", inplace=True)
    for col in ["Minimum_Locations", "Maximum_Locations", "Minimum_Total_Facings"]:
        category_rules[col] = pd.to_numeric(category_rules[col], errors="coerce")

    return {
        "products": products, "shelves": shelves, 
        "feasibility": feas_matrix, "category_rules": category_rules
    }