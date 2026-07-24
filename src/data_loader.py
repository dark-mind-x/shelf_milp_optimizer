import pandas as pd
import numpy as np
from pathlib import Path

# Try to locate files dynamically to avoid naming issues
try:
    BASE_DIR = Path(__file__).parent.parent 
except:
    BASE_DIR = Path.cwd()

def load_all(path=None):
    if path is None:
        import glob
        files = glob.glob(str(BASE_DIR / "data" / "*MILP_Dataset*.xlsx")) + glob.glob("*MILP_Dataset*.xlsx")
        path = files[0] if files else "Second_Paper_MILP_Dataset_Full.xlsx"

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

    # 5. Load Before Optimization Data (For Streamlit UI)
    before_layout = pd.DataFrame()
    try:
        import glob
        before_files = glob.glob(str(BASE_DIR / "data" / "Before_optimization*.xlsx")) + glob.glob("Before_optimization*.xlsx")
        if before_files:
            df_before = pd.read_excel(before_files[0], engine="openpyxl")
            
            # Standardize columns so the 3D visualizer can render the before state perfectly
            col_map = {
                'Product name': 'Product_Name',
                'category ': 'Category',
                'display mode': 'Display_Mode',
                'Current facings ': 'Facings',
                'Current product location ': 'Location_ID',
                'facing width': 'Facing_Width_cm',
                'Total Profit': 'Profit_Rs',
                'Total garemnts on shelf ': 'Display_Units'
            }
            before_layout = df_before.rename(columns=col_map)
            before_layout = before_layout.dropna(subset=['Location_ID', 'Facings'])
            
            # Map location IDs to shelf levels
            def get_level(loc):
                if pd.isna(loc): return "Middle"
                if 'S1' in str(loc): return "Top"
                if 'S2' in str(loc): return "Middle"
                return "Lower"
                
            before_layout['Shelf_Level'] = before_layout['Location_ID'].apply(get_level)
            for c in ['Facings', 'Facing_Width_cm', 'Profit_Rs', 'Display_Units']:
                if c in before_layout.columns:
                    before_layout[c] = pd.to_numeric(before_layout[c], errors='coerce').fillna(0)
                    
    except Exception as e:
        print(f"Notice: Before_optimization.xlsx not loaded. ({e})")

    return {
        "products": products, "shelves": shelves, 
        "feasibility": feas_matrix, "category_rules": category_rules,
        "before_layout": before_layout
    }