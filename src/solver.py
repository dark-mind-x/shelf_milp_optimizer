import pulp
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from problem import ShelfMILPProblem

def run_solver(data: dict, time_limit: int = 60, verbose: bool = True) -> dict:
    problem = ShelfMILPProblem(data)
    model, variables = problem.build()

    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    status = pulp.LpStatus[model.status]

    if status not in ("Optimal", "Integer Feasible"):
        return {
            "best_layout": pd.DataFrame(), "profit_score": 0.0, "solver_status": status,
            "improvement": {}, "category_summary": pd.DataFrame(), "shelf_summary": pd.DataFrame()
        }

    best_layout = _decode(problem, variables, data)
    
    # Clean improvement dict containing ONLY the optimized metrics
    optimized_profit = best_layout["Profit_Rs"].sum() if not best_layout.empty else 0.0
    improvement = {
        "optimized_profit": optimized_profit, 
        "products_placed": len(best_layout), 
        "total_products": len(data["products"])
    }
    
    cat_summary = _category_summary(best_layout, data)
    shelf_summary = _shelf_summary(best_layout, data)

    return {
        "best_layout": best_layout, "profit_score": pulp.value(model.objective),
        "solver_status": status, "improvement": improvement,
        "category_summary": cat_summary, "shelf_summary": shelf_summary
    }

def _decode(problem, variables, data) -> pd.DataFrame:
    x, f = variables["x"], variables["f"]
    products, shelves = data["products"], data["shelves"]
    rows = []
    
    for (i, s) in problem.feasible_pairs:
        x_val = pulp.value(x[i, s])
        if x_val is not None and x_val >= 0.5:
            facings = int(round(pulp.value(f[i, s])))
            display_units = int(round(facings * float(products.loc[i, "Garments_per_Facing_Cap"])))
            rows.append({
                "Product_ID": products.loc[i, "Product_ID"],
                "Product_Name": products.loc[i, "Product_Name"],
                "Category": products.loc[i, "Category"],
                "Display_Mode": products.loc[i, "Display_Mode"],
                "Location_ID": shelves.loc[s, "Location_ID"],
                "Shelf_Level": shelves.loc[s, "Shelf_Level"],
                "Facings": facings,
                "Display_Units": display_units,
                "Profit_Rs": float(products.loc[i, "Unit_Margin_Rs"] * display_units),
                "Facing_Width_cm": float(products.loc[i, "Facing_Width_cm"])
            })
    return pd.DataFrame(rows).sort_values("Location_ID").reset_index(drop=True)

def _category_summary(best_layout, data):
    cat_rules = data["category_rules"]
    rows = []
    for cat in cat_rules.index:
        cat_rows = best_layout[best_layout["Category"] == cat]
        locs, facings = cat_rows["Location_ID"].nunique(), cat_rows["Facings"].sum()
        min_l, max_l = cat_rules.loc[cat, "Minimum_Locations"], cat_rules.loc[cat, "Maximum_Locations"]
        min_f = cat_rules.loc[cat, "Minimum_Total_Facings"]
        
        ok = (pd.isna(min_l) or locs >= min_l) and (pd.isna(max_l) or locs <= max_l) and (pd.isna(min_f) or facings >= min_f)
        rows.append({"Category": cat, "Locations_Used": locs, "Min_Locations": min_l, "Max_Locations": max_l, "Total_Facings": facings, "Min_Total_Facings": min_f, "Within_Rules": ok})
    return pd.DataFrame(rows)

def _shelf_summary(best_layout, data):
    shelves, rows = data["shelves"], []
    for s in range(len(shelves)):
        loc = shelves.loc[s, "Location_ID"]
        on_shelf = best_layout[best_layout["Location_ID"] == loc]
        used_w = (on_shelf["Facing_Width_cm"] * on_shelf["Facings"]).sum() if not on_shelf.empty else 0.0
        shelf_w = float(shelves.loc[s, "Width_cm"])
        
        rows.append({"Location_ID": loc, "Shelf_Level": shelves.loc[s, "Shelf_Level"], "Width_Capacity": shelf_w, "Used_Width": used_w, "Utilization": used_w/shelf_w if shelf_w else 0, "Min_Target": float(shelves.loc[s, "Min_Width_Utilization"]), "Max_Target": float(shelves.loc[s, "Max_Width_Utilization"])})
    return pd.DataFrame(rows)