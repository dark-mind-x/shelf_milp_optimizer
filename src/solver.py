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
            "best_layout": pd.DataFrame(), "utility_score": 0.0, "solver_status": status,
            "improvement": {}, "category_summary": pd.DataFrame(), "shelf_summary": pd.DataFrame()
        }

    best_layout = _decode(problem, variables, data)
    improvement = _compute_improvement(best_layout, data)
    cat_summary = _category_summary(best_layout, data)
    shelf_summary = _shelf_summary(best_layout, data)

    return {
        "best_layout": best_layout, "utility_score": pulp.value(model.objective),
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
            rows.append({
                "Product_ID": products.loc[i, "Product_ID"],
                "Product_Name": products.loc[i, "Product_Name"],
                "Category": products.loc[i, "Category"],
                "Display_Mode": products.loc[i, "Display_Mode"],
                "Location_ID": shelves.loc[s, "Location_ID"],
                "Shelf_Level": shelves.loc[s, "Shelf_Level"],
                "Facings": facings,
                "Display_Units": int(round(facings * float(products.loc[i, "Garments_per_Facing_Cap"]))),
                "Utility_Score": float(products.loc[i, "Unit_Margin_Rs"] * (shelves.loc[s, "Reachability_Score"]/5.0) * shelves.loc[s, "Visibility_Multiplier"] * facings),
                "Facing_Width_cm": float(products.loc[i, "Facing_Width_cm"])
            })
    return pd.DataFrame(rows).sort_values("Location_ID").reset_index(drop=True)

def _compute_improvement(best_layout, data) -> dict:
    curr = data["current_layout"]
    loc_to_idx = dict(zip(data["shelves"]["Location_ID"], data["shelves"].index))
    prod_to_idx = dict(zip(data["products"]["Product_ID"], data["products"].index))

    current_utility = 0.0
    for _, row in curr.iterrows():
        p_idx, s_idx = prod_to_idx.get(row["Product_ID"]), loc_to_idx.get(row["Current_Location_ID"])
        if p_idx is not None and s_idx is not None:
            current_utility += (float(data["products"].loc[p_idx, "Unit_Margin_Rs"]) * (float(data["shelves"].loc[s_idx, "Reachability_Score"]) / 5.0) * float(data["shelves"].loc[s_idx, "Visibility_Multiplier"]) * float(row["Current_Facing"]))

    optimized_utility = best_layout["Utility_Score"].sum()
    gain_pct = ((optimized_utility - current_utility) / current_utility) * 100 if current_utility > 0 else 0.0

    return {"current_utility": current_utility, "optimized_utility": optimized_utility, "utility_gain_pct": gain_pct, "products_placed": len(best_layout), "total_products": len(data["products"])}

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
        
        rows.append({"Location_ID": loc, "Shelf_Level": shelves.loc[s, "Shelf_Level"], "Products": len(on_shelf), "Used_Width_cm": round(used_w, 1), "Shelf_Width_cm": shelf_w, 
                     "Utilization_%": round((used_w / shelf_w * 100) if shelf_w > 0 else 0, 1), "Min_Allowed_%": round(float(shelves.loc[s, "Min_Width_Utilization"]) * 100, 1), 
                     "Max_Allowed_%": round(float(shelves.loc[s, "Max_Width_Utilization"]) * 100, 1)})
    return pd.DataFrame(rows)