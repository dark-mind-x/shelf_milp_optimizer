import pulp
import pandas as pd

class ShelfMILPProblem:
    def __init__(self, data: dict):
        self.data           = data
        self.products       = data["products"]
        self.shelves        = data["shelves"]
        self.feasibility    = data["feasibility"]
        self.category_rules = data["category_rules"]

        self.n_products = len(self.products)
        self.n_shelves  = len(self.shelves)

        self.feasible_pairs = [
            (i, s) for i in range(self.n_products) for s in range(self.n_shelves) if self.feasibility[i, s] == 1
        ]

    def build(self):
        products, shelves = self.products, self.shelves
        model = pulp.LpProblem("Shelf_Display_Allocation", pulp.LpMaximize)

        x = {} 
        f = {} 
        for (i, s) in self.feasible_pairs:
            x[i, s] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
            f[i, s] = pulp.LpVariable(f"f_{i}_{s}", lowBound=0, cat="Integer")

        categories = self.category_rules.index.tolist()
        z = {} 
        for c in categories:
            for s in range(self.n_shelves):
                z[c, s] = pulp.LpVariable(f"z_{c}_{s}", cat="Binary")

        # --- DYNAMIC +15% PROFIT TARGET ---
        total_profit_expr = pulp.lpSum(
            products.loc[i, "Unit_Margin_Rs"] * products.loc[i, "Garments_per_Facing_Cap"] * f[i, s] 
            for (i, s) in self.feasible_pairs
        )
        
        before_layout = self.data.get("before_layout", pd.DataFrame())
        if not before_layout.empty and "Profit_Rs" in before_layout.columns:
            baseline_profit = before_layout["Profit_Rs"].sum()
        else:
            baseline_profit = 346990 
            
        target_profit = baseline_profit * 1.15  # Targets exactly +15%
        
        over_target = pulp.LpVariable("over_target", lowBound=0, cat="Continuous")
        under_target = pulp.LpVariable("under_target", lowBound=0, cat="Continuous")
        
        model += total_profit_expr - over_target + under_target == target_profit

        # Allow Stockroom drops
        for i in range(self.n_products):
            model += pulp.lpSum(x[i, s] for s in [ps for (pi, ps) in self.feasible_pairs if pi == i]) <= 1

        # --- THE 18-UNIT CAP IS COMPLETELY REMOVED ---
        # It now strictly obeys the Excel file without any artificial restrictions.
        for (i, s) in self.feasible_pairs:
            excel_min = products.loc[i, "Min_Facing"]
            excel_max = products.loc[i, "Max_Facing"]

            model += f[i, s] <= excel_max * x[i, s]
            model += f[i, s] >= excel_min * x[i, s]

        # --- FILL THE RACKS (Without Crashing) ---
        width_deficits = []
        for s in range(self.n_shelves):
            products_on_s = [i for (i, ps) in self.feasible_pairs if ps == s]
            
            if len(products_on_s) > 0:
                model += pulp.lpSum(x[i, s] for i in products_on_s) >= 1
            
            used_w = pulp.lpSum(products.loc[i, "Facing_Width_cm"] * f[i, s] for i in products_on_s)
            
            max_w = shelves.loc[s, "Max_Width_Utilization"] * shelves.loc[s, "Width_cm"]
            model += used_w <= max_w
            
            # SOFT Minimum Target: Aim for 85% full. 
            target_w = shelves.loc[s, "Width_cm"] * 0.85
            def_s = pulp.LpVariable(f"def_w_{s}", lowBound=0, cat="Continuous")
            model += used_w + def_s >= target_w
            width_deficits.append(def_s)
            
            weight_cap = shelves.loc[s, "Weight_Capacity_kg"]
            model += pulp.lpSum(products.loc[i, "Unit_Weight_kg"] * products.loc[i, "Garments_per_Facing_Cap"] * f[i, s] for i in products_on_s) <= weight_cap
            
            area_m2 = (shelves.loc[s, "Width_cm"] * shelves.loc[s, "Depth_cm"]) / 10000.0
            density_cap = shelves.loc[s, "Max_Display_Density_units_m2"] * area_m2
            model += pulp.lpSum(products.loc[i, "Garments_per_Facing_Cap"] * f[i, s] for i in products_on_s) <= density_cap

        # Category rules
        for c in categories:
            prods_in_c = [i for i in range(self.n_products) if products.loc[i, "Category"] == c]
            for s in range(self.n_shelves):
                pairs_here = [i for i in prods_in_c if (i, s) in self.feasible_pairs]
                if not pairs_here:
                    model += z[c, s] == 0
                    continue
                for i in pairs_here:
                    model += z[c, s] >= x[i, s]
                model += z[c, s] <= pulp.lpSum(x[i, s] for i in pairs_here)

            min_loc = self.category_rules.loc[c, "Minimum_Locations"]
            max_loc = self.category_rules.loc[c, "Maximum_Locations"]
            if pd.notna(min_loc): model += pulp.lpSum(z[c, s] for s in range(self.n_shelves)) >= min_loc
            if pd.notna(max_loc): model += pulp.lpSum(z[c, s] for s in range(self.n_shelves)) <= max_loc

            min_f = self.category_rules.loc[c, "Minimum_Total_Facings"]
            if pd.notna(min_f):
                pairs_in_c = [(i, s) for (i, s) in self.feasible_pairs if products.loc[i, "Category"] == c]
                cat_def = pulp.LpVariable(f"cat_def_{c}", lowBound=0, cat="Continuous")
                model += pulp.lpSum(f[i, s] for (i, s) in pairs_in_c) + cat_def >= min_f
                width_deficits.append(cat_def * 10) 

        # --- THE MASTER OBJECTIVE ---
        utility_terms = []
        for (i, s) in self.feasible_pairs:
            margin       = products.loc[i, "Unit_Margin_Rs"]
            reachability = shelves.loc[s, "Reachability_Score"] / 5.0
            visibility   = shelves.loc[s, "Visibility_Multiplier"]
            display_bonus = 0.05 * margin 
            utility_terms.append((margin * reachability * visibility + display_bonus) * f[i, s])
            
        model += pulp.lpSum(utility_terms) - (5.0 * over_target) - (5.0 * under_target) - (1.0 * pulp.lpSum(width_deficits)), "Objective"

        self.model, self.x, self.f, self.z = model, x, f, z
        return model, {"x": x, "f": f, "z": z}