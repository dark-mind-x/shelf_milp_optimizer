import pulp
import pandas as pd

class ShelfMILPProblem:
    def __init__(self, data: dict):
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

        # Objective
        utility_terms = []
        for (i, s) in self.feasible_pairs:
            margin       = products.loc[i, "Unit_Margin_Rs"]
            reachability = shelves.loc[s, "Reachability_Score"] / 5.0
            visibility   = shelves.loc[s, "Visibility_Multiplier"]
            display_bonus = 0.05 * margin 
            utility_terms.append((margin * reachability * visibility + display_bonus) * f[i, s])
            
        model += pulp.lpSum(utility_terms), "Engineering_Utility_Score"

        # Constraints: Ensure every product IS placed
        for i in range(self.n_products):
            model += pulp.lpSum(x[i, s] for s in [ps for (pi, ps) in self.feasible_pairs if pi == i]) == 1

        for (i, s) in self.feasible_pairs:
            model += f[i, s] <= products.loc[i, "Max_Facing"] * x[i, s]
            model += f[i, s] >= products.loc[i, "Min_Facing"] * x[i, s]

        for s in range(self.n_shelves):
            products_on_s = [i for (i, ps) in self.feasible_pairs if ps == s]
            
            # Width, Weight, and Density caps
            max_w = shelves.loc[s, "Max_Width_Utilization"] * shelves.loc[s, "Width_cm"]
            model += pulp.lpSum(products.loc[i, "Facing_Width_cm"] * f[i, s] for i in products_on_s) <= max_w
            
            weight_cap = shelves.loc[s, "Weight_Capacity_kg"]
            model += pulp.lpSum(products.loc[i, "Unit_Weight_kg"] * products.loc[i, "Garments_per_Facing_Cap"] * f[i, s] for i in products_on_s) <= weight_cap
            
            area_m2 = (shelves.loc[s, "Width_cm"] * shelves.loc[s, "Depth_cm"]) / 10000.0
            density_cap = shelves.loc[s, "Max_Display_Density_units_m2"] * area_m2
            model += pulp.lpSum(products.loc[i, "Garments_per_Facing_Cap"] * f[i, s] for i in products_on_s) <= density_cap

        # Category linking
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
                model += pulp.lpSum(f[i, s] for (i, s) in pairs_in_c) >= min_f

        self.model, self.x, self.f, self.z = model, x, f, z
        return model, {"x": x, "f": f, "z": z}