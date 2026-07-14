import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

CATEGORY_COLORS = {"Tops": "#4CAF50", "Denim": "#2196F3", "Shirts": "#FF9800", "Dresses": "#E91E63", "Bottoms": "#9C27B0", "Ethnic": "#FF5722", "Outerwear": "#607D8B"}

def plot_shelf_layout(best_layout: pd.DataFrame, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(13, 6))
    if best_layout.empty: return

    try:
        n_racks = max([int(loc.split("R")[1]) for loc in best_layout["Location_ID"]])
    except:
        n_racks = 5
    level_order = ["Top", "Middle", "Lower"]

    for rack in range(1, n_racks + 1):
        for row_idx, level in enumerate(level_order):
            loc = f"S{row_idx+1}R{rack}"
            x, y = rack - 1, len(level_order) - 1 - row_idx
            products_here = best_layout[best_layout["Location_ID"] == loc]
            
            ax.add_patch(mpatches.FancyBboxPatch((x+0.03, y+0.03), 0.94, 0.94, boxstyle="round,pad=0.01", facecolor="#FAFAFA" if len(products_here)>0 else "#F3F4F6", edgecolor="#E5E7EB", linewidth=1.5))
            
            if len(products_here) == 0:
                ax.text(x + 0.5, y + 0.5, "Empty", ha="center", va="center", color="#9CA3AF", fontsize=10, style="italic")
                continue

            total_facings = products_here["Facings"].sum()
            current_offset = 0.05
            usable_width = 0.9
            
            for _, prod in products_here.iterrows():
                facings = prod["Facings"]
                cat = prod["Category"]
                color = CATEGORY_COLORS.get(cat, "#999999")
                w = (facings / total_facings) * usable_width if total_facings > 0 else usable_width
                
                ax.add_patch(mpatches.Rectangle((x + current_offset, y + 0.15), w - 0.02, 0.7, facecolor=color, edgecolor="white", linewidth=1, alpha=0.9))
                if w > 0.15:
                    ax.text(x + current_offset + w/2, y + 0.5, f"{facings}F\n{cat[:3]}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                current_offset += w

            ax.text(x + 0.5, y + 0.05, loc, ha="center", va="bottom", color="#4B5563", fontsize=9, fontweight="bold")

    ax.set_xlim(0, n_racks); ax.set_ylim(0, len(level_order))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("2D Retail Shelf Layout Map", fontsize=12, fontweight="bold", pad=15)
    for spine in ax.spines.values(): spine.set_visible(False)

def plot_utilization(shelf_summary: pd.DataFrame, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(10, 5))
    x, y = np.arange(len(shelf_summary)), shelf_summary["Utilization"].values
    min_b, max_b = shelf_summary["Min_Target"].values, shelf_summary["Max_Target"].values
    
    colors = ["#10B981" if (min_b[i] <= y[i] <= max_b[i]) else "#F59E0B" if y[i] < min_b[i] else "#EF4444" for i in range(len(y))]
    
    ax.bar(x, y, color=colors, edgecolor="white", width=0.7)
    for i in range(len(shelf_summary)):
        ax.plot([i-0.35, i+0.35], [min_b[i], min_b[i]], color="#555555", linestyle="--"); ax.plot([i-0.35, i+0.35], [max_b[i], max_b[i]], color="#555555", linestyle="--")

    ax.set_xticks(x); ax.set_xticklabels(shelf_summary["Location_ID"], rotation=45, fontsize=8)
    ax.set_title("Shelf Width Utilization vs Target Band", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

def plot_category_balance(category_summary: pd.DataFrame, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(10, 5))
    cats, y = category_summary["Category"].tolist(), np.arange(len(category_summary))

    for i, row in category_summary.iterrows():
        ax.plot([row["Min_Locations"], row["Max_Locations"]], [i, i], color="#CCCCCC", linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter(row["Locations_Used"], i, s=180, color=CATEGORY_COLORS.get(row["Category"], "#999999"), edgecolor="white", zorder=3, marker="o" if row["Within_Rules"] else "X")

    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Number of Unique Locations", fontsize=9)
    ax.set_title("Category Representation vs Business Rules", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)