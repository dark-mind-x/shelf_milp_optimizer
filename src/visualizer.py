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

    n_racks = max([int(loc.split("R")[1]) for loc in best_layout["Location_ID"]])
    level_order = ["Top", "Middle", "Lower"]

    for rack in range(1, n_racks + 1):
        for row_idx, level in enumerate(level_order):
            loc = f"S{row_idx+1}R{rack}"
            x, y = rack - 1, len(level_order) - 1 - row_idx
            products_here = best_layout[best_layout["Location_ID"] == loc]
            
            ax.add_patch(mpatches.FancyBboxPatch((x+0.03, y+0.03), 0.94, 0.94, boxstyle="round,pad=0.01", facecolor="#FAFAFA" if len(products_here)>0 else "#F0F0F0", edgecolor="#999999"))
            
            if len(products_here) == 0:
                ax.text(x+0.5, y+0.5, "Empty", ha="center", va="center", fontsize=8, color="#AAAAAA")
            else:
                strip_h = 0.9 / len(products_here)
                for idx, (_, prod) in enumerate(products_here.iterrows()):
                    color = CATEGORY_COLORS.get(prod["Category"], "#999999")
                    strip_y = y + 0.05 + idx * strip_h
                    ax.add_patch(mpatches.FancyBboxPatch((x+0.06, strip_y+0.01), 0.88, strip_h-0.02, boxstyle="round,pad=0.005", facecolor=color, edgecolor="white"))
                    
                    name = prod["Product_Name"][:14] + ".." if len(prod["Product_Name"]) > 16 else prod["Product_Name"]
                    ax.text(x+0.5, strip_y+strip_h/2, f"{name} ({prod['Facings']})", ha="center", va="center", fontsize=6, color="white", fontweight="bold")

    ax.set_xlim(0, n_racks); ax.set_ylim(0, len(level_order))
    ax.set_xticks([i + 0.5 for i in range(n_racks)]); ax.set_xticklabels([f"Rack {i+1}" for i in range(n_racks)])
    ax.set_yticks([i + 0.5 for i in range(len(level_order))]); ax.set_yticklabels(["Lower (Hanging)", "Middle (Folded)", "Top (Folded)"])
    ax.set_title("Optimized Shelf Layout", fontsize=13, fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=7, frameon=False)

def plot_utilization(shelf_summary: pd.DataFrame, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(13, 5))
    x, util = np.arange(len(shelf_summary)), shelf_summary["Utilization_%"].values
    min_b, max_b = shelf_summary["Min_Allowed_%"].values, shelf_summary["Max_Allowed_%"].values
    colors = ["#FFA726" if u < mn else "#EF5350" if u > mx else "#66BB6A" for u, mn, mx in zip(util, min_b, max_b)]

    ax.bar(x, util, color=colors, edgecolor="white", width=0.7)
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
        ax.text(row["Min_Locations"] - 0.15, i, str(row["Min_Locations"]), ha="right", va="center", fontsize=8)
        ax.text(row["Max_Locations"] + 0.15, i, str(row["Max_Locations"]), ha="left", va="center", fontsize=8)

    ax.set_yticks(y); ax.set_yticklabels(cats)
    ax.set_title("Category Balance", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)