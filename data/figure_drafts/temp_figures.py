import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

 
# Test-set results for the DFT (B3LYP) target, typed in from the results table
results = pd.DataFrame(
    {"MAE (eV)":  [1.074, 0.272, 0.292, 0.247, 0.390],
     "RMSE (eV)": [1.281, 0.365, 0.388, 0.328, 0.505],
     "R2":        [-0.000, 0.919, 0.908, 0.934, 0.845]},
    index=["baseline (mean)", "A: chemistry variables", "B: Morgan counts",
           "C: combined", "D: Ridge (C)"],
)
 
models = results.drop(index="baseline (mean)")            # bars = the four models only
baseline_mae = results.loc["baseline (mean)", "MAE (eV)"]  # goes in the title instead
 
short_names = ["A", "B", "C", "D"]
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]      # same colours as the GW slide
 
# (column, panel title, y-limits) - same limits as the GW figure so the slides compare 1:1
panels = [("MAE (eV)", "MAE (eV)", (0, 0.6)),
          ("RMSE (eV)", "RMSE (eV)", (0, 0.6)),
          ("R2", "R²", (0, 1.0))]
 
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (column, title, ylim) in zip(axes, panels):
    bars = ax.bar(short_names, models[column], color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)   # value printed on each bar
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
 
fig.suptitle(f"Test Performance, DFT (B3LYP) target (baseline MAE = {baseline_mae:.2f} eV)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
fig.legend(handles, models.index, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))                   # leave room for legend + title
fig.savefig("dft_test_performance.png", dpi=300)
plt.show()

# %% shap
# Compare DFT- and GW-target model C: grouped SHAP importance from both notebooks


RUNS = {"DFT (B3LYP)": Path("results/gap_5"),      # RESULTS_DIR of gap_model.ipynb
        "GW": Path("results/gap_gw")}              # RESULTS_DIR of gap_model_gw.ipynb
COLORS = {"DFT (B3LYP)": "#5B8DB8", "GW": "#D6B656"}
SHOW_SHARE = False   # True = each variable's share of the total, instead of eV

# both notebooks must have explained the same molecules, otherwise the comparison is unfair
samples = [pd.read_csv(d / "shap_molecules.csv")["qm9_index"] for d in RUNS.values()]
assert samples[0].equals(samples[1]), "The two notebooks explained different molecules"

shap_by_target = pd.DataFrame({target: pd.read_csv(d / "shap_grouped_importance.csv", index_col=0).iloc[:, 0]
                               for target, d in RUNS.items()})

if SHOW_SHARE:
    plot_df, xlabel, fmt = shap_by_target / shap_by_target.sum() * 100, "Share of total SHAP importance [%]", "%.0f%%"
else:
    plot_df, xlabel, fmt = shap_by_target, "Mean |SHAP value| [eV]", "%.2f"

order = plot_df.mean(axis=1).sort_values().index            # most important at the top
y = np.arange(len(order))
height = 0.38                                              # two bars per row

fig, ax = plt.subplots(figsize=(9, 6))
for i, target in enumerate(plot_df.columns):
    bars = ax.barh(y + (0.5 - i) * height, plot_df.loc[order, target], height=height,
                   color=COLORS[target], label=target)
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
ax.set_yticks(y, order)
ax.set_xlabel(xlabel)
ax.set_title("Which inputs drive the predicted gap? (model C, SHAP)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Target", frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("shap_importance_dft_vs_gw.png", dpi=200)
print(shap_by_target.round(3))