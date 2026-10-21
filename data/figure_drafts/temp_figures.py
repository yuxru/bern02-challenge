import matplotlib.pyplot as plt
import pandas as pd
 
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
 