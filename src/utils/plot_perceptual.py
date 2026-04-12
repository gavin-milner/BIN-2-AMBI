# Plotting Script - Plot the results from the perceptual test .csv file

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. Load data ──────────────────────────────────────────────────────────────
# Run normally and drag-and-drop a CSV onto the script, or pass it as an arg:
#   python mushra_boxplots.py /path/to/data.csv
# If no file is given, falls back to mushra.csv in the same folder as the script.
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[-1])
else:
    csv_path = Path(__file__).resolve().parent / "mushra.csv"

if not csv_path.exists():
    sys.exit(f"ERROR: Could not find CSV file at: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Loaded: {csv_path}")

# ── 2. Per-listener z-score normalisation ────────────────────────────────────
listener_stats = (
    df.groupby("session_uuid")["rating_score"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "listener_mean", "std": "listener_std"})
)

df = df.join(listener_stats, on="session_uuid")
df["score_norm"] = (df["rating_score"] - df["listener_mean"]) / df["listener_std"]

# ── 3. Rescale back to 0-100 ─────────────────────────────────────────────────
global_min = df["score_norm"].min()
global_max = df["score_norm"].max()
df["score_rescaled"] = (df["score_norm"] - global_min) / (global_max - global_min) * 100

# ── 4. Configuration ──────────────────────────────────────────────────────────

# Figure size (width, height) in inches
FIGURE_SIZE = (8.5, 5)

# Toggle models on/off — set to False to hide a model from the plot
SHOW_STIMULUS = {
    "reference":  True,
    "DIRAC_GT":   True,
    "MAE":        True,
    "MSE":        True,
    "MSE_BG":     True,
    "MSE_STATIC": True,
    "anchor35":   True,
}

# Display labels (edit to rename any stimulus on the x-axis)
STIMULUS_LABELS = {
    "reference":  "FOA Reference",
    "DIRAC_GT":   "DIRAC_GT",
    "MAE":        "MAE",
    "MSE":        "MSE",
    "MSE_BG":     "MSE_BG",
    "MSE_STATIC": "MSE (Untrained)",
    "anchor35":   "Anchor35",
}

# Box colours — one per stimulus (edit hex codes freely)
BOX_COLOURS = {
    "reference":  "#4C72B0",
    "DIRAC_GT":   "#55A868",
    "MAE":        "#29a8c2",
    "MSE":        "#305cba",
    "MSE_BG":     "#f5e267",
    "MSE_STATIC": "#ba3a3a", 
    "anchor35":   "#BBBBBB",
}

# ── 5. Build data for boxplot ─────────────────────────────────────────────────
stimuli = [s for s in SHOW_STIMULUS if SHOW_STIMULUS[s] and s in df["rating_stimulus"].unique()]
plot_data = [df.loc[df["rating_stimulus"] == s, "score_rescaled"].values for s in stimuli]

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

bp = ax.boxplot(
    plot_data,
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.4),
    capprops=dict(linewidth=1.4),
    flierprops=dict(marker="o", markersize=5, linestyle="none", markeredgewidth=1),
)

for patch, stimulus in zip(bp["boxes"], stimuli):
    patch.set_facecolor(BOX_COLOURS[stimulus])
    patch.set_alpha(0.75)

ax.set_xticks(range(1, len(stimuli) + 1))
ax.set_xticklabels([STIMULUS_LABELS[s].replace(" ", chr(10)) for s in stimuli], fontsize=11, multialignment="center")
ax.set_ylabel("Normalised Quality Score", fontsize=12)
ax.set_xlabel("Stimulus / Model", fontsize=12)
ax.set_title("Pilot Test MUSHRA Ratings", fontsize=14)
ax.set_ylim(-5, 105)

for y in [20, 40, 60, 80, 100]:
    ax.axhline(y=y, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)

for y, label in [(10, "Bad"), (30, "Poor"), (50, "Fair"), (70, "Good"), (90, "Excellent")]:
    ax.text(len(stimuli) + 0.55, y, label, va="center", fontsize=8, color="grey")

ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()

output_path = csv_path.with_name("mushra_boxplots.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.show()