# Plotting Script - plot the results for horizontal and vertical
# decomposition of angular error

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONFIGURATION — edit these paths for your experiment
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR      = PROJECT_ROOT / "experiments" / "6000scenes_no_bg"

ROTATION_JSON   = EXP_DIR / "test_results_rotation_3.json"
STATIC_JSON     = EXP_DIR / "test_results_static_3.json"

# Scene positions: accepts either a .csv or a .jsonl manifest.
# CSV must have columns: scene_id, n_sources, azimuth_deg, elevation_deg
# JSONL must have fields: scene_id, n_sources, azimuth_deg, elevation_deg
SCENE_POSITIONS = EXP_DIR / "test_scene_positions.csv"

# Output directory (defaults to experiment folder)
OUTPUT_DIR = EXP_DIR / "figures"

# Bin width in degrees for both plots
BIN_WIDTH = 30

N_SOURCES = 5

# =============================================================================


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_scene_positions(path: Path) -> pd.DataFrame:
    """Load scene positions from either a CSV or JSONL manifest."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in (".jsonl", ".json"):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported scene positions format: {path.suffix}")


def build_diff_df(
    rot_wmae: dict,
    sta_wmae: dict,
    positions: pd.DataFrame,
    angle_col: str,
) -> pd.DataFrame:
    """
    For each single-source scene, compute ΔWMAE = static − rotation.
    Returns a DataFrame with columns [scene_id, angle_col, diff].
    """
    #df_single = positions[positions["n_sources"] == N_SOURCES].copy()
    df_single = positions.copy()
    records = []
    for _, row in df_single.iterrows():
        scene = row["scene_id"]
        if scene in rot_wmae and scene in sta_wmae:
            diff = (sta_wmae[scene] - rot_wmae[scene]) / row["n_sources"]
            records.append({
                "scene_id": scene,
                angle_col: row[angle_col],
                "diff": diff,
            })
    return pd.DataFrame(records)


def wrap_azimuth(az: pd.Series) -> pd.Series:
    """Wrap azimuth values into (−180, 180]."""
    return az.apply(lambda x: ((x + 180) % 360) - 180)


def bin_and_aggregate(
    df: pd.DataFrame,
    angle_col: str,
    bin_width: int,
    angle_min: float | None = None,
    angle_max: float | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Bin df[angle_col] into bins of bin_width degrees.
    Returns (bin_centers, bin_mean, bin_sem).
    """
    if angle_min is None:
        angle_min = np.floor(df[angle_col].min() / bin_width) * bin_width
    if angle_max is None:
        angle_max = np.ceil(df[angle_col].max() / bin_width) * bin_width

    bin_edges   = np.arange(angle_min, angle_max + 1, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df = df.copy()
    df["bin"] = pd.cut(
        df[angle_col], bins=bin_edges, labels=bin_centers, include_lowest=True
    ).astype(float)

    grouped  = df.groupby("bin")["diff"]
    bin_mean = grouped.mean().reindex(bin_centers)
    bin_sem  = grouped.sem().reindex(bin_centers)

    return bin_centers, bin_mean, bin_sem


def bar_colors(means: pd.Series) -> list[str]:
    return ["#2ca02c" if (v >= 0 or np.isnan(v)) else "#d62728" for v in means]


def legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color="#2ca02c", alpha=0.8, label="Rotation better (bin mean > 0)"),
        mpatches.Patch(color="#d62728", alpha=0.8, label="Rotation worse  (bin mean < 0)"),
    ]


# ---------------------------------------------------------------------------
# Plot 1 — Horizontal WMAE vs Azimuth
# ---------------------------------------------------------------------------

def plot_horizontal(rot_json: dict, sta_json: dict, positions: pd.DataFrame, output_dir: Path):
    rot_wmae = rot_json["per_scene_horiz_wmae"]
    sta_wmae = sta_json["per_scene_horiz_wmae"]

    df = build_diff_df(rot_wmae, sta_wmae, positions, "azimuth_deg")
    df["azimuth_deg"] = wrap_azimuth(df["azimuth_deg"])

    bin_centers, bin_mean, bin_sem = bin_and_aggregate(
        df, "azimuth_deg", BIN_WIDTH, angle_min=-180, angle_max=180
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(bin_centers, bin_mean, width=BIN_WIDTH - 1,
           color=bar_colors(bin_mean), alpha=0.75, zorder=2)
    ax.errorbar(bin_centers, bin_mean, yerr=bin_sem,
                fmt="none", color="black", capsize=3, linewidth=1.1, zorder=3)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

    ax.set_xlabel("Azimuth (degrees)", fontsize=12)
    ax.set_ylabel("Mean ΔWMAE  (Static − Rotation)  [°]", fontsize=12)
    ax.set_title(
        "Horizontal WMAE: Rotation vs Static "
        f"(No BG, {BIN_WIDTH}° Bins, "
        "Error Bars = ±1 SEM)",
        fontsize=13,
    )
    ax.set_xlim(-185, 185)
    ax.set_xticks(range(-180, 181, 30))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    ax.legend(handles=legend_patches(), loc="upper right", fontsize=10)

    plt.tight_layout()
    out = output_dir / "wmae_horiz_diff.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    n_improved = (df["diff"] > 0).sum()
    n_worsened = (df["diff"] < 0).sum()
    print(f"  Scenes improved: {n_improved} / {len(df)}  |  "
          f"worsened: {n_worsened} / {len(df)}  |  "
          f"mean ΔWMAE: {df['diff'].mean():.3f}°")


# ---------------------------------------------------------------------------
# Plot 2 — Vertical WMAE vs Elevation
# ---------------------------------------------------------------------------

def plot_vertical(rot_json: dict, sta_json: dict, positions: pd.DataFrame, output_dir: Path):
    rot_wmae = rot_json["per_scene_vert_wmae"]
    sta_wmae = sta_json["per_scene_vert_wmae"]

    df = build_diff_df(rot_wmae, sta_wmae, positions, "elevation_deg")

    bin_centers, bin_mean, bin_sem = bin_and_aggregate(
        df, "elevation_deg", BIN_WIDTH
    )

    elev_min = bin_centers[0] - BIN_WIDTH
    elev_max = bin_centers[-1] + BIN_WIDTH

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(bin_centers, bin_mean, height=BIN_WIDTH - 1,
            color=bar_colors(bin_mean), alpha=0.75, zorder=2)
    ax.errorbar(bin_mean, bin_centers, xerr=bin_sem,
                fmt="none", color="black", capsize=3, linewidth=1.1, zorder=3)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")

    ax.set_ylabel("Elevation (degrees)", fontsize=12)
    ax.set_xlabel("Mean ΔWMAE  (Static − Rotation)  [°]", fontsize=12)
    ax.set_title(
        "Vertical WMAE: Rotation vs Static "
        f"(No BG {BIN_WIDTH}° Bins "
        "Error Bars = ±1 SEM)",
        fontsize=12,
    )
    ax.set_ylim(elev_min, elev_max)
    ax.set_yticks(range(int(elev_min), int(elev_max) + 1, 20))
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.2)
    ax.legend(handles=legend_patches(), loc="lower right", fontsize=10)

    plt.tight_layout()
    out = output_dir / "wmae_vert_diff.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    n_improved = (df["diff"] > 0).sum()
    n_worsened = (df["diff"] < 0).sum()
    print(f"  Scenes improved: {n_improved} / {len(df)}  |  "
          f"worsened: {n_worsened} / {len(df)}  |  "
          f"mean ΔWMAE: {df['diff'].mean():.3f}°")


# ---------------------------------------------------------------------------
# Plot 3 — Polar bar chart: Horizontal WMAE vs Azimuth
# ---------------------------------------------------------------------------

def plot_horizontal_polar(rot_json: dict, sta_json: dict, positions: pd.DataFrame, output_dir: Path):
    rot_wmae = rot_json["per_scene_horiz_wmae"]
    sta_wmae = sta_json["per_scene_horiz_wmae"]

    df = build_diff_df(rot_wmae, sta_wmae, positions, "azimuth_deg")
    df["azimuth_deg"] = wrap_azimuth(df["azimuth_deg"])

    bin_centers, bin_mean, bin_sem = bin_and_aggregate(
        df, "azimuth_deg", BIN_WIDTH, angle_min=-180, angle_max=180
    )

    # Convert to polar: 0° (front) at top, clockwise
    az_rad    = np.deg2rad(bin_centers)
    theta     = - az_rad
    bar_width = np.deg2rad(BIN_WIDTH) * 0.85

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for t, mean, sem in zip(theta, bin_mean, bin_sem):
        if np.isnan(mean):
            continue
        color = "#2ca02c" if mean >= 0 else "#d62728"
        ax.bar(t, abs(mean), width=bar_width,
               bottom=0,
               color=color, alpha=0.75, zorder=2)
        # ax.errorbar(t, abs(mean), yerr=sem, fmt="none",
        #             color="black", capsize=2, linewidth=0.9, zorder=3)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°\n(Front)", "45°", "90°\n(Right)", "135°",
                         "±180°\n(Rear)", "-135°", "-90°\n(Left)", "-45°"],
                       fontsize=8)
    ax.set_ylabel("|Mean ΔWMAE| (°)", labelpad=30, fontsize=10)
    ax.set_title(
        "Horizontal WMAE Improvement by Azimuth\n"
        f"(No BG, {BIN_WIDTH}° Bins, Error Bars = ±1 SEM)",
        fontsize=11, pad=20,
    )

    better = mpatches.Patch(color="#2ca02c", alpha=0.8, label="Rotation better")
    worse  = mpatches.Patch(color="#d62728", alpha=0.8, label="Rotation worse")
    ax.legend(handles=[better, worse], loc="lower right",
              bbox_to_anchor=(1.25, -0.05), fontsize=9)

    plt.tight_layout()
    out = output_dir / "wmae_horiz_polar.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading results from:\n  {ROTATION_JSON}\n  {STATIC_JSON}")
    rot_json = load_json(ROTATION_JSON)
    sta_json = load_json(STATIC_JSON)

    print(f"Loading scene positions from:\n  {SCENE_POSITIONS}")
    positions = load_scene_positions(SCENE_POSITIONS)
    print(f"  Total events: {len(positions)}  |  "
          f"Single-source scenes: {(positions['n_sources'] == 1).sum()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Horizontal ──")
    plot_horizontal(rot_json, sta_json, positions, OUTPUT_DIR)

    print("\n── Vertical ──")
    plot_vertical(rot_json, sta_json, positions, OUTPUT_DIR)

    print("\n── Horizontal Polar ──")
    plot_horizontal_polar(rot_json, sta_json, positions, OUTPUT_DIR)