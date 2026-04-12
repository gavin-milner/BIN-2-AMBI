# Plotting Script - Plot spectrogram results from an experiment

import ast
import csv
import json
from math import comb, floor, ceil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Primary experiment
EXP_DIR = PROJECT_ROOT / "experiments" / "6000scenes"

# Optional second experiment for side-by-side comparison
COMPARE_TWO_EXPERIMENTS = False
EXP_DIR_2 = PROJECT_ROOT / "experiments" / "6000scenes_synthetic_ht"

# Input JSON files
ROTATION_JSON = EXP_DIR / "test_results_rotation_3.json"
STATIC_JSON = EXP_DIR / "test_results_static_3.json"
NO_ROT_JSON = EXP_DIR / "test_results_no_rotation.json"

MANIFEST_JSONL = EXP_DIR / "manifest.jsonl"
SCENE_POSITIONS_CSV = EXP_DIR / "test_scene_positions.csv"

# Second experiment files (if COMPARE_TWO_EXPERIMENTS is True)
ROTATION_JSON_2 = EXP_DIR_2 / "test_results_rotation.json"
STATIC_JSON_2 = EXP_DIR_2 / "test_results_static.json"
NO_ROT_JSON_2 = EXP_DIR_2 / "test_results_no_rotation.json"

MANIFEST_JSONL_2 = EXP_DIR_2 / "manifest.jsonl"
SCENE_POSITIONS_CSV_2 = EXP_DIR_2 / "test_scene_positions.csv"

# Toggle which models to include
COMPARE_ROTATION = True
COMPARE_STATIC = True
COMPARE_NO_ROT = False

# Metric to test
# Options:
#   "per_scene_wmae"
#   "per_scene_mse_norm"
METRIC_KEY = "per_scene_wmae"

# Wilcoxon config
# diff = model_b - model_a
# "greater" means model_a tends to have LOWER error than model_b
WILCOXON_ALTERNATIVE = "greater"
WILCOXON_ZERO_METHOD = "wilcox"

# Output
SAVE_TXT = True
SAVE_JSON = True
SAVE_DIFF_HISTOGRAMS = True
SAVE_GROUP_PLOTS = True
SAVE_AMBIGUITY_BOXPLOT = True
# Plot type for ambiguity plot: "boxplot" or "violin"
AMBIGUITY_PLOT_TYPE = "boxplot"
SAVE_FAILURE_RATE_PLOT = True
SAVE_NSOURCES_AMBIGUITY_HEATMAP = True

TXT_OUTPUT = EXP_DIR / "significance_results.txt"
JSON_OUTPUT = EXP_DIR / "significance_results.json"
HIST_DIR = EXP_DIR / "significance_histograms"
PLOT_DIR = EXP_DIR / "significance_plots"

# Group plot filenames
N_EVENTS_PLOT = PLOT_DIR / "wmae_vs_num_sources_with_std.pdf"
SNR_PLOT = PLOT_DIR / "wmae_vs_snr_bucket_with_std.pdf"
DIFFICULTY_PLOT = PLOT_DIR / "wmae_vs_difficulty_with_std.pdf"
BOXPLOT_N_EVENTS = PLOT_DIR / "wmae_vs_num_sources_boxplot.pdf"
BOXPLOT_AMBIGUITY = PLOT_DIR / "wmae_vs_ambiguity_category_boxplot.pdf"
FAILURE_RATE_PLOT = PLOT_DIR / "failure_rate_vs_threshold.pdf"
NSOURCES_AMBIGUITY_HEATMAP = PLOT_DIR / "delta_wmae_nsources_ambiguity_heatmap.pdf"
SNR_NSOURCES_PLOT = PLOT_DIR / "wmae_vs_nsources_per_snr_bucket.pdf"

# Symmetry / skew warning thresholds
ABS_SKEW_WARNING_THRESHOLD = 1.0
MEAN_MEDIAN_GAP_STD_THRESHOLD = 0.5

SHOW_PLOTS = True

# Plot style
# Scale factor for larger PDFs in 4x4 grid layouts
PLOT_SCALE = 1
FIGSIZE = (9 * PLOT_SCALE, 6 * PLOT_SCALE)
LINEWIDTH = 2.0
CAPSIZE = 4
MARKER_MEAN = "o"
LINESTYLE = (0, (4, 3))

# Font scaling to match increased figure size
plt.rcParams['font.size'] = 11 * PLOT_SCALE
plt.rcParams['axes.labelsize'] = 12 * PLOT_SCALE
plt.rcParams['axes.titlesize'] = 14 * PLOT_SCALE
plt.rcParams['xtick.labelsize'] = 10 * PLOT_SCALE
plt.rcParams['ytick.labelsize'] = 10 * PLOT_SCALE
plt.rcParams['legend.fontsize'] = 10 * PLOT_SCALE

# Per-model colours (in order: rotation, static, no_rot, ...)
MODEL_COLORS = [
    "tab:green",
    "tab:red",
    "tab:green",
]

GRID_ALPHA = 0.22
GRID_LINEWIDTH = 0.6


# ============================================================
# EXPERIMENT LABEL HELPERS
# ============================================================

def experiment_background_suffix(exp_dir: Path) -> str:
    exp_name = exp_dir.name.lower()

    parts = []

    if "no_bg" in exp_name:
        parts.append("No Background")
    else:
        parts.append("W/Background")

    if "synthetic_ht" in exp_name:
        parts.append("Synthetic HT")

    return f"({', '.join(parts)})"


PLOT_TITLE_SUFFIX = experiment_background_suffix(EXP_DIR)


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def get_enabled_model_entries():
    models = []

    if COMPARE_ROTATION:
        models.append(("rotation", ROTATION_JSON))
    if COMPARE_STATIC:
        models.append(("static", STATIC_JSON))
    if COMPARE_NO_ROT:
        models.append(("no_rotation", NO_ROT_JSON))

    if len(models) < 2:
        raise ValueError("At least two model JSON files must be enabled.")

    return models


def load_enabled_model_jsons():
    model_entries = get_enabled_model_entries()
    model_jsons = []

    missing = [path for _, path in model_entries if not path.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required JSON result files:\n{missing_str}")

    for _, path in model_entries:
        model_jsons.append(load_json(path))

    return model_jsons


def load_enabled_model_jsons_for_exp(exp_dir, rotation_json, static_json, no_rot_json):
    """Load model JSONs for a specific experiment directory."""
    models = []

    if COMPARE_ROTATION:
        models.append(("rotation", rotation_json))
    if COMPARE_STATIC:
        models.append(("static", static_json))
    if COMPARE_NO_ROT:
        models.append(("no_rotation", no_rot_json))

    if len(models) < 1:
        return []

    model_jsons = []
    for _, path in models:
        if path.exists():
            model_jsons.append(load_json(path))

    return model_jsons


def extract_metric_map(results_json, metric_key):
    if metric_key not in results_json:
        raise KeyError(
            f"Could not find '{metric_key}' in JSON for model "
            f"{results_json.get('model_name', 'unknown')}"
        )
    return results_json[metric_key]


def align_scene_metrics(map_a, map_b):
    common_scene_ids = sorted(set(map_a.keys()) & set(map_b.keys()))

    if len(common_scene_ids) == 0:
        raise ValueError("No overlapping scene IDs between the two models.")

    a = np.asarray([map_a[sid] for sid in common_scene_ids], dtype=np.float64)
    b = np.asarray([map_b[sid] for sid in common_scene_ids], dtype=np.float64)

    return common_scene_ids, a, b


def cliffs_delta_paired(diff):
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    n = len(diff)
    if n == 0:
        return 0.0
    return float((pos - neg) / n)


def sample_skewness(x):
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std == 0.0:
        return 0.0
    centered = x - x.mean()
    return float(np.mean(centered ** 3) / (std ** 3))


def symmetry_warning(diff):
    diff = np.asarray(diff, dtype=np.float64)

    if len(diff) < 10:
        return {
            "n": int(len(diff)),
            "skewness": None,
            "mean_minus_median_over_std": None,
            "warning": "Too few paired samples for a useful symmetry heuristic.",
        }

    std = diff.std()
    mean = diff.mean()
    median = np.median(diff)
    skew = sample_skewness(diff)

    if std == 0.0:
        gap_ratio = 0.0
    else:
        gap_ratio = float(abs(mean - median) / std)

    warning_parts = []
    if abs(skew) > ABS_SKEW_WARNING_THRESHOLD:
        warning_parts.append(
            f"abs(skewness)={abs(skew):.3f} exceeds {ABS_SKEW_WARNING_THRESHOLD:.3f}"
        )
    if gap_ratio > MEAN_MEDIAN_GAP_STD_THRESHOLD:
        warning_parts.append(
            f"|mean-median|/std={gap_ratio:.3f} exceeds {MEAN_MEDIAN_GAP_STD_THRESHOLD:.3f}"
        )

    if warning_parts:
        warning = (
            "Paired differences may be strongly skewed / asymmetric for Wilcoxon: "
            + "; ".join(warning_parts)
            + ". Consider interpreting Wilcoxon alongside the sign test and the histogram."
        )
    else:
        warning = "No strong symmetry warning from the simple heuristics."

    return {
        "n": int(len(diff)),
        "skewness": float(skew),
        "mean_minus_median_over_std": float(gap_ratio),
        "warning": warning,
    }


def binomial_two_sided_pvalue(n_positive, n_nonzero):
    if n_nonzero == 0:
        return None

    k = min(n_positive, n_nonzero - n_positive)
    p = 2.0 * sum(comb(n_nonzero, i) for i in range(0, k + 1)) / (2 ** n_nonzero)
    return min(1.0, p)


def binomial_one_sided_pvalue_greater(n_positive, n_nonzero):
    if n_nonzero == 0:
        return None

    p = sum(comb(n_nonzero, i) for i in range(n_positive, n_nonzero + 1)) / (2 ** n_nonzero)
    return min(1.0, p)


def sign_test(diff, alternative="greater"):
    diff = np.asarray(diff, dtype=np.float64)
    nonzero = diff[diff != 0]

    n_nonzero = int(len(nonzero))
    n_positive = int(np.sum(nonzero > 0))
    n_negative = int(np.sum(nonzero < 0))

    if n_nonzero == 0:
        return {
            "n_nonzero": 0,
            "n_positive": 0,
            "n_negative": 0,
            "pvalue": None,
            "alternative": alternative,
        }

    if alternative == "greater":
        pvalue = binomial_one_sided_pvalue_greater(n_positive, n_nonzero)
    elif alternative == "two-sided":
        pvalue = binomial_two_sided_pvalue(n_positive, n_nonzero)
    else:
        raise ValueError("Sign test alternative must be 'greater' or 'two-sided'.")

    return {
        "n_nonzero": n_nonzero,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pvalue": float(pvalue),
        "alternative": alternative,
    }


def metric_display_name(metric_key):
    if metric_key == "per_scene_wmae":
        return "Per-Scene WMAE (°)"
    if metric_key == "per_scene_mse_norm":
        return "Per-Scene Normalized MSE"
    return metric_key


def metric_axis_label(metric_key):
    if metric_key == "per_scene_wmae":
        return "Weighted Mean Angular Error (°)"
    if metric_key == "per_scene_mse_norm":
        return "Normalized MSE"
    return metric_key


def metric_short_name(metric_key):
    if metric_key == "per_scene_wmae":
        return "WMAE"
    if metric_key == "per_scene_mse_norm":
        return "Normalized MSE"
    return metric_key


def format_metric_value(value, metric_key):
    if value is None:
        return "N/A"
    if metric_key == "per_scene_wmae":
        return f"{value:.6f}°"
    return f"{value:.6e}"


def safe_name(text):
    return text.lower().replace(" ", "_").replace(".", "").replace("/", "_")


def save_difference_histogram(diff, model_a_name, model_b_name, metric_key):
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 5))
    plt.hist(diff, bins=30)
    plt.axvline(0.0, linewidth=1.5)

    if metric_key == "per_scene_wmae":
        plt.xlabel(f"Paired Difference ({model_b_name} - {model_a_name}) (°)")
    else:
        plt.xlabel(f"Paired Difference ({model_b_name} - {model_a_name})")

    plt.ylabel("Count")
    plt.title(f"Paired Differences: {model_a_name} vs {model_b_name}")
    plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    plt.tight_layout()

    out_path = HIST_DIR / f"{safe_name(model_a_name)}_vs_{safe_name(model_b_name)}_{metric_key}_hist.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return out_path


def write_line(f, line=""):
    print(line)
    if f is not None:
        f.write(line + "\n")


def choose_y_axis_step(y_min: float, y_max: float) -> float:
    y_range = y_max - y_min
    if y_range <= 20:
        return 5.0
    return 10.0


def rounded_y_limits(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    step = choose_y_axis_step(y_min, y_max)
    lower = step * floor(y_min / step)
    upper = step * ceil(y_max / step)

    if lower == upper:
        upper = lower + step

    return lower, upper


# ============================================================
# GROUP PLOT HELPERS
# ============================================================

def parse_group_key(key):
    if key == "None":
        return None

    try:
        parsed = ast.literal_eval(key)
        if isinstance(parsed, list):
            return tuple(parsed)
        return parsed
    except Exception:
        pass

    try:
        return int(key)
    except Exception:
        return key


def get_group_distribution(results_json, group_name):
    group_distributions = results_json.get("group_distributions", {})
    if group_name not in group_distributions:
        return {}
    return group_distributions[group_name]


def collect_group_stats(results_json, group_name, metric_name="wmae"):
    raw_group = get_group_distribution(results_json, group_name)
    out = {}

    for key_str, group_stats in raw_group.items():
        metric_stats = group_stats.get(metric_name, {})
        parsed_key = parse_group_key(key_str)

        if metric_stats.get("count", 0) == 0:
            continue

        out[parsed_key] = {
            "label": key_str,
            "mean": metric_stats.get("mean", None),
            "std": metric_stats.get("std", None),
            "max": metric_stats.get("max", None),
            "min": metric_stats.get("min", None),
            "p50": metric_stats.get("p50", None),
            "p90": metric_stats.get("p90", None),
            "p95": metric_stats.get("p95", None),
            "p99": metric_stats.get("p99", None),
            "count": metric_stats.get("count", 0),
        }

    return out


def sort_n_event_keys(keys):
    numeric = [k for k in keys if isinstance(k, int)]
    other = [k for k in keys if not isinstance(k, int)]
    return sorted(numeric) + sorted(other, key=lambda x: str(x))


def sort_difficulty_keys(keys):
    def _difficulty_key(k):
        if isinstance(k, (int, float)):
            return float(k)
        try:
            return float(k)
        except Exception:
            return float("inf")
    return sorted(keys, key=_difficulty_key)


def snr_sort_key(k):
    if k is None:
        return (1e9, 1e9, "None")

    if isinstance(k, (list, tuple)) and len(k) >= 2:
        return (float(k[0]), float(k[1]), str(k))

    return (1e9 - 1, 1e9 - 1, str(k))


def format_snr_label(k):
    if k is None:
        return "None"
    if isinstance(k, (list, tuple)) and len(k) >= 2:
        low = int(round(k[0]))
        high = int(round(k[1]))
        return f"[{low}, {high}]"
    return str(int(round(k)))


def format_difficulty_label(k):
    if isinstance(k, float) and k.is_integer():
        return str(int(k))
    return str(k)


def load_manifest(manifest_path: Path):
    manifest = {}
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            manifest[record["scene_id"]] = record
    return manifest


_MANIFEST_GROUP_FIELD = {
    "n_events": "n_sources",
    "difficulty": "difficulty",
}


def get_scene_group_mapping(results_json, group_name):
    assignments = results_json.get("scene_group_assignments", {})
    if group_name in assignments:
        return assignments[group_name]

    mapping_key = f"scene_{group_name}"
    if mapping_key in results_json:
        return results_json[mapping_key]

    if MANIFEST_JSONL.exists():
        manifest_field = _MANIFEST_GROUP_FIELD.get(group_name, group_name)
        manifest = load_manifest(MANIFEST_JSONL)
        scene_ids = set(results_json.get("per_scene_wmae", {}).keys())
        mapping = {}
        for sid in scene_ids:
            if sid in manifest and manifest_field in manifest[sid]:
                mapping[sid] = manifest[sid][manifest_field]
        if mapping:
            return mapping

    return {}


def plot_group_metric_with_std(model_jsons, group_name, output_path, metric_name="wmae"):
    metric_title = metric_short_name(METRIC_KEY)

    if group_name == "n_events":
        title = f"{metric_title} by Number of Sources {PLOT_TITLE_SUFFIX}"
        x_label = "Number of Sources in a Scene"
        get_sorted_keys = sort_n_event_keys
        tick_formatter = lambda k: str(k)
    elif group_name == "snr_bucket":
        title = f"{metric_title} by SNR Bucket {PLOT_TITLE_SUFFIX}"
        x_label = "SNR Bucket (dB)"
        get_sorted_keys = lambda keys: sorted(keys, key=snr_sort_key)
        tick_formatter = format_snr_label
    elif group_name == "difficulty":
        title = f"{metric_title} by Difficulty Score {PLOT_TITLE_SUFFIX}"
        x_label = "Difficulty Score"
        get_sorted_keys = sort_difficulty_keys
        tick_formatter = format_difficulty_label
    else:
        raise ValueError(f"Unsupported group_name: {group_name}")

    model_group_stats = []
    all_keys = set()

    for model_json in model_jsons:
        stats = collect_group_stats(model_json, group_name, metric_name=metric_name)
        if stats:
            model_group_stats.append((model_json.get("model_name", "Unknown Model"), stats))
            all_keys.update(stats.keys())

    if not model_group_stats:
        return None

    sorted_keys = get_sorted_keys(list(all_keys))
    x_base = np.arange(len(sorted_keys), dtype=np.float64)

    fig = plt.figure(figsize=FIGSIZE)

    plotted_values = []

    for m_idx, (model_name, stats) in enumerate(model_group_stats):
        xs = []
        means = []
        stds = []
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]

        for idx, key in enumerate(sorted_keys):
            if key not in stats:
                continue

            mean = stats[key]["mean"]
            std = stats[key]["std"]

            if mean is None or std is None:
                continue

            xs.append(x_base[idx])
            means.append(mean)
            stds.append(std)
            plotted_values.extend([mean - std, mean + std])

        if not xs:
            continue

        # plt.errorbar(
        #     xs,
        #     means,
        #     yerr=stds,
        #     marker=MARKER_MEAN,
        #     linestyle=LINESTYLE,
        #     linewidth=LINEWIDTH,
        #     capsize=CAPSIZE,
        #     color=color,
        #     label=model_name,
        # )

        plt.plot(
                xs,
                means,
                marker=MARKER_MEAN,
                linestyle=LINESTYLE,
                linewidth=LINEWIDTH,
                color=color,
                label=model_name,
            )

    y_limits = rounded_y_limits(plotted_values)
    if y_limits is not None:
        plt.ylim(*y_limits)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric_axis_label(METRIC_KEY))
    plt.xticks(
        x_base,
        [tick_formatter(k) for k in sorted_keys],
        rotation=0,
    )
    plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    return output_path


def plot_group_metric_boxplot(model_jsons, group_name, output_path):
    metric_title = metric_short_name(METRIC_KEY)

    if group_name == "n_events":
        title = f"{metric_title} Distribution by Number of Sources {PLOT_TITLE_SUFFIX}"
        x_label = "Number of Sources in a Scene"
        get_sorted_keys = sort_n_event_keys
        tick_formatter = lambda k: str(k)
    elif group_name == "snr_bucket":
        title = f"{metric_title} Distribution by SNR Bucket {PLOT_TITLE_SUFFIX}"
        x_label = "SNR Bucket"
        get_sorted_keys = lambda keys: sorted(keys, key=snr_sort_key)
        tick_formatter = format_snr_label
    else:
        raise ValueError(f"Unsupported group_name: {group_name}")

    all_keys = set()
    model_data = []
    use_raw = True

    for model_json in model_jsons:
        model_name = model_json.get("model_name", "Unknown Model")
        per_scene = model_json.get(METRIC_KEY, {})
        group_map = get_scene_group_mapping(model_json, group_name)
        stats = collect_group_stats(model_json, group_name, metric_name="wmae")

        if group_map and per_scene:
            grouped_raw = {}
            for scene_id, metric_val in per_scene.items():
                if scene_id in group_map:
                    key = parse_group_key(str(group_map[scene_id]))
                    grouped_raw.setdefault(key, []).append(metric_val)
            model_data.append((model_name, grouped_raw, stats))
            all_keys.update(grouped_raw.keys())
        elif stats:
            use_raw = False
            model_data.append((model_name, None, stats))
            all_keys.update(stats.keys())
        else:
            continue

    if not model_data:
        return None

    if not use_raw:
        print(
            "  WARNING [boxplot]: Manifest not found or scene_group_assignments "
            "missing — falling back to summary-stat box plot."
        )

    sorted_keys = get_sorted_keys(list(all_keys))
    n_models = len(model_data)
    n_groups = len(sorted_keys)

    box_width = 0.6 / n_models
    x_base = np.arange(n_groups, dtype=np.float64)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for m_idx, (model_name, grouped_raw, stats) in enumerate(model_data):
        offset = (m_idx - (n_models - 1) / 2.0) * box_width
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]

        if use_raw and grouped_raw is not None:
            positions = []
            data = []

            for g_idx, key in enumerate(sorted_keys):
                if key in grouped_raw and len(grouped_raw[key]) > 0:
                    positions.append(x_base[g_idx] + offset)
                    data.append(grouped_raw[key])

            if not data:
                continue

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=True,
                showmeans=True,
                meanline=True,
                meanprops=dict(
                    color=color,
                    linestyle=":",
                    linewidth=LINEWIDTH * 0.8,
                ),
                manage_ticks=False,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_linewidth(LINEWIDTH * 0.6)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_linewidth(LINEWIDTH * 0.6)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(LINEWIDTH * 0.7)
            for flier in bp["fliers"]:
                flier.set_markersize(3)

        else:
            bxp_stats = []
            positions = []
            synth_means = []

            for g_idx, key in enumerate(sorted_keys):
                if key not in stats:
                    continue
                s = stats[key]
                mean = s.get("mean")
                std = s.get("std")
                med = s.get("p50")
                mn = s.get("min")
                mx = s.get("max")

                if any(v is None for v in [mean, std, med, mn, mx]):
                    continue

                bxp_stats.append({
                    "whislo": mn,
                    "q1": max(mn, mean - std),
                    "med": med,
                    "q3": min(mx, mean + std),
                    "whishi": mx,
                    "fliers": [],
                })
                positions.append(x_base[g_idx] + offset)
                synth_means.append(mean)

            if not bxp_stats:
                continue

            bp = ax.bxp(
                bxp_stats,
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_linewidth(LINEWIDTH * 0.6)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_linewidth(LINEWIDTH * 0.6)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(LINEWIDTH * 0.7)

            half_w = box_width * 0.85 / 2.0
            for pos, m_val in zip(positions, synth_means):
                ax.hlines(
                    m_val,
                    pos - half_w,
                    pos + half_w,
                    colors=color,
                    linestyles=":",
                    linewidth=LINEWIDTH * 0.8,
                )

        ax.plot([], [], color=color, linewidth=LINEWIDTH, label=model_name)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_axis_label(METRIC_KEY))
    ax.set_xticks(x_base)
    ax.set_xticklabels([tick_formatter(k) for k in sorted_keys])
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="grey", linewidth=LINEWIDTH * 0.8, linestyle=":"))
    labels.append("Mean")
    handles.append(Line2D([0], [0], marker="o", markeredgecolor="black", markerfacecolor="none", linestyle="None", markersize=3))
    labels.append("Outliers")
    ax.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_path


# ============================================================
# WMAE vs N SOURCES PER SNR BUCKET (rotation model)
# ============================================================

def plot_wmae_by_nsources_per_snr_bucket(rotation_json, output_path):
    """
    Line plot of mean WMAE vs number of sources, with one line per SNR bucket.
    Uses the rotation model only. Reads n_sources and snr_bucket directly
    from the manifest file.
    """
    from collections import defaultdict

    manifest_path = MANIFEST_JSONL
    if not manifest_path.exists():
        print(f"WARNING [snr_nsources plot]: manifest not found at {manifest_path}. Skipping.")
        return None

    manifest = load_manifest(manifest_path)

    per_scene_wmae = rotation_json.get(METRIC_KEY, {})
    if not per_scene_wmae:
        print("WARNING [snr_nsources plot]: no per-scene WMAE found. Skipping.")
        return None

    # group wmae values by (snr_bucket, n_sources)
    cell_data = defaultdict(list)
    for scene_id, wmae_val in per_scene_wmae.items():
        rec = manifest.get(scene_id)
        if rec is None:
            continue
        n = rec.get("n_sources")
        snr = rec.get("snr_bucket")
        if n is None or snr is None:
            continue
        snr_key = tuple(snr) if isinstance(snr, list) else snr
        cell_data[(snr_key, n)].append(wmae_val)

    if not cell_data:
        print("WARNING [snr_nsources plot]: no data after grouping. Skipping.")
        return None

    snr_buckets = sorted(
        set(k[0] for k in cell_data),
        key=lambda x: x[0] if isinstance(x, tuple) else float(x),
    )
    n_sources_all = sorted(set(k[1] for k in cell_data))
    x_base = np.arange(len(n_sources_all), dtype=np.float64)

    cmap = plt.cm.get_cmap("viridis", len(snr_buckets))
    colors = [cmap(i) for i in range(len(snr_buckets))]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for idx, snr in enumerate(snr_buckets):
        means = []
        xs = []
        for n_idx, n in enumerate(n_sources_all):
            vals = cell_data.get((snr, n), [])
            if vals:
                means.append(float(np.mean(vals)))
                xs.append(x_base[n_idx])

        if not xs:
            continue

        label = format_snr_label(list(snr) if isinstance(snr, tuple) else snr)
        ax.plot(
            xs,
            means,
            marker=MARKER_MEAN,
            linestyle=LINESTYLE,
            linewidth=LINEWIDTH,
            color=colors[idx],
            label=f"SNR {label} dB",
        )

    ax.set_title(
        f"{metric_short_name(METRIC_KEY)} vs Number of Sources by SNR Bucket "
        f"{PLOT_TITLE_SUFFIX}"
    )
    ax.set_xlabel("Number of Sources in a Scene")
    ax.set_ylabel(metric_axis_label(METRIC_KEY))
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(n) for n in n_sources_all])
    ax.legend(title="SNR Bucket", loc="upper left")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    return output_path


# ============================================================
# AMBIGUITY / LOCALISABILITY ANALYSIS
# ============================================================

def load_scene_positions_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def classify_scene_ambiguity(csv_rows):
    scene_events = {}
    for row in csv_rows:
        sid = row["scene_id"]
        ambig = row["front_back_ambiguous"].strip().lower() == "true"
        scene_events.setdefault(sid, []).append(ambig)

    mapping = {}
    for sid, flags in scene_events.items():
        n_ambig = sum(flags)
        if n_ambig > 0:
            mapping[sid] = "Has Ambiguous"
        else:
            mapping[sid] = "Fully Localisable"

    return mapping


_AMBIGUITY_ORDER = ["Fully Localisable", "Has Ambiguous"]


def plot_ambiguity_boxplot(model_jsons, scene_ambiguity_map, output_path, exp_label=""):
    metric_title = metric_short_name(METRIC_KEY)

    all_categories = set()
    model_data = []

    for model_json in model_jsons:
        model_name = model_json.get("model_name", "Unknown Model")
        per_scene = model_json.get(METRIC_KEY, {})

        grouped = {}
        for scene_id, metric_val in per_scene.items():
            if scene_id in scene_ambiguity_map:
                cat = scene_ambiguity_map[scene_id]
                grouped.setdefault(cat, []).append(metric_val)

        if grouped:
            model_data.append((model_name, grouped))
            all_categories.update(grouped.keys())

    if not model_data:
        return None

    sorted_cats = [c for c in _AMBIGUITY_ORDER if c in all_categories]
    n_models = len(model_data)
    n_groups = len(sorted_cats)

    box_width = 0.6 / n_models
    x_base = np.arange(n_groups, dtype=np.float64)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for m_idx, (model_name, grouped) in enumerate(model_data):
        offset = (m_idx - (n_models - 1) / 2.0) * box_width
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]

        positions = []
        data = []
        for g_idx, cat in enumerate(sorted_cats):
            if cat in grouped and len(grouped[cat]) > 0:
                positions.append(x_base[g_idx] + offset)
                data.append(grouped[cat])

        if not data:
            continue

        if AMBIGUITY_PLOT_TYPE == "violin":
            vp = ax.violinplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                showmedians=True,
                showextrema=True,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.5)
                body.set_linewidth(LINEWIDTH * 0.6)
            for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
                vp[part].set_color(color)
                vp[part].set_linewidth(LINEWIDTH * 0.7)
            vp["cmedians"].set_color("black")
            for pos, d in zip(positions, data):
                ax.plot(pos, np.mean(d), marker="o", color=color,
                        markersize=4, zorder=3)
        else:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=True,
                showmeans=True,
                meanline=True,
                meanprops=dict(
                    color=color,
                    linestyle=":",
                    linewidth=LINEWIDTH * 0.8,
                ),
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_linewidth(LINEWIDTH * 0.6)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_linewidth(LINEWIDTH * 0.6)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(LINEWIDTH * 0.7)
            for flier in bp["fliers"]:
                flier.set_markersize(3)

        ax.plot([], [], color=color, linewidth=LINEWIDTH, label=model_name)

    title = f"{metric_title} by Front-Back Ambiguity {exp_label}" if exp_label else f"{metric_title} by Front-Back Ambiguity {PLOT_TITLE_SUFFIX}"
    ax.set_title(title)
    ax.set_xlabel("Scene Ambiguity Category")
    ax.set_ylabel(metric_axis_label(METRIC_KEY))
    ax.set_xticks(x_base)

    count_labels = []
    for cat in sorted_cats:
        n = len(model_data[0][1].get(cat, []))
        count_labels.append(f"{cat}\n(n={n})")
    ax.set_xticklabels(count_labels)

    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=LINEWIDTH * 0.7, linestyle="-"))
    labels.append("Median")
    if AMBIGUITY_PLOT_TYPE == "boxplot":
        handles.append(Line2D([0], [0], color="grey", linewidth=LINEWIDTH * 0.8, linestyle=":"))
        labels.append("Mean")
        handles.append(Line2D([0], [0], marker="o", markeredgecolor="black", markerfacecolor="none", linestyle="None", markersize=3))
        labels.append("Outliers")
    else:
        handles.append(Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=4))
        labels.append("Mean")
    ax.legend(handles, labels, loc="upper left",)# bbox_to_anchor=(0.21, 0.98))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_ambiguity_boxplot_sidebyside(model_jsons_1, scene_ambiguity_map_1,
                                      model_jsons_2, scene_ambiguity_map_2,
                                      output_path):
    """Create single plot with all models: no_bg on left, bg on right."""
    metric_title = metric_short_name(METRIC_KEY)

    # Prepare data for experiment 1 (no bg)
    all_categories_1 = set()
    model_data_1 = []
    for model_json in model_jsons_1:
        model_name = model_json.get("model_name", "Unknown Model")
        per_scene = model_json.get(METRIC_KEY, {})
        grouped = {}
        for scene_id, metric_val in per_scene.items():
            if scene_id in scene_ambiguity_map_1:
                cat = scene_ambiguity_map_1[scene_id]
                grouped.setdefault(cat, []).append(metric_val)
        if grouped:
            model_data_1.append((model_name, grouped))
            all_categories_1.update(grouped.keys())

    # Prepare data for experiment 2 (bg)
    all_categories_2 = set()
    model_data_2 = []
    for model_json in model_jsons_2:
        model_name = model_json.get("model_name", "Unknown Model")
        per_scene = model_json.get(METRIC_KEY, {})
        grouped = {}
        for scene_id, metric_val in per_scene.items():
            if scene_id in scene_ambiguity_map_2:
                cat = scene_ambiguity_map_2[scene_id]
                grouped.setdefault(cat, []).append(metric_val)
        if grouped:
            model_data_2.append((model_name, grouped))
            all_categories_2.update(grouped.keys())

    if not model_data_1 or not model_data_2:
        return None

    # Use common categories
    sorted_cats = [c for c in _AMBIGUITY_ORDER if c in (all_categories_1 | all_categories_2)]
    n_groups = len(sorted_cats)
    n_models_per_exp = max(len(model_data_1), len(model_data_2))

    # Layout: [no_bg_cat1] [no_bg_cat2] [bg_cat1] [bg_cat2]
    # Much wider spacing between experiments
    box_width = 0.8 / n_models_per_exp

    # Create positions for each category pair
    x_ticks = []
    x_labels = []

    # No background group (left): positions 0-1 for localisable, 2-3 for ambiguous
    # Background group (right): positions 6-7 for localisable, 8-9 for ambiguous
    for g_idx, cat in enumerate(sorted_cats):
        # NO_BG positions
        no_bg_base = g_idx * 2
        x_ticks.append(no_bg_base + 0.5)
        x_labels.append(f"{cat}\n(No BG)")

        # BG positions (with big gap after NO_BG group)
        bg_base = 4 + g_idx * 2
        x_ticks.append(bg_base + 0.5)
        x_labels.append(f"{cat}\n(W/BG)")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot experiment 1 (no bg) models FIRST (left side)
    exp1_colors = ["tab:green", "tab:red"]  # Rotation, Static for exp1
    for m_idx, (model_name, grouped) in enumerate(model_data_1):
        color = exp1_colors[m_idx % len(exp1_colors)]

        positions = []
        data = []
        for g_idx, cat in enumerate(sorted_cats):
            if cat in grouped and len(grouped[cat]) > 0:
                no_bg_base = g_idx * 2
                # Spread models within no_bg region (centered at 0.5)
                offset = (m_idx - (n_models_per_exp - 1) / 2.0) * box_width
                positions.append(no_bg_base + 0.5 + offset)
                data.append(grouped[cat])

        if not data:
            continue

        if AMBIGUITY_PLOT_TYPE == "violin":
            vp = ax.violinplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                showmedians=True,
                showextrema=True,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.5)
                body.set_linewidth(LINEWIDTH * 0.6)
            for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
                vp[part].set_color(color)
                vp[part].set_linewidth(LINEWIDTH * 0.7)
            vp["cmedians"].set_color("black")
            for pos, d in zip(positions, data):
                ax.plot(pos, np.mean(d), marker="o", color=color,
                        markersize=4, zorder=3)
        else:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=True,
                showmeans=True,
                meanline=True,
                meanprops=dict(
                    color=color,
                    linestyle=":",
                    linewidth=LINEWIDTH * 0.8,
                ),
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_linewidth(LINEWIDTH * 0.6)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_linewidth(LINEWIDTH * 0.6)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(LINEWIDTH * 0.7)
            for flier in bp["fliers"]:
                flier.set_markersize(3)

        ax.plot([], [], color=color, linewidth=LINEWIDTH, label=f"{model_name} (NO BG)")

    # Plot experiment 2 (bg) models SECOND (right side)
    exp2_colors = ["tab:blue", "tab:orange"]  # Rotation, Static for exp2
    for m_idx, (model_name, grouped) in enumerate(model_data_2):
        color = exp2_colors[m_idx % len(exp2_colors)]

        positions = []
        data = []
        for g_idx, cat in enumerate(sorted_cats):
            if cat in grouped and len(grouped[cat]) > 0:
                bg_base = 4 + g_idx * 2
                # Spread models within bg region (centered at 0.5)
                offset = (m_idx - (n_models_per_exp - 1) / 2.0) * box_width
                positions.append(bg_base + 0.5 + offset)
                data.append(grouped[cat])

        if not data:
            continue

        if AMBIGUITY_PLOT_TYPE == "violin":
            vp = ax.violinplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                showmedians=True,
                showextrema=True,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.5)
                body.set_linewidth(LINEWIDTH * 0.6)
            for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
                vp[part].set_color(color)
                vp[part].set_linewidth(LINEWIDTH * 0.7)
            vp["cmedians"].set_color("black")
            for pos, d in zip(positions, data):
                ax.plot(pos, np.mean(d), marker="o", color=color,
                        markersize=4, zorder=3)
        else:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=True,
                showmeans=True,
                meanline=True,
                meanprops=dict(
                    color=color,
                    linestyle=":",
                    linewidth=LINEWIDTH * 0.8,
                ),
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_linewidth(LINEWIDTH * 0.6)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_linewidth(LINEWIDTH * 0.6)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(LINEWIDTH * 0.7)
            for flier in bp["fliers"]:
                flier.set_markersize(3)

        ax.plot([], [], color=color, linewidth=LINEWIDTH, label=f"{model_name} (W/BG)")

    ax.set_title(f"{metric_title} by Front-Back Ambiguity (Synthetic HT)")
    ax.set_xlabel("Scene Ambiguity Category")
    ax.set_ylabel(metric_axis_label(METRIC_KEY))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=LINEWIDTH * 0.7, linestyle="-"))
    labels.append("Median")
    if AMBIGUITY_PLOT_TYPE == "boxplot":
        handles.append(Line2D([0], [0], color="grey", linewidth=LINEWIDTH * 0.8, linestyle=":"))
        labels.append("Mean")
        handles.append(Line2D([0], [0], marker="o", markeredgecolor="black", markerfacecolor="none", linestyle="None", markersize=3))
        labels.append("Outliers")
    else:
        handles.append(Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=4))
        labels.append("Mean")
    ax.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_path



# ============================================================
# FAILURE RATE VS THRESHOLD PLOT
# ============================================================

def plot_failure_rate_vs_threshold(model_jsons, output_path):
    """
    ROC-style plot: for each WMAE threshold, what fraction of scenes
    exceed it (catastrophic failure rate)? Plotted for all enabled models.
    A single summary number — area under the curve — is printed per model.
    """
    thresholds = np.linspace(0, 90, 500)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for m_idx, model_json in enumerate(model_jsons):
        model_name = model_json.get("model_name", "Unknown Model")
        per_scene = np.array(list(model_json.get(METRIC_KEY, {}).values()))
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]

        failure_rates = np.array([np.mean(per_scene > t) for t in thresholds])
        auc = np.trapz(failure_rates, thresholds) / 90.0  # normalised to [0,1]

        ax.plot(thresholds, failure_rates * 100,
                color=color, linewidth=LINEWIDTH,
                label=f"{model_name}  (nAUC={auc:.3f})")
        print(f"  {model_name}: nAUC={auc:.4f}")

    ax.set_xlabel("WMAE Threshold (°)")
    ax.set_ylabel("Scenes Exceeding Threshold (%)")
    ax.set_title(f"Catastrophic Failure Rate vs WMAE Threshold {PLOT_TITLE_SUFFIX}")
    ax.axvline(45, color="grey", linestyle=":", linewidth=1.2,
               label="45° (hemisphere confusion)")
    ax.axvline(90, color="grey", linestyle="--", linewidth=1.2,
               label="90°")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================
# N_SOURCES × AMBIGUITY HEATMAP
# ============================================================

def plot_nsources_ambiguity_heatmap(model_jsons, scene_ambiguity_map, output_path):
    """
    Heatmap of mean ΔWMAE (static − rotation) with n_sources on x-axis
    and ambiguity category on y-axis. Requires exactly two models:
    [rotation, static].
    """
    if len(model_jsons) < 2:
        print("WARNING: heatmap requires at least 2 models (rotation + static). Skipping.")
        return None

    rot_json = model_jsons[0]
    sta_json = model_jsons[1]

    rot_wmae = rot_json.get(METRIC_KEY, {})
    sta_wmae = sta_json.get(METRIC_KEY, {})

    # Build scene → n_sources mapping from manifest
    if not MANIFEST_JSONL.exists():
        print(f"WARNING: Manifest not found at {MANIFEST_JSONL}. Skipping heatmap.")
        return None

    manifest = load_manifest(MANIFEST_JSONL)
    scene_nsources = {sid: rec["n_sources"] for sid, rec in manifest.items()
                      if "n_sources" in rec}

    # Accumulate ΔWMAE per (n_sources, ambiguity) cell
    from collections import defaultdict
    cell_diffs = defaultdict(list)

    for scene_id in set(rot_wmae.keys()) & set(sta_wmae.keys()):
        if scene_id not in scene_ambiguity_map or scene_id not in scene_nsources:
            continue
        diff = sta_wmae[scene_id] - rot_wmae[scene_id]
        n = scene_nsources[scene_id]
        cat = scene_ambiguity_map[scene_id]
        cell_diffs[(n, cat)].append(diff)

    all_n = sorted(set(k[0] for k in cell_diffs))
    all_cats = [c for c in _AMBIGUITY_ORDER if c in set(k[1] for k in cell_diffs)]

    # Build matrix
    matrix = np.full((len(all_cats), len(all_n)), np.nan)
    annot = np.empty((len(all_cats), len(all_n)), dtype=object)

    for i, cat in enumerate(all_cats):
        for j, n in enumerate(all_n):
            vals = cell_diffs.get((n, cat), [])
            if vals:
                matrix[i, j] = np.mean(vals)
                annot[i, j] = f"{np.mean(vals):.1f}\n(n={len(vals)})"
            else:
                annot[i, j] = ""

    vmax = np.nanmax(np.abs(matrix))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ΔWMAE  (Static − Rotation)  [°]", fontsize=10)

    ax.set_xticks(range(len(all_n)))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.set_yticks(range(len(all_cats)))
    ax.set_yticklabels(all_cats)
    ax.set_xlabel("Number of Sources")
    ax.set_ylabel("Ambiguity Category")
    ax.set_title(f"Mean ΔWMAE by N Sources & Ambiguity {PLOT_TITLE_SUFFIX}\n(green = rotation better, red = static better)")

    for i in range(len(all_cats)):
        for j in range(len(all_n)):
            if annot[i, j]:
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================
# COMPARISON
# ============================================================

def summarise_pairwise_comparison(model_a_json, model_b_json, metric_key):
    name_a = model_a_json.get("model_name", "Model A")
    name_b = model_b_json.get("model_name", "Model B")
    split_a = model_a_json.get("split", "unknown")
    split_b = model_b_json.get("split", "unknown")

    if split_a != split_b:
        raise ValueError(
            f"Split mismatch: {name_a} uses '{split_a}' but {name_b} uses '{split_b}'"
        )

    map_a = extract_metric_map(model_a_json, metric_key)
    map_b = extract_metric_map(model_b_json, metric_key)

    common_scene_ids, a, b = align_scene_metrics(map_a, map_b)
    diff = b - a

    n_improved = int(np.sum(diff > 0))
    n_worsened = int(np.sum(diff < 0))
    n_tied = int(np.sum(diff == 0))

    if np.allclose(diff, 0.0):
        wilcoxon_stat = None
        wilcoxon_p = None
    else:
        result = wilcoxon(
            diff,
            zero_method=WILCOXON_ZERO_METHOD,
            alternative=WILCOXON_ALTERNATIVE,
        )
        wilcoxon_stat = float(result.statistic)
        wilcoxon_p = float(result.pvalue)

    sign_result = sign_test(diff, alternative=WILCOXON_ALTERNATIVE)
    sym_result = symmetry_warning(diff)

    hist_path = None
    if SAVE_DIFF_HISTOGRAMS:
        hist_path = save_difference_histogram(diff, name_a, name_b, metric_key)

    summary = {
        "comparison": f"{name_a} vs {name_b}",
        "model_a_name": name_a,
        "model_b_name": name_b,
        "split": split_a,
        "metric_key": metric_key,
        "n_common_scenes": int(len(common_scene_ids)),
        "mean_model_a": float(a.mean()),
        "std_model_a": float(a.std()),
        "mean_model_b": float(b.mean()),
        "std_model_b": float(b.std()),
        "mean_paired_improvement": float(diff.mean()),
        "median_paired_improvement": float(np.median(diff)),
        "std_paired_improvement": float(diff.std()),
        "p50_improvement": float(np.percentile(diff, 50)),
        "p90_improvement": float(np.percentile(diff, 90)),
        "p95_improvement": float(np.percentile(diff, 95)),
        "p99_improvement": float(np.percentile(diff, 99)),
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "n_tied": n_tied,
        "percent_improved": float(100.0 * n_improved / len(diff)),
        "percent_worsened": float(100.0 * n_worsened / len(diff)),
        "percent_tied": float(100.0 * n_tied / len(diff)),
        "effect_sign_fraction": cliffs_delta_paired(diff),
        "wilcoxon_alternative": WILCOXON_ALTERNATIVE,
        "wilcoxon_zero_method": WILCOXON_ZERO_METHOD,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "sign_test": sign_result,
        "symmetry_check": sym_result,
        "histogram_path": str(hist_path) if hist_path is not None else None,
    }

    return summary


def print_summary(summary, f=None):
    metric_key = summary["metric_key"]

    write_line(f, "=" * 72)
    write_line(f, f"Comparison: {summary['comparison']}")
    write_line(f, f"Split: {summary['split']}")
    write_line(f, f"Metric: {metric_display_name(metric_key)}")
    write_line(f, f"Common Scenes: {summary['n_common_scenes']}")
    write_line(f)

    write_line(
        f,
        f"Model A Mean: {format_metric_value(summary['mean_model_a'], metric_key)} | "
        f"Std: {format_metric_value(summary['std_model_a'], metric_key)}",
    )
    write_line(
        f,
        f"Model B Mean: {format_metric_value(summary['mean_model_b'], metric_key)} | "
        f"Std: {format_metric_value(summary['std_model_b'], metric_key)}",
    )
    write_line(f)

    write_line(
        f,
        f"Mean Paired Improvement (B - A):   "
        f"{format_metric_value(summary['mean_paired_improvement'], metric_key)}"
    )
    write_line(
        f,
        f"Median Paired Improvement (B - A): "
        f"{format_metric_value(summary['median_paired_improvement'], metric_key)}"
    )
    write_line(
        f,
        f"Std Paired Improvement:            "
        f"{format_metric_value(summary['std_paired_improvement'], metric_key)}"
    )
    write_line(
        f,
        f"P50 Improvement: {format_metric_value(summary['p50_improvement'], metric_key)} | "
        f"P90: {format_metric_value(summary['p90_improvement'], metric_key)} | "
        f"P95: {format_metric_value(summary['p95_improvement'], metric_key)} | "
        f"P99: {format_metric_value(summary['p99_improvement'], metric_key)}"
    )
    write_line(f)

    write_line(f, f"Scenes Improved: {summary['n_improved']} ({summary['percent_improved']:.2f}%)")
    write_line(f, f"Scenes Worsened: {summary['n_worsened']} ({summary['percent_worsened']:.2f}%)")
    write_line(f, f"Scenes Tied:     {summary['n_tied']} ({summary['percent_tied']:.2f}%)")
    write_line(f, f"Effect Sign Fraction: {summary['effect_sign_fraction']:.6f}")
    write_line(f)

    if summary["wilcoxon_pvalue"] is None:
        write_line(f, "Wilcoxon Test: all paired differences are zero; no test performed.")
    else:
        write_line(f, f"Wilcoxon Alternative: {summary['wilcoxon_alternative']}")
        write_line(f, f"Wilcoxon Zero Method: {summary['wilcoxon_zero_method']}")
        write_line(f, f"Wilcoxon Statistic:   {summary['wilcoxon_statistic']:.6f}")
        write_line(f, f"Wilcoxon p-value:     {summary['wilcoxon_pvalue']:.6e}")
    write_line(f)

    sign_test_result = summary["sign_test"]
    if sign_test_result["pvalue"] is None:
        write_line(f, "Sign Test: all paired differences are zero; no test performed.")
    else:
        write_line(f, f"Sign Test Alternative: {sign_test_result['alternative']}")
        write_line(f, f"Sign Test Nonzero Pairs: {sign_test_result['n_nonzero']}")
        write_line(f, f"Sign Test Positive:      {sign_test_result['n_positive']}")
        write_line(f, f"Sign Test Negative:      {sign_test_result['n_negative']}")
        write_line(f, f"Sign Test p-value:       {sign_test_result['pvalue']:.6e}")
    write_line(f)

    sym = summary["symmetry_check"]
    write_line(f, "Symmetry Check for Paired Differences:")
    if sym["skewness"] is None:
        write_line(f, f"  {sym['warning']}")
    else:
        write_line(f, f"  Skewness: {sym['skewness']:.6f}")
        write_line(f, f"  |Mean - Median| / Std: {sym['mean_minus_median_over_std']:.6f}")
        write_line(f, f"  {sym['warning']}")

    if summary["histogram_path"] is not None:
        write_line(f, f"Histogram: {summary['histogram_path']}")


# ============================================================
# MAIN
# ============================================================

def main():
    if COMPARE_TWO_EXPERIMENTS:
        # Load both experiments and create side-by-side plots
        model_jsons_1 = load_enabled_model_jsons_for_exp(EXP_DIR, ROTATION_JSON, STATIC_JSON, NO_ROT_JSON)
        model_jsons_2 = load_enabled_model_jsons_for_exp(EXP_DIR_2, ROTATION_JSON_2, STATIC_JSON_2, NO_ROT_JSON_2)

        # Create side-by-side ambiguity plot
        if SAVE_AMBIGUITY_BOXPLOT and SCENE_POSITIONS_CSV.exists() and SCENE_POSITIONS_CSV_2.exists():
            csv_rows_1 = load_scene_positions_csv(SCENE_POSITIONS_CSV)
            scene_ambiguity_map_1 = classify_scene_ambiguity(csv_rows_1)
            exp_suffix_1 = experiment_background_suffix(EXP_DIR)

            csv_rows_2 = load_scene_positions_csv(SCENE_POSITIONS_CSV_2)
            scene_ambiguity_map_2 = classify_scene_ambiguity(csv_rows_2)
            exp_suffix_2 = experiment_background_suffix(EXP_DIR_2)

            plot_ambiguity_boxplot_sidebyside(
                model_jsons_1=model_jsons_1,
                scene_ambiguity_map_1=scene_ambiguity_map_1,
                model_jsons_2=model_jsons_2,
                scene_ambiguity_map_2=scene_ambiguity_map_2,
                output_path=PLOT_DIR / "wmae_vs_ambiguity_sidebyside_boxplot.pdf",
            )

            print(f"✓ Side-by-side ambiguity plot created: wmae_vs_ambiguity_sidebyside_boxplot.pdf")

            if SAVE_NSOURCES_AMBIGUITY_HEATMAP:
                plot_nsources_ambiguity_heatmap(
                    model_jsons=model_jsons_1,
                    scene_ambiguity_map=scene_ambiguity_map_1,
                    output_path=PLOT_DIR / "delta_wmae_nsources_ambiguity_heatmap_exp1.pdf",
                )
                plot_nsources_ambiguity_heatmap(
                    model_jsons=model_jsons_2,
                    scene_ambiguity_map=scene_ambiguity_map_2,
                    output_path=PLOT_DIR / "delta_wmae_nsources_ambiguity_heatmap_exp2.pdf",
                )
                print(f"✓ N-sources ambiguity heatmaps saved.")

        elif SAVE_AMBIGUITY_BOXPLOT:
            print(f"WARNING: Missing CSV files for side-by-side comparison")

        if SAVE_FAILURE_RATE_PLOT:
            plot_failure_rate_vs_threshold(
                model_jsons=model_jsons_1,
                output_path=PLOT_DIR / "failure_rate_vs_threshold_exp1.pdf",
            )
            plot_failure_rate_vs_threshold(
                model_jsons=model_jsons_2,
                output_path=PLOT_DIR / "failure_rate_vs_threshold_exp2.pdf",
            )
            print(f"✓ Failure rate plots saved.")
    else:
        # Original single-experiment mode
        model_jsons = load_enabled_model_jsons()
        summaries = []

        for i in range(len(model_jsons)):
            for j in range(i + 1, len(model_jsons)):
                summary = summarise_pairwise_comparison(
                    model_jsons[i],
                    model_jsons[j],
                    METRIC_KEY,
                )
                summaries.append(summary)

        n_events_plot_path = None
        snr_plot_path = None
        difficulty_plot_path = None
        boxplot_path = None
        ambiguity_plot_path = None
        snr_nsources_plot_path = None

        if SAVE_GROUP_PLOTS:
            n_events_plot_path = plot_group_metric_with_std(
                model_jsons=model_jsons,
                group_name="n_events",
                output_path=N_EVENTS_PLOT,
                metric_name="wmae",
            )

            snr_plot_path = plot_group_metric_with_std(
                model_jsons=model_jsons,
                group_name="snr_bucket",
                output_path=SNR_PLOT,
                metric_name="wmae",
            )

            difficulty_plot_path = plot_group_metric_with_std(
                model_jsons=model_jsons,
                group_name="difficulty",
                output_path=DIFFICULTY_PLOT,
                metric_name="wmae",
            )

            boxplot_path = plot_group_metric_boxplot(
                model_jsons=model_jsons,
                group_name="n_events",
                output_path=BOXPLOT_N_EVENTS,
            )

            if MANIFEST_JSONL.exists():
                snr_nsources_plot_path = plot_wmae_by_nsources_per_snr_bucket(
                    rotation_json=model_jsons[0],
                    output_path=SNR_NSOURCES_PLOT,
                )

        if SAVE_AMBIGUITY_BOXPLOT and SCENE_POSITIONS_CSV.exists():
            csv_rows = load_scene_positions_csv(SCENE_POSITIONS_CSV)
            scene_ambiguity_map = classify_scene_ambiguity(csv_rows)
            ambiguity_plot_path = plot_ambiguity_boxplot(
                model_jsons=model_jsons,
                scene_ambiguity_map=scene_ambiguity_map,
                output_path=BOXPLOT_AMBIGUITY,
            )
            if SAVE_NSOURCES_AMBIGUITY_HEATMAP:
                plot_nsources_ambiguity_heatmap(
                    model_jsons=model_jsons,
                    scene_ambiguity_map=scene_ambiguity_map,
                    output_path=NSOURCES_AMBIGUITY_HEATMAP,
                )
                print(f"✓ N-sources ambiguity heatmap saved.")
        elif SAVE_AMBIGUITY_BOXPLOT:
            print(
                f"WARNING: SAVE_AMBIGUITY_BOXPLOT is True but "
                f"SCENE_POSITIONS_CSV not found at {SCENE_POSITIONS_CSV}"
            )

        txt_handle = open(TXT_OUTPUT, "w") if SAVE_TXT else None
        try:
            write_line(txt_handle, "===== SIGNIFICANCE RESULTS =====")
            write_line(txt_handle, f"Experiment: {EXP_DIR.name}")
            write_line(txt_handle, f"Plot Title Suffix: {PLOT_TITLE_SUFFIX}")
            write_line(txt_handle, f"Metric: {metric_display_name(METRIC_KEY)}")
            write_line(txt_handle, f"Alternative Hypothesis: {WILCOXON_ALTERNATIVE}")
            write_line(txt_handle, f"Zero Method: {WILCOXON_ZERO_METHOD}")
            write_line(txt_handle)

            for summary in summaries:
                print_summary(summary, txt_handle)
                write_line(txt_handle)

            write_line(txt_handle, "=" * 72)
            write_line(txt_handle, "GROUP PLOTS")
            if n_events_plot_path is not None:
                write_line(txt_handle, f"WMAE vs Number of Sources Plot: {n_events_plot_path}")
            else:
                write_line(txt_handle, "WMAE vs Number of Sources Plot: not generated")

            if snr_plot_path is not None:
                write_line(txt_handle, f"WMAE vs SNR Bucket Plot: {snr_plot_path}")
            else:
                write_line(txt_handle, "WMAE vs SNR Bucket Plot: not generated (SNR bucket info may be unavailable)")

            if difficulty_plot_path is not None:
                write_line(txt_handle, f"WMAE vs Difficulty Plot: {difficulty_plot_path}")
            else:
                write_line(txt_handle, "WMAE vs Difficulty Plot: not generated")

            if boxplot_path is not None:
                write_line(txt_handle, f"WMAE Box Plot by Sources: {boxplot_path}")
            else:
                write_line(txt_handle, "WMAE Box Plot by Sources: not generated")

            if ambiguity_plot_path is not None:
                write_line(txt_handle, f"WMAE Box Plot by Ambiguity: {ambiguity_plot_path}")
            else:
                write_line(txt_handle, "WMAE Box Plot by Ambiguity: not generated")

            if snr_nsources_plot_path is not None:
                write_line(txt_handle, f"WMAE vs N Sources per SNR Bucket Plot: {snr_nsources_plot_path}")
            else:
                write_line(txt_handle, "WMAE vs N Sources per SNR Bucket Plot: not generated")
        finally:
            if txt_handle is not None:
                txt_handle.close()

        if SAVE_JSON:
            output = {
                "experiment_name": EXP_DIR.name,
                "plot_title_suffix": PLOT_TITLE_SUFFIX,
                "metric_key": METRIC_KEY,
                "metric_display_name": metric_display_name(METRIC_KEY),
                "wilcoxon_alternative": WILCOXON_ALTERNATIVE,
                "wilcoxon_zero_method": WILCOXON_ZERO_METHOD,
                "comparisons": summaries,
                "group_plots": {
                    "n_events_plot": str(n_events_plot_path) if n_events_plot_path is not None else None,
                    "snr_plot": str(snr_plot_path) if snr_plot_path is not None else None,
                    "difficulty_plot": str(difficulty_plot_path) if difficulty_plot_path is not None else None,
                    "boxplot_n_events": str(boxplot_path) if boxplot_path is not None else None,
                    "boxplot_ambiguity": str(ambiguity_plot_path) if ambiguity_plot_path is not None else None,
                    "snr_nsources_plot": str(snr_nsources_plot_path) if snr_nsources_plot_path is not None else None,
                },
            }
            with open(JSON_OUTPUT, "w") as f:
                json.dump(output, f, indent=2)

        if SAVE_TXT:
            print(f"\nSaved significance report to: {TXT_OUTPUT}")
        if SAVE_JSON:
            print(f"Saved significance JSON to: {JSON_OUTPUT}")
        if SAVE_DIFF_HISTOGRAMS:
            print(f"Saved histograms to: {HIST_DIR}")
        if SAVE_GROUP_PLOTS or SAVE_AMBIGUITY_BOXPLOT:
            print(f"Saved plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()