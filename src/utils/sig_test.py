# Statistical Testing - Perform Wilcoxon and Bootstrapping Statistical tests
# from spectrogram results

import ast
import json
from math import comb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "experiments" / "6000scenes"

# Input JSON files
ROTATION_JSON = EXP_DIR / "test_results_rotation_3.json"
STATIC_JSON = EXP_DIR / "test_results_static_3.json"
NO_ROT_JSON = EXP_DIR / "test_results_no_rotation.json"

MANIFEST_JSONL = EXP_DIR / "manifest.jsonl"

# Toggle which models to include
COMPARE_ROTATION = True
COMPARE_STATIC = True
COMPARE_NO_ROT = False

# ---- Metric toggles ----
# Enable/disable significance testing for each metric independently.
# Each enabled metric produces its own output files.
TEST_WMAE = True        # per_scene_wmae       -> significance_results_wmae.*
TEST_HORIZ_WMAE = True  # per_scene_horiz_wmae -> significance_results_horiz_wmae.*
TEST_VERT_WMAE = True   # per_scene_vert_wmae  -> significance_results_vert_wmae.*

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

# Base output dirs (metric-specific filenames are derived automatically)
HIST_DIR = EXP_DIR / "significance_histograms"
PLOT_DIR = EXP_DIR / "significance_plots"

# Per-metric output paths  (txt, json, n_events_plot, snr_plot, boxplot)
_METRIC_OUTPUTS = {
    "per_scene_wmae": {
        "txt":     EXP_DIR / "significance_results_wmae.txt",
        "json":    EXP_DIR / "significance_results_wmae.json",
        "n_events_plot": PLOT_DIR / "wmae_vs_num_sources_with_std.pdf",
        "snr_plot":      PLOT_DIR / "wmae_vs_snr_bucket_with_std.pdf",
        "boxplot":       PLOT_DIR / "wmae_vs_num_sources_boxplot.pdf",
    },
    "per_scene_horiz_wmae": {
        "txt":     EXP_DIR / "significance_results_horiz_wmae.txt",
        "json":    EXP_DIR / "significance_results_horiz_wmae.json",
        "n_events_plot": PLOT_DIR / "horiz_wmae_vs_num_sources_with_std.pdf",
        "snr_plot":      PLOT_DIR / "horiz_wmae_vs_snr_bucket_with_std.pdf",
        "boxplot":       PLOT_DIR / "horiz_wmae_vs_num_sources_boxplot.pdf",
    },
    "per_scene_vert_wmae": {
        "txt":     EXP_DIR / "significance_results_vert_wmae.txt",
        "json":    EXP_DIR / "significance_results_vert_wmae.json",
        "n_events_plot": PLOT_DIR / "vert_wmae_vs_num_sources_with_std.pdf",
        "snr_plot":      PLOT_DIR / "vert_wmae_vs_snr_bucket_with_std.pdf",
        "boxplot":       PLOT_DIR / "vert_wmae_vs_num_sources_boxplot.pdf",
    },
}

# Symmetry / skew warning thresholds
ABS_SKEW_WARNING_THRESHOLD = 1.0
MEAN_MEDIAN_GAP_STD_THRESHOLD = 0.5

# Plot style
FIGSIZE = (8, 5)
LINEWIDTH = 2.0
CAPSIZE = 4
MARKER_MEAN = "o"
LINESTYLE = (0, (4, 3))

# Per-model colours (in order: rotation, static, no_rot, ...)
MODEL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
]

# Bootstrap tail analysis
SAVE_BOOTSTRAP_TAIL_ANALYSIS = True
BOOTSTRAP_N_RESAMPLES = 10000
BOOTSTRAP_RANDOM_SEED = 42
BOOTSTRAP_QUANTILES = [90, 95, 99]
BOOTSTRAP_CI_LEVEL = 0.95

# Top/bottom scene analysis
TOP_K_IMPROVEMENT_SCENES = 6


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def get_enabled_model_paths():
    models = []

    if COMPARE_ROTATION:
        models.append(ROTATION_JSON)
    if COMPARE_STATIC:
        models.append(STATIC_JSON)
    if COMPARE_NO_ROT:
        models.append(NO_ROT_JSON)

    if len(models) < 2:
        raise ValueError("At least two model JSON files must be enabled.")

    return models


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
    if metric_key == "per_scene_horiz_wmae":
        return "Per-Scene Horizontal WMAE (°)"
    if metric_key == "per_scene_vert_wmae":
        return "Per-Scene Vertical WMAE (°)"
    if metric_key == "per_scene_mse_norm":
        return "Per-Scene Normalized MSE"
    return metric_key


def format_metric_value(value, metric_key):
    if value is None:
        return "N/A"
    if metric_key in ("per_scene_wmae", "per_scene_horiz_wmae", "per_scene_vert_wmae"):
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
    plt.grid(True)
    plt.tight_layout()

    out_path = HIST_DIR / f"{safe_name(model_a_name)}_vs_{safe_name(model_b_name)}_{metric_key}_hist.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return out_path


def write_line(f, line=""):
    print(line)
    if f is not None:
        f.write(line + "\n")


def bootstrap_quantile_difference_paired(a, b, q, n_resamples=10000, seed=42, ci_level=0.95):
    """
    Paired bootstrap for the difference in the q-th percentile.

    Returns results for:
      diff = quantile(B) - quantile(A)

    Positive diff means Model A has LOWER tail error than Model B,
    since lower error is better.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError("Paired bootstrap requires arrays of the same length.")

    n = len(a)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a_bs = a[idx]
        b_bs = b[idx]

        qa = np.percentile(a_bs, q)
        qb = np.percentile(b_bs, q)

        diffs[i] = qb - qa

    alpha = 1.0 - ci_level
    lower = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))

    point_a = float(np.percentile(a, q))
    point_b = float(np.percentile(b, q))
    point_diff = float(point_b - point_a)

    pvalue_greater = float((np.sum(diffs <= 0) + 1) / (n_resamples + 1))

    return {
        "quantile": int(q),
        "model_a_quantile": point_a,
        "model_b_quantile": point_b,
        "difference_b_minus_a": point_diff,
        "ci_level": float(ci_level),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_pvalue_greater": pvalue_greater,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def summarise_bootstrap_tail_analysis(model_a_json, model_b_json, metric_key):
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

    quantile_results = []
    if SAVE_BOOTSTRAP_TAIL_ANALYSIS:
        for q in BOOTSTRAP_QUANTILES:
            result = bootstrap_quantile_difference_paired(
                a=a,
                b=b,
                q=q,
                n_resamples=BOOTSTRAP_N_RESAMPLES,
                seed=BOOTSTRAP_RANDOM_SEED + int(q),
                ci_level=BOOTSTRAP_CI_LEVEL,
            )
            quantile_results.append(result)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "model_a_name": name_a,
        "model_b_name": name_b,
        "split": split_a,
        "metric_key": metric_key,
        "n_common_scenes": int(len(common_scene_ids)),
        "quantile_results": quantile_results,
        "interpretation_note": (
            "Differences are reported as quantile(Model B) - quantile(Model A). "
            "Positive values mean Model A has lower tail error than Model B."
        ),
    }


def print_bootstrap_tail_summary(summary, f=None):
    metric_key = summary["metric_key"]

    write_line(f, "=" * 72)
    write_line(f, f"BOOTSTRAP TAIL ANALYSIS: {summary['comparison']}")
    write_line(f, f"Split: {summary['split']}")
    write_line(f, f"Metric: {metric_display_name(metric_key)}")
    write_line(f, f"Common Scenes: {summary['n_common_scenes']}")
    write_line(f)
    write_line(
        f,
        f"Model A source: {summary['model_a_name']} "
        f"(first model in the pairwise comparison / first aligned metric array)"
    )
    write_line(
        f,
        f"Model B source: {summary['model_b_name']} "
        f"(second model in the pairwise comparison / second aligned metric array)"
    )
    write_line(f)
    write_line(f, summary["interpretation_note"])
    write_line(f)

    if not summary["quantile_results"]:
        write_line(f, "No bootstrap tail results generated.")
        return

    for result in summary["quantile_results"]:
        q = result["quantile"]
        ci_pct = int(round(100 * result["ci_level"]))

        write_line(f, f"{q}th Percentile Analysis")
        write_line(
            f,
            f"  Model A {q}th percentile: "
            f"{format_metric_value(result['model_a_quantile'], metric_key)}"
        )
        write_line(
            f,
            f"  Model B {q}th percentile: "
            f"{format_metric_value(result['model_b_quantile'], metric_key)}"
        )
        write_line(
            f,
            f"  Difference (B - A):       "
            f"{format_metric_value(result['difference_b_minus_a'], metric_key)}"
        )
        write_line(
            f,
            f"  {ci_pct}% Bootstrap CI:     "
            f"[{format_metric_value(result['ci_lower'], metric_key)}, "
            f"{format_metric_value(result['ci_upper'], metric_key)}]"
        )
        write_line(
            f,
            f"  One-sided bootstrap p-value "
            f"(H1: Model A has lower {q}th percentile error): "
            f"{result['bootstrap_pvalue_greater']:.6e}"
        )

        if result["ci_lower"] > 0:
            write_line(
                f,
                f"  Interpretation: The entire CI is above 0, which supports that "
                f"Model A has lower {q}th percentile error than Model B."
            )
        elif result["ci_upper"] < 0:
            write_line(
                f,
                f"  Interpretation: The entire CI is below 0, which supports that "
                f"Model B has lower {q}th percentile error than Model A."
            )
        else:
            write_line(
                f,
                f"  Interpretation: The CI crosses 0, so the difference in the "
                f"{q}th percentile is inconclusive."
            )

        write_line(f)


def get_top_improvement_scenes(model_a_json, model_b_json, metric_key, top_k=6):
    """Return the top_k scenes with the largest (B - A) difference.

    Positive improvement_b_minus_a means Model A has lower error than Model B.
    Scenes are sorted descending so the largest differences appear first.
    """
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
    diff = b - a  # positive = Model A is better

    records = []
    for i, sid in enumerate(common_scene_ids):
        records.append({
            "scene_id": sid,
            "model_a_value": float(a[i]),
            "model_b_value": float(b[i]),
            "improvement_b_minus_a": float(diff[i]),
        })

    records_sorted = sorted(records, key=lambda x: x["improvement_b_minus_a"], reverse=True)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "model_a_name": name_a,
        "model_b_name": name_b,
        "split": split_a,
        "metric_key": metric_key,
        "top_k": int(top_k),
        "top_scenes": records_sorted[:top_k],
        "interpretation_note": (
            "Scenes are ranked by (Model B - Model A) descending (largest difference first). "
            "Positive values mean Model A has lower error than Model B."
        ),
    }


def print_top_improvement_scenes(summary, f=None):
    metric_key = summary["metric_key"]

    write_line(f, "=" * 72)
    write_line(f, f"TOP {summary['top_k']} SCENES WITH LARGEST IMPROVEMENT (WORST FOR Model B)")
    write_line(f, f"Comparison: {summary['comparison']}")
    write_line(f, f"Split: {summary['split']}")
    write_line(f)
    write_line(f, f"Model A source: {summary['model_a_name']}")
    write_line(f, f"Model B source: {summary['model_b_name']}")
    write_line(f, summary["interpretation_note"])
    write_line(f)

    for i, rec in enumerate(summary["top_scenes"], 1):
        write_line(f, f"{i}. Scene ID: {rec['scene_id']}")
        write_line(
            f,
            f"   Model A ({summary['model_a_name']}): "
            f"{format_metric_value(rec['model_a_value'], metric_key)}"
        )
        write_line(
            f,
            f"   Model B ({summary['model_b_name']}): "
            f"{format_metric_value(rec['model_b_value'], metric_key)}"
        )
        write_line(
            f,
            f"   Improvement (B - A): "
            f"{format_metric_value(rec['improvement_b_minus_a'], metric_key)}"
        )
        write_line(f)


def get_bottom_improvement_scenes(model_a_json, model_b_json, metric_key, top_k=6):
    """Return the top_k scenes with the smallest (B - A) difference.

    These are scenes where the two models perform most similarly, i.e. the
    smallest absolute advantage of Model A over Model B (or where Model B
    is actually better).  Scenes are sorted ascending so the smallest
    differences appear first.
    """
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
    diff = b - a  # positive = Model A is better

    records = []
    for i, sid in enumerate(common_scene_ids):
        records.append({
            "scene_id": sid,
            "model_a_value": float(a[i]),
            "model_b_value": float(b[i]),
            "improvement_b_minus_a": float(diff[i]),
        })

    # Ascending sort: smallest (B - A) first — closest performance or Model B winning
    records_sorted = sorted(records, key=lambda x: x["improvement_b_minus_a"])

    return {
        "comparison": f"{name_a} vs {name_b}",
        "model_a_name": name_a,
        "model_b_name": name_b,
        "split": split_a,
        "metric_key": metric_key,
        "top_k": int(top_k),
        "bottom_scenes": records_sorted[:top_k],
        "interpretation_note": (
            "Scenes are ranked by (Model B - Model A) ascending (smallest difference first). "
            "These are scenes where Model A's advantage over Model B is smallest, "
            "or where Model B outperforms Model A."
        ),
    }


def get_closest_to_zero_scenes(model_a_json, model_b_json, metric_key, top_k=6):
    """Return the top_k scenes where the paired difference (B - A) is closest to zero.

    These are scenes where the two models perform most similarly regardless of
    which one is slightly ahead. Scenes are sorted by abs(B - A) ascending.
    """
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

    records = []
    for i, sid in enumerate(common_scene_ids):
        records.append({
            "scene_id": sid,
            "model_a_value": float(a[i]),
            "model_b_value": float(b[i]),
            "improvement_b_minus_a": float(diff[i]),
            "abs_difference": float(abs(diff[i])),
        })

    # Sort by absolute difference ascending — closest to zero first
    records_sorted = sorted(records, key=lambda x: x["abs_difference"])

    return {
        "comparison": f"{name_a} vs {name_b}",
        "model_a_name": name_a,
        "model_b_name": name_b,
        "split": split_a,
        "metric_key": metric_key,
        "top_k": int(top_k),
        "closest_scenes": records_sorted[:top_k],
        "interpretation_note": (
            "Scenes are ranked by abs(Model B - Model A) ascending. "
            "These are the scenes where the two models perform most similarly."
        ),
    }


def print_closest_to_zero_scenes(summary, f=None):
    metric_key = summary["metric_key"]

    write_line(f, "=" * 72)
    write_line(f, f"TOP {summary['top_k']} SCENES WITH DIFFERENCE CLOSEST TO ZERO (NEAR DRAWS)")
    write_line(f, f"Comparison: {summary['comparison']}")
    write_line(f, f"Split: {summary['split']}")
    write_line(f)
    write_line(f, f"Model A source: {summary['model_a_name']}")
    write_line(f, f"Model B source: {summary['model_b_name']}")
    write_line(f, summary["interpretation_note"])
    write_line(f)

    for i, rec in enumerate(summary["closest_scenes"], 1):
        write_line(f, f"{i}. Scene ID: {rec['scene_id']}")
        write_line(
            f,
            f"   Model A ({summary['model_a_name']}): "
            f"{format_metric_value(rec['model_a_value'], metric_key)}"
        )
        write_line(
            f,
            f"   Model B ({summary['model_b_name']}): "
            f"{format_metric_value(rec['model_b_value'], metric_key)}"
        )
        write_line(
            f,
            f"   Difference (B - A): "
            f"{format_metric_value(rec['improvement_b_minus_a'], metric_key)} "
            f"(abs: {format_metric_value(rec['abs_difference'], metric_key)})"
        )
        write_line(f)


def print_bottom_improvement_scenes(summary, f=None):
    metric_key = summary["metric_key"]

    write_line(f, "=" * 72)
    write_line(f, f"TOP {summary['top_k']} SCENES WITH SMALLEST IMPROVEMENT (BEST FOR Model B)")
    write_line(f, f"Comparison: {summary['comparison']}")
    write_line(f, f"Split: {summary['split']}")
    write_line(f)
    write_line(f, f"Model A source: {summary['model_a_name']}")
    write_line(f, f"Model B source: {summary['model_b_name']}")
    write_line(f, summary["interpretation_note"])
    write_line(f)

    for i, rec in enumerate(summary["bottom_scenes"], 1):
        write_line(f, f"{i}. Scene ID: {rec['scene_id']}")
        write_line(
            f,
            f"   Model A ({summary['model_a_name']}): "
            f"{format_metric_value(rec['model_a_value'], metric_key)}"
        )
        write_line(
            f,
            f"   Model B ({summary['model_b_name']}): "
            f"{format_metric_value(rec['model_b_value'], metric_key)}"
        )
        write_line(
            f,
            f"   Improvement (B - A): "
            f"{format_metric_value(rec['improvement_b_minus_a'], metric_key)}"
        )
        write_line(f)


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


def collect_group_wmae_stats(results_json, group_name):
    raw_group = get_group_distribution(results_json, group_name)
    out = {}

    for key_str, group_stats in raw_group.items():
        wmae = group_stats.get("wmae", {})
        parsed_key = parse_group_key(key_str)

        if wmae.get("count", 0) == 0:
            continue

        out[parsed_key] = {
            "label": key_str,
            "mean": wmae.get("mean", None),
            "std": wmae.get("std", None),
            "max": wmae.get("max", None),
            "min": wmae.get("min", None),
            "p50": wmae.get("p50", None),
            "p90": wmae.get("p90", None),
            "p95": wmae.get("p95", None),
            "p99": wmae.get("p99", None),
            "count": wmae.get("count", 0),
        }

    return out


def sort_n_event_keys(keys):
    numeric = [k for k in keys if isinstance(k, int)]
    other = [k for k in keys if not isinstance(k, int)]
    return sorted(numeric) + sorted(other, key=lambda x: str(x))


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
        return f"{k[0]} to {k[1]}"
    return str(k)


def load_manifest(manifest_path: Path):
    """Load the JSONL manifest and return a dict mapping scene_id -> record."""
    manifest = {}
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            manifest[record["scene_id"]] = record
    return manifest


# Manifest field name -> group_name used in results JSON
_MANIFEST_GROUP_FIELD = {
    "n_events": "n_sources",
    "difficulty": "difficulty",
}


def get_scene_group_mapping(results_json, group_name):
    """Build a dict mapping scene_id -> group_key.

    Priority order:
      1. ``scene_group_assignments`` in the results JSON (inline).
      2. ``scene_<group_name>`` in the results JSON (legacy).
      3. Manifest JSONL file (using MANIFEST_JSONL config path).
    """
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


def plot_group_wmae_with_std(model_jsons, group_name, output_path):
    """Mean ± std error-bar plot (no max scatter)."""
    if group_name == "n_events":
        title = "Weighted Mean Angular Error by Number of Sources"
        x_label = "Number of Sources in a Scene"
        get_sorted_keys = sort_n_event_keys
        tick_formatter = lambda k: str(k)
    elif group_name == "snr_bucket":
        title = "Weighted Mean Angular Error by SNR Bucket"
        x_label = "SNR Bucket"
        get_sorted_keys = lambda keys: sorted(keys, key=snr_sort_key)
        tick_formatter = format_snr_label
    else:
        raise ValueError(f"Unsupported group_name: {group_name}")

    model_group_stats = []
    all_keys = set()

    for model_json in model_jsons:
        stats = collect_group_wmae_stats(model_json, group_name)
        if stats:
            model_group_stats.append((model_json.get("model_name", "Unknown Model"), stats))
            all_keys.update(stats.keys())

    if not model_group_stats:
        return None

    sorted_keys = get_sorted_keys(list(all_keys))
    x_base = np.arange(len(sorted_keys), dtype=np.float64)

    fig = plt.figure(figsize=FIGSIZE)

    for model_name, stats in model_group_stats:
        xs = []
        means = []
        stds = []

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

        if not xs:
            continue

        plt.errorbar(
            xs,
            means,
            yerr=stds,
            marker=MARKER_MEAN,
            linestyle=LINESTYLE,
            linewidth=LINEWIDTH,
            capsize=CAPSIZE,
            label=model_name,
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Weighted Mean Angular Error (°)")
    plt.xticks(
        x_base,
        [tick_formatter(k) for k in sorted_keys],
        rotation=30 if group_name == "snr_bucket" else 0,
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_group_wmae_boxplot(model_jsons, group_name, output_path):
    """Box-and-whisker plot of per-scene WMAE grouped by n_events.

    Uses raw per-scene values when ``scene_group_assignments`` (or
    ``scene_<group_name>``) is present in the JSON. Otherwise falls back
    to a synthetic box plot built from summary statistics.
    """
    if group_name == "n_events":
        title = "WMAE Distribution by Number of Sources (W/Background)"
        x_label = "Number of Sources in a Scene"
        get_sorted_keys = sort_n_event_keys
        tick_formatter = lambda k: str(k)
    elif group_name == "snr_bucket":
        title = "WMAE Distribution by SNR Bucket (W/Background)"
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
        per_scene = model_json.get("per_scene_wmae", {})
        group_map = get_scene_group_mapping(model_json, group_name)
        stats = collect_group_wmae_stats(model_json, group_name)

        if group_map and per_scene:
            grouped_raw = {}
            for scene_id, wmae_val in per_scene.items():
                if scene_id in group_map:
                    key = parse_group_key(str(group_map[scene_id]))
                    grouped_raw.setdefault(key, []).append(wmae_val)
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
            "missing — falling back to summary-stat box plot (no real quartiles "
            "or outliers). Set MANIFEST_JSONL to your manifest.jsonl path."
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
                    m_val, pos - half_w, pos + half_w,
                    colors=color, linestyles=":", linewidth=LINEWIDTH * 0.8,
                )

        ax.plot([], [], color=color, linewidth=LINEWIDTH, label=model_name)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Weighted Mean Angular Error (°)")
    ax.set_xticks(x_base)
    ax.set_xticklabels([tick_formatter(k) for k in sorted_keys])
    ax.grid(True, axis="y")

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="grey", linewidth=LINEWIDTH * 0.8, linestyle=":"))
    labels.append("Mean")
    handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=3))
    labels.append("Outliers")
    ax.legend(handles, labels)

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

def _get_enabled_metrics():
    """Return list of metric keys to run based on toggle flags."""
    enabled = []
    if TEST_WMAE:
        enabled.append("per_scene_wmae")
    if TEST_HORIZ_WMAE:
        enabled.append("per_scene_horiz_wmae")
    if TEST_VERT_WMAE:
        enabled.append("per_scene_vert_wmae")
    if not enabled:
        raise ValueError("At least one metric toggle (TEST_WMAE / TEST_HORIZ_WMAE / TEST_VERT_WMAE) must be True.")
    return enabled


def _run_metric(model_jsons, metric_key):
    """Run the full significance pipeline for a single metric key.

    Writes its own .txt and .json output files and returns the paths
    of any files that were saved.
    """
    paths = _METRIC_OUTPUTS[metric_key]
    txt_path  = paths["txt"]
    json_path = paths["json"]
    n_events_plot_path = None
    snr_plot_path      = None
    boxplot_path       = None

    summaries               = []
    bootstrap_tail_summaries = []
    top_improvement_summaries   = []
    bottom_improvement_summaries = []
    closest_to_zero_summaries    = []

    for i in range(len(model_jsons)):
        for j in range(i + 1, len(model_jsons)):
            summaries.append(
                summarise_pairwise_comparison(model_jsons[i], model_jsons[j], metric_key)
            )
            bootstrap_tail_summaries.append(
                summarise_bootstrap_tail_analysis(model_jsons[i], model_jsons[j], metric_key)
            )
            top_improvement_summaries.append(
                get_top_improvement_scenes(model_jsons[i], model_jsons[j], metric_key,
                                           top_k=TOP_K_IMPROVEMENT_SCENES)
            )
            bottom_improvement_summaries.append(
                get_bottom_improvement_scenes(model_jsons[i], model_jsons[j], metric_key,
                                              top_k=TOP_K_IMPROVEMENT_SCENES)
            )
            closest_to_zero_summaries.append(
                get_closest_to_zero_scenes(model_jsons[i], model_jsons[j], metric_key,
                                           top_k=TOP_K_IMPROVEMENT_SCENES)
            )

    # Group plots are only generated for the full WMAE (group_distributions
    # keys are "wmae"-based and don't exist for horiz/vert decompositions).
    if SAVE_GROUP_PLOTS and metric_key == "per_scene_wmae":
        n_events_plot_path = plot_group_wmae_with_std(
            model_jsons=model_jsons,
            group_name="n_events",
            output_path=paths["n_events_plot"],
        )
        snr_plot_path = plot_group_wmae_with_std(
            model_jsons=model_jsons,
            group_name="snr_bucket",
            output_path=paths["snr_plot"],
        )
        boxplot_path = plot_group_wmae_boxplot(
            model_jsons=model_jsons,
            group_name="n_events",
            output_path=paths["boxplot"],
        )

    # ---- TXT output ----
    txt_handle = open(txt_path, "w") if SAVE_TXT else None
    try:
        write_line(txt_handle, "===== SIGNIFICANCE RESULTS =====")
        write_line(txt_handle, f"Metric: {metric_display_name(metric_key)}")
        write_line(txt_handle, f"Alternative Hypothesis: {WILCOXON_ALTERNATIVE}")
        write_line(txt_handle, f"Zero Method: {WILCOXON_ZERO_METHOD}")
        write_line(txt_handle)

        for summary in summaries:
            print_summary(summary, txt_handle)
            write_line(txt_handle)

        if SAVE_BOOTSTRAP_TAIL_ANALYSIS:
            write_line(txt_handle, "=" * 72)
            write_line(txt_handle, "BOOTSTRAP TAIL ANALYSIS")
            write_line(txt_handle, f"Resamples: {BOOTSTRAP_N_RESAMPLES}")
            write_line(txt_handle, f"CI Level: {BOOTSTRAP_CI_LEVEL:.2f}")
            write_line(txt_handle, f"Quantiles: {BOOTSTRAP_QUANTILES}")
            write_line(txt_handle, f"Random Seed: {BOOTSTRAP_RANDOM_SEED}")
            write_line(txt_handle)

            for bootstrap_summary in bootstrap_tail_summaries:
                print_bootstrap_tail_summary(bootstrap_summary, txt_handle)
                write_line(txt_handle)

        write_line(txt_handle, "=" * 72)
        write_line(txt_handle, "TOP IMPROVEMENT SCENES ANALYSIS")
        write_line(txt_handle)
        for top_summary in top_improvement_summaries:
            print_top_improvement_scenes(top_summary, txt_handle)
            write_line(txt_handle)

        write_line(txt_handle, "=" * 72)
        write_line(txt_handle, "BOTTOM IMPROVEMENT SCENES ANALYSIS")
        write_line(txt_handle)
        for bottom_summary in bottom_improvement_summaries:
            print_bottom_improvement_scenes(bottom_summary, txt_handle)
            write_line(txt_handle)

        write_line(txt_handle, "=" * 72)
        write_line(txt_handle, "CLOSEST TO ZERO SCENES ANALYSIS (NEAR DRAWS)")
        write_line(txt_handle)
        for closest_summary in closest_to_zero_summaries:
            print_closest_to_zero_scenes(closest_summary, txt_handle)
            write_line(txt_handle)

        if SAVE_GROUP_PLOTS and metric_key == "per_scene_wmae":
            write_line(txt_handle, "=" * 72)
            write_line(txt_handle, "GROUP PLOTS")
            write_line(txt_handle,
                f"WMAE vs Number of Sources Plot: {n_events_plot_path}"
                if n_events_plot_path is not None
                else "WMAE vs Number of Sources Plot: not generated")
            write_line(txt_handle,
                f"WMAE vs SNR Bucket Plot: {snr_plot_path}"
                if snr_plot_path is not None
                else "WMAE vs SNR Bucket Plot: not generated (SNR bucket info may be unavailable)")
            write_line(txt_handle,
                f"WMAE Box Plot by Sources: {boxplot_path}"
                if boxplot_path is not None
                else "WMAE Box Plot by Sources: not generated")
    finally:
        if txt_handle is not None:
            txt_handle.close()

    # ---- JSON output ----
    if SAVE_JSON:
        output = {
            "metric_key": metric_key,
            "metric_display_name": metric_display_name(metric_key),
            "wilcoxon_alternative": WILCOXON_ALTERNATIVE,
            "wilcoxon_zero_method": WILCOXON_ZERO_METHOD,
            "comparisons": summaries,
            "bootstrap_tail_analysis": bootstrap_tail_summaries,
            "top_improvement_scenes": top_improvement_summaries,
            "bottom_improvement_scenes": bottom_improvement_summaries,
            "closest_to_zero_scenes": closest_to_zero_summaries,
            "group_plots": {
                "n_events_plot": str(n_events_plot_path) if n_events_plot_path is not None else None,
                "snr_plot":      str(snr_plot_path)      if snr_plot_path      is not None else None,
                "boxplot_n_events": str(boxplot_path)    if boxplot_path       is not None else None,
            },
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

    return txt_path, json_path


def main():
    model_paths  = get_enabled_model_paths()
    model_jsons  = [load_json(path) for path in model_paths]
    enabled_metrics = _get_enabled_metrics()

    for metric_key in enabled_metrics:
        print(f"\n{'=' * 72}")
        print(f"Running significance tests for: {metric_display_name(metric_key)}")
        print(f"{'=' * 72}")

        txt_path, json_path = _run_metric(model_jsons, metric_key)

        if SAVE_TXT:
            print(f"Saved significance report to: {txt_path}")
        if SAVE_JSON:
            print(f"Saved significance JSON  to: {json_path}")

    if SAVE_DIFF_HISTOGRAMS:
        print(f"\nSaved histograms to: {HIST_DIR}")
    if SAVE_GROUP_PLOTS:
        print(f"Saved group plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()