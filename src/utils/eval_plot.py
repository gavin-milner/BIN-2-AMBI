# Plotting script - Plot the results of the AMBIQUAL evaluation.

from __future__ import annotations

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_manifest(manifest_path: Path) -> Dict[str, int]:
    """Load manifest.jsonl and extract scene_id -> n_sources mapping"""
    scene_sources = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            scene_id = data["scene_id"]
            n_sources = data["n_sources"]
            scene_sources[scene_id] = n_sources
    return scene_sources


def load_ambiqual_csv(csv_path: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Load ambiqual CSV and extract scene_id -> LA mappings
    
    Returns:
        Tuple of (scene_gt_la, scene_pred_la, scene_la_percent)
    """
    scene_gt_la = {}
    scene_pred_la = {}
    scene_la_percent = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row["scene_id"]
            
            # GT Resynth LA
            gt_la = row.get("GT_Resynth_LA")
            if gt_la and gt_la.lower() != "none":
                try:
                    scene_gt_la[scene_id] = float(gt_la)
                except ValueError:
                    pass
            
            # Pred Resynth LA
            pred_la = row.get("Pred_Resynth_LA")
            if pred_la and pred_la.lower() != "none":
                try:
                    scene_pred_la[scene_id] = float(pred_la)
                except ValueError:
                    pass
            
            # LA Percent
            la_percent = row.get("LA_percent")
            if la_percent and la_percent.lower() != "none":
                try:
                    scene_la_percent[scene_id] = float(la_percent)
                except ValueError:
                    pass
    
    return scene_gt_la, scene_pred_la, scene_la_percent


def merge_data(scene_sources: Dict[str, int], scene_gt_la: Dict[str, float], 
               scene_pred_la: Dict[str, float], scene_la_percent: Dict[str, float]) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Merge manifest and ambiqual data, keeping only scenes present in both"""
    n_sources_list = []
    gt_la_list = []
    pred_la_list = []
    la_percent_list = []
    
    for scene_id, gt_la in scene_gt_la.items():
        if scene_id in scene_sources and scene_id in scene_pred_la and scene_id in scene_la_percent:
            n_sources_list.append(scene_sources[scene_id])
            gt_la_list.append(gt_la)
            pred_la_list.append(scene_pred_la[scene_id])
            la_percent_list.append(scene_la_percent[scene_id])
    
    return n_sources_list, gt_la_list, pred_la_list, la_percent_list


def plot_la_vs_sources(n_sources_list: List[int], la_percent_list: List[float], 
                       gt_la_list: List[float], pred_la_list: List[float], output_path: Path) -> None:
    """Create separate plots for absolute and relative LA vs number of sources"""
    
    # Calculate means for GT and Pred
    unique_sources = sorted(set(n_sources_list))
    gt_means = []
    gt_stds = []
    pred_means = []
    pred_stds = []
    percent_means = []
    percent_stds = []
    
    for n_src in unique_sources:
        gt_las = [gt_la for n, gt_la in zip(n_sources_list, gt_la_list) if n == n_src]
        pred_las = [pred_la for n, pred_la in zip(n_sources_list, pred_la_list) if n == n_src]
        percents = [pct for n, pct in zip(n_sources_list, la_percent_list) if n == n_src]
        gt_means.append(np.mean(gt_las))
        gt_stds.append(np.std(gt_las))
        pred_means.append(np.mean(pred_las))
        pred_stds.append(np.std(pred_las))
        percent_means.append(np.mean(percents))
        percent_stds.append(np.std(percents))
    
    # ===== Plot 1: Absolute LA values =====
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.errorbar(unique_sources, gt_means, yerr=gt_stds, fmt='o-', color='blue', linewidth=2.5, 
                markersize=8, capsize=5, label='GT Mean ± Std', elinewidth=2)
    ax1.errorbar(unique_sources, pred_means, yerr=pred_stds, fmt='s-', color='orange', linewidth=2.5, 
                markersize=8, capsize=5, label='Pred Mean ± Std', elinewidth=2)
    
    ax1.set_xlabel("Number of Sources", fontsize=12)
    ax1.set_ylabel("LA (Absolute Score 0-1)", fontsize=12)
    ax1.set_title("Absolute LA: GT Resynth vs Pred Resynth (No Background, MSE ACCDOA)", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(unique_sources)
    ax1.set_ylim([0, 1])
    
    plt.tight_layout()
    output_absolute = output_path.with_stem(output_path.stem + "_absolute")
    plt.savefig(output_absolute, format='pdf', bbox_inches='tight')
    print(f"Plot saved to: {output_absolute}")
    plt.close(fig1)
    
    # ===== Plot 2: Percentage (Pred % of GT) =====
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Scatter plot for individual data points
    ax2.scatter(n_sources_list, la_percent_list, alpha=0.4, s=30, edgecolors='lightgreen', linewidth=0.5, color='lightgreen')
    
    ax2.errorbar(unique_sources, percent_means, yerr=percent_stds, fmt='o-', color='green', linewidth=2.5, 
                markersize=8, capsize=5, label='Mean ± Std', elinewidth=2)
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Perfect (100%)')
    
    ax2.set_xlabel("Number of Sources", fontsize=12)
    ax2.set_ylabel("LA (Pred % of GT Resynth)", fontsize=12)
    ax2.set_title("Relative LA: Pred as % of GT (No Background, MSE ACCDOA)", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(unique_sources)
    
    plt.tight_layout()
    output_relative = output_path.with_stem(output_path.stem + "_relative")
    plt.savefig(output_relative, format='pdf', bbox_inches='tight')
    print(f"Plot saved to: {output_relative}")
    plt.close(fig2)
    
    # Print statistics
    print("\n=== LA Accuracy by Number of Sources (Absolute) ===")
    for n_src, gt_mean, gt_std, pred_mean, pred_std, pct_mean, pct_std in zip(
            unique_sources, gt_means, gt_stds, pred_means, pred_stds, percent_means, percent_stds):
        count = sum(1 for n in n_sources_list if n == n_src)
        print(f"Sources: {n_src} | Count: {count:4d}")
        print(f"  GT:   Mean: {gt_mean:6.4f} | Std: {gt_std:6.4f}")
        print(f"  Pred: Mean: {pred_mean:6.4f} | Std: {pred_std:6.4f}")
        print(f"  Pred as % of GT: {pct_mean:6.2f}% | Std: {pct_std:6.2f}%")


def main() -> None:
    # -------------------------
    # CONFIG
    # -------------------------
    ROOT = Path(__file__).resolve().parents[2]
    DATASET_NAME = "6000scenes_no_bg"
    EXPERIMENT_NAME = "CRNN_V15_Rotation"
    
    MANIFEST_PATH = ROOT / "Datasets" / DATASET_NAME / "manifest.jsonl"
    AMBIQUAL_CSV = ROOT / "Experiments" / EXPERIMENT_NAME / "ambiqual_per_scene.csv"
    OUTPUT_PLOT = ROOT / "Experiments" / EXPERIMENT_NAME / "la_vs_sources.pdf"
    
    # -------------------------
    # Load and process data
    # -------------------------
    print(f"Loading manifest from: {MANIFEST_PATH}")
    scene_sources = load_manifest(MANIFEST_PATH)
    print(f"Loaded {len(scene_sources)} scenes from manifest")
    
    print(f"Loading ambiqual CSV from: {AMBIQUAL_CSV}")
    scene_gt_la, scene_pred_la, scene_la_percent = load_ambiqual_csv(AMBIQUAL_CSV)
    print(f"Loaded {len(scene_gt_la)} GT LA, {len(scene_pred_la)} Pred LA, {len(scene_la_percent)} percent values from ambiqual CSV")
    
    n_sources_list, gt_la_list, pred_la_list, la_percent_list = merge_data(scene_sources, scene_gt_la, scene_pred_la, scene_la_percent)
    print(f"Merged {len(n_sources_list)} scenes with data in all files")
    
    # -------------------------
    # Generate plot
    # -------------------------
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plot_la_vs_sources(n_sources_list, la_percent_list, gt_la_list, pred_la_list, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
