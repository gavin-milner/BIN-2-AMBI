# Plotting script - Check the distribution of ambiguous scenes.

import csv
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_scene_positions_csv(csv_path: Path):
    """Load CSV and return list of rows."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def visualize_ambiguity_scatter(csv_path: Path):
    """Create scatter plot of sources on equirectangular plane colored by ambiguity."""
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        return
    
    rows = load_scene_positions_csv(csv_path)
    print(f"Loaded {len(rows)} total source entries\n")
    
    # Group by scene to classify
    scene_events = defaultdict(list)
    for row in rows:
        sid = row["scene_id"]
        az = float(row["azimuth_deg"])
        el = float(row["elevation_deg"])
        ambig = row["front_back_ambiguous"].strip().lower() == "true"
        
        scene_events[sid].append({
            "azimuth": az,
            "elevation": el,
            "ambiguous": ambig,
        })
    
    # Classify each scene and build visualization data
    easily_az, easily_el = [], []
    some_az, some_el = [], []
    all_az, all_el = [], []
    
    category_counts = {"Easily Localisable": 0, "Some Ambiguous": 0, "All Ambiguous": 0}
    
    for sid, events in scene_events.items():
        n_sources = len(events)
        n_ambig = sum(1 for e in events if e["ambiguous"])
        
        # Determine category
        if n_ambig == n_sources:
            cat = "All Ambiguous"
        elif n_ambig > 0:
            cat = "Some Ambiguous"
        else:
            cat = "Easily Localisable"
        
        category_counts[cat] += n_sources
        
        # Collect points
        for event in events:
            az = event["azimuth"]
            el = event["elevation"]
            
            if cat == "Easily Localisable":
                easily_az.append(az)
                easily_el.append(el)
            elif cat == "Some Ambiguous":
                some_az.append(az)
                some_el.append(el)
            else:  # All Ambiguous
                all_az.append(az)
                all_el.append(el)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot points
    ax.scatter(easily_az, easily_el, c='green', s=50, alpha=0.6, 
               label=f'Easily Localisable (n={category_counts["Easily Localisable"]})', 
               edgecolors='darkgreen', linewidth=0.5)
    ax.scatter(some_az, some_el, c='orange', s=50, alpha=0.6, 
               label=f'Some Ambiguous (n={category_counts["Some Ambiguous"]})', 
               edgecolors='darkorange', linewidth=0.5)
    ax.scatter(all_az, all_el, c='red', s=50, alpha=0.6, 
               label=f'All Ambiguous (n={category_counts["All Ambiguous"]})', 
               edgecolors='darkred', linewidth=0.5)
    
    # Add shaded regions for front-back ambiguity zones (±30°)
    # Front zone: -30 to +30 (330 to 30)
    ax.axvspan(-30, 30, alpha=0.1, color='purple', label='Front ambiguity zone (±30°)')
    ax.axvspan(330, 360, alpha=0.1, color='purple')
    
    # Rear zone: 150 to 210
    ax.axvspan(150, 210, alpha=0.1, color='purple', label='Rear ambiguity zone (±30°)')
    
    # Formatting
    ax.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax.set_ylabel('Elevation (degrees)', fontsize=12)
    ax.set_title('Source Locations by Ambiguity Category (Equirectangular Plane)', fontsize=14)
    
    # Set axis limits
    ax.set_xlim(-180, 360)
    ax.set_ylim(-90, 90)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Equator (el=0°)')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Front (az=0°)')
    ax.axvline(x=180, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Back (az=180°)')
    
    # X-axis ticks
    ax.set_xticks([-180, -90, 0, 90, 180, 270, 360])
    ax.set_xticklabels(['-180°', '-90°\n(Left)', '0°\n(Front)', '90°\n(Right)', '180°\n(Back)', '270°', '360°'])
    
    # Y-axis ticks
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_yticklabels(['-90°\n(Down)', '-45°', '0°\n(Horiz)', '+45°', '+90°\n(Up)'])
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.parent / "ambiguity_scatter_plot.pdf"
    plt.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
    print(f"✓ Scatter plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 80)
    print("AMBIGUITY CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Easily Localisable: {category_counts['Easily Localisable']} sources")
    print(f"Some Ambiguous:     {category_counts['Some Ambiguous']} sources")
    print(f"All Ambiguous:      {category_counts['All Ambiguous']} sources")
    print(f"Total:              {sum(category_counts.values())} sources")
    

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_ambiguity_scatter.py <path_to_csv>")
        print("\nExample: python debug_ambiguity_scatter.py datasets/6000scenes_no_bg/test_scene_positions.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    visualize_ambiguity_scatter(csv_path)
