"""
Publication-quality radar chart for the manifold activation experiment.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Load results
with open("/Users/x/src/lang-tokens/manifold_results.json") as f:
    results = json.load(f)

group_mass = results["group_mass"]
groups_list = list(results["concept_groups"].keys())
n_groups = len(groups_list)

# Cleaner group labels
group_labels = ["Value &\nFundamental", "Macro &\nGlobal", "Quantitative &\nTechnical", "General &\nNeutral"]

# Personas to plot
radar_personas = [
    ("Warren Buffett", '#c0392b', '-', 'o', 2.5),
    ("George Soros", '#e67e22', '-', 's', 2.5),
    ("Jim Simons", '#2980b9', '-', '^', 2.5),
    ("a random person", '#7f8c8d', '--', 'D', 2.0),
]

# Setup
angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_facecolor('#fafafa')

# Grid styling
ax.set_rlabel_position(45)
ax.yaxis.grid(True, color='#cccccc', linewidth=0.5)
ax.xaxis.grid(True, color='#cccccc', linewidth=0.5)
ax.spines['polar'].set_visible(False)

# Plot each persona
for persona, color, ls, marker, lw in radar_personas:
    values = [group_mass[persona][g] for g in groups_list]
    values += values[:1]
    ax.plot(angles, values, ls, linewidth=lw, label=persona, color=color,
            marker=marker, markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.fill(angles, values, alpha=0.08, color=color)

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(group_labels, fontsize=13, fontweight='bold', color='#333')

# Radial ticks
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%'],
                    fontsize=9, color='#666')
ax.set_ylim(0, 0.78)

# Title
ax.set_title("Where Does the Probability Mass Land?\n"
             "Concept activation profile by entity name",
             fontsize=16, fontweight='bold', pad=25, color='#222')

# Legend
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08), fontsize=12,
                   frameon=True, fancybox=True, shadow=False,
                   edgecolor='#cccccc', facecolor='white')
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_radar_final.png", dpi=200, bbox_inches='tight',
            facecolor='white')
print("Saved: manifold_radar_final.png")

# Also make a cleaner version of the heatmap focused on the key story
# Show Buffett vs Soros vs Random as a grouped concept chart
fig, ax = plt.subplots(figsize=(14, 6))

concept_groups_ordered = results["concept_groups"]
all_concepts = []
group_boundaries = []
group_labels_bottom = []
pos = 0
for group, concepts in concept_groups_ordered.items():
    group_boundaries.append(pos)
    group_labels_bottom.append(group.split("/")[0].strip())
    all_concepts.extend(concepts)
    pos += len(concepts)

personas_to_show = ["Warren Buffett", "George Soros", "Jim Simons", "a random person"]
colors_bar = ['#c0392b', '#e67e22', '#2980b9', '#7f8c8d']

x = np.arange(len(all_concepts))
width = 0.2
for i, (persona, color) in enumerate(zip(personas_to_show, colors_bar)):
    probs = [results["persona_avg_probs"][persona].get(c, 0) for c in all_concepts]
    ax.bar(x + i * width - width * 1.5, probs, width, label=persona, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(all_concepts, rotation=70, ha='right', fontsize=8)
ax.set_ylabel("P(concept as next token)", fontsize=12)
ax.set_title("Token as Pointer: Named Entities Activate Distinct Concept Distributions",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_yscale('log')
ax.set_ylim(1e-7, 0.05)
ax.grid(True, alpha=0.2, axis='y')

# Group separators and labels
for b, label in zip(group_boundaries, group_labels_bottom):
    ax.axvline(x=b - 0.5, color='#999', linestyle='-', alpha=0.4, linewidth=1)
    ax.text(b + 3, 0.03, label, fontsize=10, fontweight='bold', color='#555',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_bars_final.png", dpi=200, bbox_inches='tight',
            facecolor='white')
print("Saved: manifold_bars_final.png")
