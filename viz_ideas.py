#!/usr/bin/env python3
"""
Clean visualization of the philosopher × idea circuit attribution.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

with open("data/circuit_philosophers_ideas_results.json") as f:
    data = json.load(f)

per_concept = data["per_concept"]
concept_axes = data["concept_axes"]

# Build concept → axis mapping and axis → color mapping
concept_to_axis = {}
for ax, concepts in concept_axes.items():
    for c in concepts:
        concept_to_axis[c] = ax

axis_colors = {
    "Kantian Ideas": '#c0392b',
    "Nietzschean Ideas": '#e67e22',
    "Marxist Ideas": '#2980b9',
    "Wittgensteinian Ideas": '#27ae60',
    "Existentialist Ideas": '#8e44ad',
    "Shared Philosophy": '#95a5a6',
}

axis_short = {
    "Kantian Ideas": 'Kant',
    "Nietzschean Ideas": 'Nietzsche',
    "Marxist Ideas": 'Marx',
    "Wittgensteinian Ideas": 'Wittgenstein',
    "Existentialist Ideas": 'Existentialist',
    "Shared Philosophy": 'Shared',
}

# Subtract baseline
baseline = per_concept["a random person"]
thinkers = ["Immanuel Kant", "Friedrich Nietzsche", "Karl Marx",
            "Ludwig Wittgenstein", "Jean-Paul Sartre"]
short_names = ["Kant", "Nietzsche", "Marx", "Wittgenstein", "Sartre"]

# ============================================================
# Viz 1: Top concepts per thinker (horizontal bars, color-coded)
# ============================================================

N_TOP = 8

fig, axes = plt.subplots(len(thinkers), 1, figsize=(10, 12), sharex=False)
fig.subplots_adjust(hspace=0.55)

for ax, thinker, short in zip(axes, thinkers, short_names):
    # Differential attribution
    diffs = {c: per_concept[thinker][c] - baseline[c] for c in per_concept[thinker]}
    sorted_concepts = sorted(diffs.items(), key=lambda x: -x[1])

    # Take top N
    top = sorted_concepts[:N_TOP]
    concepts = [c for c, _ in top]
    values = [v for _, v in top]
    colors = [axis_colors[concept_to_axis[c]] for c in concepts]

    bars = ax.barh(range(N_TOP), values, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(N_TOP))
    ax.set_yticklabels(concepts, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(short, fontsize=13, fontweight='bold', loc='left', pad=5)
    ax.axvline(x=0, color='#333', linewidth=0.5)
    ax.set_xlim(-1, max(values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Label values
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}', va='center', fontsize=9, color='#333')

# Legend
legend_patches = [mpatches.Patch(color=c, label=axis_short[ax])
                  for ax, c in axis_colors.items() if ax != "Shared Philosophy"]
legend_patches.append(mpatches.Patch(color=axis_colors["Shared Philosophy"], label="Shared"))
fig.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=10,
           frameon=True, edgecolor='#ccc', fancybox=True,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle("What Ideas Does Each Name Activate?\n"
             "Top concepts by circuit attribution (vs. random person baseline)",
             fontsize=15, fontweight='bold', y=1.02)
plt.savefig("figures/ideas_top_concepts.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: figures/ideas_top_concepts.png")


# ============================================================
# Viz 2: "Ownership matrix" — does each thinker activate their own ideas?
# Simplified to: fraction of top-N concepts from own cluster vs. others
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

thinker_own_axis = {
    "Immanuel Kant": "Kantian Ideas",
    "Friedrich Nietzsche": "Nietzschean Ideas",
    "Karl Marx": "Marxist Ideas",
    "Ludwig Wittgenstein": "Wittgensteinian Ideas",
    "Jean-Paul Sartre": "Existentialist Ideas",
}

bar_width = 0.6
for i, (thinker, short) in enumerate(zip(thinkers, short_names)):
    diffs = {c: per_concept[thinker][c] - baseline[c] for c in per_concept[thinker]}
    sorted_concepts = sorted(diffs.items(), key=lambda x: -x[1])
    top_n = sorted_concepts[:10]

    own_axis = thinker_own_axis[thinker]
    own_count = sum(1 for c, _ in top_n if concept_to_axis[c] == own_axis)
    other_named = sum(1 for c, _ in top_n
                      if concept_to_axis[c] != own_axis and concept_to_axis[c] != "Shared Philosophy")
    shared_count = sum(1 for c, _ in top_n if concept_to_axis[c] == "Shared Philosophy")

    own_color = axis_colors[own_axis]
    ax.barh(i, own_count, bar_width, color=own_color, label='Own ideas' if i == 0 else "")
    ax.barh(i, other_named, bar_width, left=own_count, color='#bdc3c7',
            label="Other thinker's ideas" if i == 0 else "")
    ax.barh(i, shared_count, bar_width, left=own_count + other_named, color='#95a5a6',
            label='Shared concepts' if i == 0 else "")

    ax.text(own_count / 2, i, f'{own_count}', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

ax.set_yticks(range(len(short_names)))
ax.set_yticklabels(short_names, fontsize=12, fontweight='bold')
ax.set_xlabel("Number of concepts in top 10", fontsize=12)
ax.set_title("How Many of a Thinker's Top-10 Activated Concepts\nBelong to Their Own Idea Cluster?",
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 11)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("figures/ideas_ownership.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: figures/ideas_ownership.png")


# ============================================================
# Viz 3: Network-style — thinker nodes connected to concept nodes
# ============================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, len(thinkers) - 0.5)
ax.axis('off')

# Place thinkers on the left, concepts on the right
thinker_y = {short: i for i, short in enumerate(short_names)}

# For each thinker, get top 5 concepts and draw connections
all_shown_concepts = set()
concept_positions = {}  # concept -> y position

# First pass: collect all concepts to show
for thinker, short in zip(thinkers, short_names):
    diffs = {c: per_concept[thinker][c] - baseline[c] for c in per_concept[thinker]}
    sorted_concepts = sorted(diffs.items(), key=lambda x: -x[1])
    for c, v in sorted_concepts[:6]:
        all_shown_concepts.add(c)

# Assign y positions to concepts
shown_sorted = sorted(all_shown_concepts)
# Group by axis
grouped = {}
for c in shown_sorted:
    ax_name = concept_to_axis[c]
    grouped.setdefault(ax_name, []).append(c)

y_pos = 0
spacing = (len(thinkers) - 1) / max(len(all_shown_concepts) - 1, 1)
for ax_name in axis_colors:
    if ax_name in grouped:
        for c in grouped[ax_name]:
            concept_positions[c] = y_pos * spacing
            y_pos += 1

# Normalize concept y positions to fit
if concept_positions:
    max_cy = max(concept_positions.values())
    min_cy = min(concept_positions.values())
    if max_cy > min_cy:
        for c in concept_positions:
            concept_positions[c] = (concept_positions[c] - min_cy) / (max_cy - min_cy) * (len(thinkers) - 1)

# Draw
for thinker, short in zip(thinkers, short_names):
    ty = thinker_y[short]
    own_axis = thinker_own_axis[thinker]
    color = axis_colors[own_axis]

    # Thinker node
    ax.text(0.5, ty, short, fontsize=13, fontweight='bold', ha='right', va='center',
            color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.15, edgecolor=color))

    # Top concepts with connections
    diffs = {c: per_concept[thinker][c] - baseline[c] for c in per_concept[thinker]}
    sorted_concepts = sorted(diffs.items(), key=lambda x: -x[1])

    for c, v in sorted_concepts[:6]:
        if c not in concept_positions:
            continue
        cy = concept_positions[c]
        c_color = axis_colors[concept_to_axis[c]]
        alpha = min(1.0, max(0.2, v / 8.0))
        lw = min(4.0, max(0.5, v / 2.0))

        ax.plot([1.0, 9.0], [ty, cy], color=c_color, alpha=alpha, linewidth=lw, zorder=1)

# Draw concept nodes on the right
for c, cy in concept_positions.items():
    c_color = axis_colors[concept_to_axis[c]]
    ax.text(9.5, cy, c, fontsize=10, ha='left', va='center', color=c_color, fontweight='bold')
    ax.scatter([9.3], [cy], c=c_color, s=40, zorder=5, edgecolors='white', linewidths=0.5)

ax.set_title("Named Entity → Idea Activation Network\n"
             "(line thickness = circuit attribution strength, differential vs. random)",
             fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/ideas_network.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: figures/ideas_network.png")
