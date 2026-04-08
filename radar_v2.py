"""
Radar chart v2: 6 meaningful concept axes instead of 4.
Drop the uninformative "General/Neutral" bucket and split into
more specific investment philosophy dimensions.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load raw per-concept probabilities
with open("/Users/x/src/lang-tokens/manifold_results.json") as f:
    results = json.load(f)

persona_avg_probs = results["persona_avg_probs"]

# New 6-axis grouping — each axis is a recognizable investment dimension
concept_axes = {
    "Value\nAnalysis": [
        "value", "intrinsic", "undervalued", "moat", "margin",
        "fundamental", "earnings", "book",
    ],
    "Long-term\nCompounding": [
        "compounding", "patience", "durable", "ownership",
        "dividends", "conservative", "quality",
    ],
    "Macro &\nGeopolitics": [
        "currency", "macro", "sovereign", "geopolitical",
        "reflexivity", "global", "crisis", "emerging",
    ],
    "Risk &\nHedging": [
        "hedge", "derivatives", "options", "leverage",
        "volatility", "arbitrage", "speculation",
    ],
    "Quantitative\nMethods": [
        "algorithmic", "quantitative", "statistical",
        "systematic", "correlation", "signals", "frequency", "momentum",
    ],
    "Growth &\nMarkets": [
        "growth", "stocks", "market", "capital",
        "portfolio", "index", "returns",
    ],
}

# Compute group mass for each persona
axis_names = list(concept_axes.keys())
n_axes = len(axis_names)

personas_to_plot = [
    ("Warren Buffett", '#c0392b', '-', 'o', 2.8),
    ("George Soros", '#e67e22', '-', 's', 2.8),
    ("Jim Simons", '#2980b9', '-', '^', 2.8),
    ("a random person", '#7f8c8d', '--', 'D', 2.0),
]

# Compute normalized mass per axis
persona_axis_mass = {}
for persona, *_ in personas_to_plot:
    total = 0
    axis_totals = {}
    for axis, concepts in concept_axes.items():
        s = sum(persona_avg_probs[persona].get(c, 0) for c in concepts)
        axis_totals[axis] = s
        total += s
    # Normalize so axes sum to 1
    persona_axis_mass[persona] = {ax: v / total if total > 0 else 0
                                   for ax, v in axis_totals.items()}

# Print for reference
for persona, *_ in personas_to_plot:
    print(f"\n{persona}:")
    for ax in axis_names:
        bar = "#" * int(persona_axis_mass[persona][ax] * 100)
        print(f"  {ax:20s}: {persona_axis_mass[persona][ax]:.3f}  {bar}")

# Build radar
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_facecolor('#fafafa')

# Grid styling
ax.set_rlabel_position(30)
ax.yaxis.grid(True, color='#cccccc', linewidth=0.5)
ax.xaxis.grid(True, color='#cccccc', linewidth=0.5)
ax.spines['polar'].set_visible(False)

for persona, color, ls, marker, lw in personas_to_plot:
    values = [persona_axis_mass[persona][ax] for ax in axis_names]
    values += values[:1]
    ax.plot(angles, values, ls, linewidth=lw, label=persona, color=color,
            marker=marker, markersize=9, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.fill(angles, values, alpha=0.07, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(axis_names, fontsize=12, fontweight='bold', color='#333')

# Radial ticks — adjusted for the new scale
max_val = max(persona_axis_mass[p][ax]
              for p, *_ in personas_to_plot
              for ax in axis_names)
tick_step = 0.05
ticks = np.arange(tick_step, max_val + tick_step * 2, tick_step)
ax.set_yticks(ticks)
ax.set_yticklabels([f'{t:.0%}' for t in ticks], fontsize=8, color='#666')
ax.set_ylim(0, max_val + 0.05)

ax.set_title("Where Does the Probability Mass Land?\n"
             "Concept activation profile by entity name\n",
             fontsize=16, fontweight='bold', pad=25, color='#222')

legend = ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08), fontsize=12,
                   frameon=True, fancybox=True, shadow=False,
                   edgecolor='#cccccc', facecolor='white')
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_radar_final.png", dpi=200, bbox_inches='tight',
            facecolor='white')
print("\nSaved: manifold_radar_final.png")
