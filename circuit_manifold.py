#!/usr/bin/env python3
"""
Circuit-level manifold: Use Anthropic's circuit-tracer to measure how much
attribution flows from entity tokens to each investment concept.

For each entity: run attribution on "The investment philosophy of {entity} is about"
with custom targets = concept token unembed vectors.
Sum attribution from entity-position features to each concept target.
This gives: "how much does the entity causally drive each concept in the output?"
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.attribution.targets import CustomTarget
import json
import warnings
import os
warnings.filterwarnings('ignore')
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"

# ============================================================
# Setup
# ============================================================

DEVICE = "mps"
MODEL_ID = "google/gemma-2-2b"
TRANSCODER = "mntss/gemma-scope-transcoders"

print(f"Loading {MODEL_ID}...")
model = ReplacementModel.from_pretrained(MODEL_ID, TRANSCODER, device=DEVICE)
print("Loaded.")

# Access unembed and tokenizer directly from the ReplacementModel
W_U = model.W_U  # [d_model, d_vocab] unembed matrix
tokenizer = model.tokenizer

# ============================================================
# Concepts and entities
# ============================================================

CONCEPT_AXES = {
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

ALL_CONCEPTS = [c for concepts in CONCEPT_AXES.values() for c in concepts]
CONCEPT_TO_AXIS = {c: ax for ax, concepts in CONCEPT_AXES.items() for c in concepts}

ENTITIES = {
    "Warren Buffett": "Warren Buffett",
    "George Soros": "George Soros",
    "Jim Simons": "Jim Simons",
    "a random person": "a random person",
}

TEMPLATE = "The investment philosophy of {entity} is about"


def get_concept_targets():
    """Build CustomTarget for each concept using its unembed vector."""
    targets = []
    for concept in ALL_CONCEPTS:
        # Get token id for the concept (with space prefix for typical GPT tokenization)
        token_ids = tokenizer.encode(" " + concept, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode(concept, add_special_tokens=False)
        tid = token_ids[0]
        # Get the unembed direction for this token
        unembed_vec = W_U[:, tid].detach().clone()
        targets.append(CustomTarget(
            token_str=concept,
            prob=1.0 / len(ALL_CONCEPTS),  # equal weight
            vec=unembed_vec,
        ))
    return targets


def find_entity_positions(prompt, entity):
    """Find token positions of the entity in the prompt."""
    tokens = tokenizer.encode(prompt)
    for prefix in [" ", ""]:
        entity_toks = tokenizer.encode(prefix + entity, add_special_tokens=False)
        for i in range(len(tokens) - len(entity_toks) + 1):
            if tokens[i:i + len(entity_toks)] == entity_toks:
                return list(range(i, i + len(entity_toks)))
    # Fallback
    return list(range(1, min(4, len(tokens))))


# ============================================================
# Run attribution for each entity
# ============================================================

concept_targets = get_concept_targets()
print(f"\nBuilt {len(concept_targets)} concept targets")
print(f"Sample: {concept_targets[0].token_str} -> vec shape {concept_targets[0].vec.shape}")

results = {}

for entity_name, entity_str in ENTITIES.items():
    prompt = TEMPLATE.format(entity=entity_str)
    print(f"\n{'='*60}")
    print(f"Entity: {entity_name}")
    print(f"Prompt: '{prompt}'")

    entity_positions = find_entity_positions(prompt, entity_str)
    print(f"Entity positions: {entity_positions}")

    # Run attribution with concept targets
    print("Running attribution...")
    graph = attribute(
        prompt, model,
        attribution_targets=concept_targets,
        verbose=True,
    )
    print(f"Graph: {graph.active_features.shape[0]} features, {graph.adjacency_matrix.shape}")

    adj = graph.adjacency_matrix
    n_feat = graph.active_features.shape[0]
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    n_errors = n_layers * n_pos
    error_end = n_feat + n_errors
    embed_end = error_end + n_pos
    logit_start = embed_end

    # For each concept target, sum attribution FROM entity-position features
    # adj[target_row, source_col] convention
    entity_feat_indices = [i for i in range(n_feat)
                           if graph.active_features[i][1].item() in entity_positions]
    print(f"Features at entity positions: {len(entity_feat_indices)}")

    concept_attributions = {}
    for ci, concept in enumerate(ALL_CONCEPTS):
        logit_row = logit_start + ci
        # Sum direct effects from entity-position features to this concept logit
        total_attr = sum(adj[logit_row, fi].item() for fi in entity_feat_indices)
        concept_attributions[concept] = total_attr

    results[entity_name] = concept_attributions

    # Print top/bottom
    sorted_ca = sorted(concept_attributions.items(), key=lambda x: -x[1])
    print(f"\n  Top attributed concepts:")
    for c, a in sorted_ca[:8]:
        print(f"    {c:20s}: {a:+.4f}  ({CONCEPT_TO_AXIS[c]})")
    print(f"  Bottom attributed:")
    for c, a in sorted_ca[-5:]:
        print(f"    {c:20s}: {a:+.4f}  ({CONCEPT_TO_AXIS[c]})")

# ============================================================
# Build radar from circuit attribution
# ============================================================

axis_names = list(CONCEPT_AXES.keys())

# Aggregate by axis: sum attributions per concept group
axis_attribution = {}
for entity in ENTITIES:
    axis_attribution[entity] = {}
    for ax, concepts in CONCEPT_AXES.items():
        axis_attribution[entity][ax] = sum(results[entity][c] for c in concepts)

# Print
print("\n\n" + "="*60)
print("AXIS-LEVEL ATTRIBUTION")
print("="*60)
for entity in ENTITIES:
    print(f"\n{entity}:")
    for ax in axis_names:
        val = axis_attribution[entity][ax]
        print(f"  {ax:25s}: {val:+.4f}")

# Differential vs baseline
baseline = "a random person"
axis_diff = {}
for entity in ENTITIES:
    if entity == baseline:
        continue
    axis_diff[entity] = {}
    for ax in axis_names:
        axis_diff[entity][ax] = axis_attribution[entity][ax] - axis_attribution[baseline][ax]

# ============================================================
# Radar chart
# ============================================================

n_axes = len(axis_names)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

radar_config = [
    ("Warren Buffett", '#c0392b', '-', 'o'),
    ("George Soros", '#e67e22', '-', 's'),
    ("Jim Simons", '#2980b9', '-', '^'),
    ("a random person", '#7f8c8d', '--', 'D'),
]

# Raw attribution radar
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_facecolor('#fafafa')
ax.yaxis.grid(True, color='#ccc', linewidth=0.5)
ax.xaxis.grid(True, color='#ccc', linewidth=0.5)
ax.spines['polar'].set_visible(False)

for persona, color, ls, marker in radar_config:
    vals = [axis_attribution[persona][a] for a in axis_names]
    vals += vals[:1]
    ax.plot(angles, vals, ls, linewidth=2.8, label=persona, color=color,
            marker=marker, markersize=9, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5)
    ax.fill(angles, vals, alpha=0.07, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(axis_names, fontsize=12, fontweight='bold', color='#333')
ax.set_title("Circuit Attribution: Entity → Concept\n"
             "(sum of direct effects from entity features to concept logits)\n",
             fontsize=15, fontweight='bold', pad=25, color='#222')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.08), fontsize=12,
          frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
plt.tight_layout()
plt.savefig("figures/circuit_manifold_radar.png", dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: figures/circuit_manifold_radar.png")

# Differential radar (entity - baseline)
if axis_diff:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor('#fafafa')
    ax.yaxis.grid(True, color='#ccc', linewidth=0.5)
    ax.xaxis.grid(True, color='#ccc', linewidth=0.5)
    ax.spines['polar'].set_visible(False)

    for persona, color, ls, marker in radar_config:
        if persona not in axis_diff:
            continue
        vals = [axis_diff[persona][a] for a in axis_names]
        vals += vals[:1]
        ax.plot(angles, vals, ls, linewidth=2.8, label=f"{persona} − random", color=color,
                marker=marker, markersize=9, markeredgecolor='white',
                markeredgewidth=1.5, zorder=5)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=12, fontweight='bold', color='#333')
    ax.set_title("Differential Circuit Attribution vs. Random Person\n"
                 "(positive = entity contributes MORE to this concept)\n",
                 fontsize=15, fontweight='bold', pad=25, color='#222')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.08), fontsize=12,
              frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
    # Add zero circle
    zero_vals = [0] * (n_axes + 1)
    ax.plot(angles, zero_vals, 'k--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/circuit_manifold_differential.png", dpi=200, bbox_inches='tight', facecolor='white')
    print("Saved: figures/circuit_manifold_differential.png")

# Per-concept bar chart
fig, ax_plot = plt.subplots(figsize=(18, 7))
entities_to_plot = list(ENTITIES.keys())
colors_bar = ['#c0392b', '#e67e22', '#2980b9', '#7f8c8d']
x = np.arange(len(ALL_CONCEPTS))
width = 0.2

for i, (entity, color) in enumerate(zip(entities_to_plot, colors_bar)):
    vals = [results[entity][c] for c in ALL_CONCEPTS]
    ax_plot.bar(x + i * width - width * 1.5, vals, width, label=entity, color=color, alpha=0.85)

ax_plot.set_xticks(x)
ax_plot.set_xticklabels(ALL_CONCEPTS, rotation=70, ha='right', fontsize=8)
ax_plot.set_ylabel("Circuit attribution (entity features → concept logit)", fontsize=11)
ax_plot.set_title("Per-Concept Circuit Attribution from Entity Tokens", fontsize=14, fontweight='bold')
ax_plot.legend(fontsize=10)
ax_plot.grid(True, alpha=0.2, axis='y')
ax_plot.axhline(y=0, color='black', linewidth=0.5)

cumlen = 0
for ax_name, concepts in CONCEPT_AXES.items():
    ax_plot.axvline(x=cumlen - 0.5, color='#999', linestyle='-', alpha=0.4)
    cumlen += len(concepts)

plt.tight_layout()
plt.savefig("figures/circuit_manifold_bars.png", dpi=150, bbox_inches='tight')
print("Saved: figures/circuit_manifold_bars.png")

# Save results
save_data = {
    "model": MODEL_ID,
    "template": TEMPLATE,
    "concept_axes": {k.replace('\n', ' '): v for k, v in CONCEPT_AXES.items()},
    "per_concept": {e: {c: round(v, 4) for c, v in ca.items()} for e, ca in results.items()},
    "axis_attribution": {e: {k.replace('\n', ' '): round(v, 4) for k, v in aa.items()}
                         for e, aa in axis_attribution.items()},
    "axis_differential": {e: {k.replace('\n', ' '): round(v, 4) for k, v in ad.items()}
                          for e, ad in axis_diff.items()},
}
with open("data/circuit_manifold_results.json", "w") as f:
    json.dump(save_data, f, indent=2)
print("Saved: data/circuit_manifold_results.json")

print("\n" + "="*60)
print("CIRCUIT MANIFOLD EXPERIMENT COMPLETE")
print("="*60)
