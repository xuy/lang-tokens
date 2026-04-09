#!/usr/bin/env python3
"""
Circuit-level manifold: Philosophers domain.
Same methodology as the investor experiment — circuit attribution from
entity features to concept logit directions — applied to philosophers.
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

DEVICE = "mps"
MODEL_ID = "google/gemma-2-2b"
TRANSCODER = "mntss/gemma-scope-transcoders"

print(f"Loading {MODEL_ID}...")
model = ReplacementModel.from_pretrained(MODEL_ID, TRANSCODER, device=DEVICE)
W_U = model.W_U
tokenizer = model.tokenizer
print("Loaded.")

# ============================================================
# Philosopher concepts — 6 axes of philosophical thought
# ============================================================

CONCEPT_AXES = {
    "Ethics &\nMorality": [
        "virtue", "duty", "morality", "justice", "ethics",
        "happiness", "good", "evil",
    ],
    "Epistemology &\nReason": [
        "knowledge", "reason", "truth", "logic", "empirical",
        "certainty", "skepticism", "rationality",
    ],
    "Metaphysics &\nExistence": [
        "existence", "consciousness", "reality", "being",
        "freedom", "absurdity", "nothingness", "essence",
    ],
    "Politics &\nSociety": [
        "power", "state", "revolution", "equality",
        "liberty", "democracy", "oppression", "class",
    ],
    "Language &\nMind": [
        "language", "meaning", "mind", "intention",
        "perception", "representation", "interpretation", "concept",
    ],
    "Science &\nMethod": [
        "science", "observation", "hypothesis", "evidence",
        "experiment", "causation", "induction", "falsification",
    ],
}

ALL_CONCEPTS = [c for concepts in CONCEPT_AXES.values() for c in concepts]
CONCEPT_TO_AXIS = {c: ax for ax, concepts in CONCEPT_AXES.items() for c in concepts}

ENTITIES = {
    "Immanuel Kant": "Immanuel Kant",
    "Friedrich Nietzsche": "Friedrich Nietzsche",
    "Karl Marx": "Karl Marx",
    "Ludwig Wittgenstein": "Ludwig Wittgenstein",
    "Jean-Paul Sartre": "Jean-Paul Sartre",
    "a random person": "a random person",
}

TEMPLATE = "The philosophy of {entity} is fundamentally about"


def get_concept_targets():
    targets = []
    for concept in ALL_CONCEPTS:
        token_ids = tokenizer.encode(" " + concept, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode(concept, add_special_tokens=False)
        tid = token_ids[0]
        unembed_vec = W_U[:, tid].detach().clone()
        targets.append(CustomTarget(token_str=concept, prob=1.0 / len(ALL_CONCEPTS), vec=unembed_vec))
    return targets


def find_entity_positions(prompt, entity):
    tokens = tokenizer.encode(prompt)
    for prefix in [" ", ""]:
        entity_toks = tokenizer.encode(prefix + entity, add_special_tokens=False)
        for i in range(len(tokens) - len(entity_toks) + 1):
            if tokens[i:i + len(entity_toks)] == entity_toks:
                return list(range(i, i + len(entity_toks)))
    return list(range(1, min(4, len(tokens))))


concept_targets = get_concept_targets()
print(f"Built {len(concept_targets)} concept targets")

results = {}
axis_names = list(CONCEPT_AXES.keys())

for entity_name, entity_str in ENTITIES.items():
    prompt = TEMPLATE.format(entity=entity_str)
    entity_positions = find_entity_positions(prompt, entity_str)
    print(f"\n{'='*60}")
    print(f"{entity_name} | positions {entity_positions}")
    print(f"'{prompt}'")

    graph = attribute(prompt, model, attribution_targets=concept_targets, verbose=True)
    adj = graph.adjacency_matrix
    n_feat = graph.active_features.shape[0]
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    n_errors = n_layers * n_pos
    logit_start = n_feat + n_errors + n_pos

    entity_feat_indices = [i for i in range(n_feat)
                           if graph.active_features[i][1].item() in entity_positions]
    print(f"  Features at entity: {len(entity_feat_indices)}")

    concept_attributions = {}
    for ci, concept in enumerate(ALL_CONCEPTS):
        logit_row = logit_start + ci
        total_attr = sum(adj[logit_row, fi].item() for fi in entity_feat_indices)
        concept_attributions[concept] = total_attr

    results[entity_name] = concept_attributions
    sorted_ca = sorted(concept_attributions.items(), key=lambda x: -x[1])
    print(f"  Top: {[(c, f'{a:+.2f}') for c, a in sorted_ca[:6]]}")
    print(f"  Bot: {[(c, f'{a:+.2f}') for c, a in sorted_ca[-3:]]}")

# ============================================================
# Aggregate and plot
# ============================================================

axis_attribution = {}
for entity in ENTITIES:
    axis_attribution[entity] = {}
    for ax, concepts in CONCEPT_AXES.items():
        axis_attribution[entity][ax] = sum(results[entity][c] for c in concepts)

baseline = "a random person"
axis_diff = {}
for entity in ENTITIES:
    if entity == baseline:
        continue
    axis_diff[entity] = {ax: axis_attribution[entity][ax] - axis_attribution[baseline][ax]
                         for ax in axis_names}

print("\n\nAXIS-LEVEL ATTRIBUTION:")
for entity in ENTITIES:
    print(f"\n{entity}:")
    for ax in axis_names:
        print(f"  {ax:25s}: {axis_attribution[entity][ax]:+.2f}")

# Radar
n_axes = len(axis_names)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

radar_config = [
    ("Immanuel Kant", '#c0392b', '-', 'o'),
    ("Friedrich Nietzsche", '#e67e22', '-', 's'),
    ("Karl Marx", '#2980b9', '-', '^'),
    ("Ludwig Wittgenstein", '#27ae60', '-', 'D'),
    ("Jean-Paul Sartre", '#8e44ad', '-', 'v'),
    ("a random person", '#7f8c8d', '--', 'X'),
]

# Raw
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_facecolor('#fafafa')
ax.yaxis.grid(True, color='#ccc', linewidth=0.5)
ax.xaxis.grid(True, color='#ccc', linewidth=0.5)
ax.spines['polar'].set_visible(False)

for persona, color, ls, marker in radar_config:
    vals = [axis_attribution[persona][a] for a in axis_names]
    vals += vals[:1]
    ax.plot(angles, vals, ls, linewidth=2.5, label=persona, color=color,
            marker=marker, markersize=8, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5)
    ax.fill(angles, vals, alpha=0.05, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(axis_names, fontsize=11, fontweight='bold', color='#333')
ax.set_title("Circuit Attribution: Philosopher → Concept\n"
             "(direct effects from entity features to concept logits)\n",
             fontsize=15, fontweight='bold', pad=25, color='#222')
ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.08), fontsize=10,
          frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
plt.tight_layout()
plt.savefig("figures/circuit_philosophers_radar.png", dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: figures/circuit_philosophers_radar.png")

# Differential
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
    ax.plot(angles, vals, ls, linewidth=2.5, label=f"{persona.split()[-1]} − random",
            color=color, marker=marker, markersize=8, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5)
    ax.fill(angles, vals, alpha=0.05, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(axis_names, fontsize=11, fontweight='bold', color='#333')
zero_vals = [0] * (n_axes + 1)
ax.plot(angles, zero_vals, 'k--', linewidth=0.8, alpha=0.5)
ax.set_title("Differential Circuit Attribution vs. Random Person\n(Philosophers)\n",
             fontsize=15, fontweight='bold', pad=25, color='#222')
ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.08), fontsize=10,
          frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
plt.tight_layout()
plt.savefig("figures/circuit_philosophers_differential.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: figures/circuit_philosophers_differential.png")

# Save
save_data = {
    "model": MODEL_ID, "template": TEMPLATE, "domain": "philosophers",
    "concept_axes": {k.replace('\n', ' '): v for k, v in CONCEPT_AXES.items()},
    "per_concept": {e: {c: round(v, 4) for c, v in ca.items()} for e, ca in results.items()},
    "axis_attribution": {e: {k.replace('\n', ' '): round(v, 4) for k, v in aa.items()}
                         for e, aa in axis_attribution.items()},
    "axis_differential": {e: {k.replace('\n', ' '): round(v, 4) for k, v in ad.items()}
                          for e, ad in axis_diff.items()},
}
with open("data/circuit_philosophers_results.json", "w") as f:
    json.dump(save_data, f, indent=2)
print("Saved: data/circuit_philosophers_results.json")

print("\nDONE")
