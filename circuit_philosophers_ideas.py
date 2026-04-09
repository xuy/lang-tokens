#!/usr/bin/env python3
"""
Circuit-level manifold: Philosophers — IDEA-level concepts.
Instead of broad subjects (ethics, politics), use specific philosophical
ideas associated with individual thinkers. The pointer thesis predicts
that "Marx" activates "alienation" and "dialectical" while
"Kant" activates "categorical" and "transcendental".
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
# Idea-level concepts — grouped by thinker association
# ============================================================

CONCEPT_AXES = {
    "Kantian\nIdeas": [
        "categorical", "transcendental", "imperative", "noumenal",
        "synthetic", "autonomy", "maxim", "rational",
    ],
    "Nietzschean\nIdeas": [
        "nihilism", "eternal", "genealogy", "perspectivism",
        "tragic", "noble", "overcoming", "resentment",
    ],
    "Marxist\nIdeas": [
        "alienation", "dialectical", "surplus", "bourgeois",
        "proletariat", "exploitation", "materialism", "commodity",
    ],
    "Wittgensteinian\nIdeas": [
        "grammar", "proposition", "atomic", "resemblance",
        "silence", "rule", "beetle", "ordinary",
    ],
    "Existentialist\nIdeas": [
        "absurdity", "authenticity", "anguish", "nothingness",
        "phenomenology", "contingency", "facticity", "engagement",
    ],
    "Shared\nPhilosophy": [
        "freedom", "truth", "consciousness", "existence",
        "morality", "knowledge", "reason", "justice",
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
    print(f"{entity_name} | pos {entity_positions}")
    print(f"'{prompt}'")

    graph = attribute(prompt, model, attribution_targets=concept_targets, verbose=True)
    adj = graph.adjacency_matrix
    n_feat = graph.active_features.shape[0]
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    logit_start = n_feat + (n_layers * n_pos) + n_pos

    entity_feat_indices = [i for i in range(n_feat)
                           if graph.active_features[i][1].item() in entity_positions]
    print(f"  Entity features: {len(entity_feat_indices)}")

    concept_attributions = {}
    for ci, concept in enumerate(ALL_CONCEPTS):
        logit_row = logit_start + ci
        total_attr = sum(adj[logit_row, fi].item() for fi in entity_feat_indices)
        concept_attributions[concept] = total_attr

    results[entity_name] = concept_attributions
    sorted_ca = sorted(concept_attributions.items(), key=lambda x: -x[1])
    print(f"  Top: {[(c, f'{a:+.2f}') for c, a in sorted_ca[:8]]}")
    print(f"  Bot: {[(c, f'{a:+.2f}') for c, a in sorted_ca[-3:]]}")

# ============================================================
# Aggregate, analyze, plot
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

# Print
print("\n\nAXIS-LEVEL ATTRIBUTION:")
for entity in ENTITIES:
    print(f"\n{entity}:")
    for ax in axis_names:
        print(f"  {ax:25s}: {axis_attribution[entity][ax]:+.2f}")

print("\n\nDIFFERENTIAL vs random:")
for entity in axis_diff:
    print(f"\n{entity}:")
    for ax in axis_names:
        print(f"  {ax:25s}: {axis_diff[entity][ax]:+.2f}")

# Key test: does each thinker most activate their OWN idea cluster?
print("\n\n--- KEY TEST: Does each thinker activate their own ideas most? ---")
thinker_axis_map = {
    "Immanuel Kant": "Kantian\nIdeas",
    "Friedrich Nietzsche": "Nietzschean\nIdeas",
    "Karl Marx": "Marxist\nIdeas",
    "Ludwig Wittgenstein": "Wittgensteinian\nIdeas",
    "Jean-Paul Sartre": "Existentialist\nIdeas",
}
for entity, own_axis in thinker_axis_map.items():
    diff = axis_diff[entity]
    ranked = sorted(diff.items(), key=lambda x: -x[1])
    own_rank = next(i for i, (ax, _) in enumerate(ranked) if ax == own_axis) + 1
    own_val = diff[own_axis]
    top_ax, top_val = ranked[0]
    print(f"  {entity:25s}: own ideas rank #{own_rank} ({own_val:+.2f}), "
          f"top = {top_ax.replace(chr(10), ' ')} ({top_val:+.2f})")

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

# Differential radar
fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))
ax.set_facecolor('#fafafa')
ax.yaxis.grid(True, color='#ccc', linewidth=0.5)
ax.xaxis.grid(True, color='#ccc', linewidth=0.5)
ax.spines['polar'].set_visible(False)

for persona, color, ls, marker in radar_config:
    if persona not in axis_diff:
        continue
    vals = [axis_diff[persona][a] for a in axis_names]
    vals += vals[:1]
    short = persona.split()[-1]
    ax.plot(angles, vals, ls, linewidth=2.5, label=f"{short}",
            color=color, marker=marker, markersize=8, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5)
    ax.fill(angles, vals, alpha=0.05, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(axis_names, fontsize=11, fontweight='bold', color='#333')
ax.plot(angles, [0] * (n_axes + 1), 'k--', linewidth=0.8, alpha=0.5)
ax.set_title("Does Each Thinker Activate Their Own Ideas?\n"
             "(Circuit attribution, differential vs. random person)\n",
             fontsize=15, fontweight='bold', pad=25, color='#222')
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.08), fontsize=11,
          frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
plt.tight_layout()
plt.savefig("figures/circuit_philosophers_ideas_radar.png", dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: figures/circuit_philosophers_ideas_radar.png")

# Per-concept heatmap: entities × concepts
fig, ax_hm = plt.subplots(figsize=(20, 5))
entity_names_short = ["Kant", "Nietzsche", "Marx", "Wittgenstein", "Sartre", "random"]
entity_names_full = list(ENTITIES.keys())
mat = np.array([[results[e][c] for c in ALL_CONCEPTS] for e in entity_names_full])

im = ax_hm.imshow(mat, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                    vmin=-4, vmax=6)
ax_hm.set_yticks(range(len(entity_names_short)))
ax_hm.set_yticklabels(entity_names_short, fontsize=11)
ax_hm.set_xticks(range(len(ALL_CONCEPTS)))
ax_hm.set_xticklabels(ALL_CONCEPTS, rotation=70, ha='right', fontsize=8)

# Group separators
cumlen = 0
for ax_name, concepts in CONCEPT_AXES.items():
    ax_hm.axvline(x=cumlen - 0.5, color='white', linewidth=2)
    ax_hm.text(cumlen + len(concepts) / 2 - 0.5, -1.2, ax_name.replace('\n', ' '),
               ha='center', fontsize=9, fontweight='bold', color='#333')
    cumlen += len(concepts)

plt.colorbar(im, ax=ax_hm, label='Attribution', shrink=0.8)
ax_hm.set_title("Circuit Attribution: Philosopher × Specific Idea",
                 fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/circuit_philosophers_ideas_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved: figures/circuit_philosophers_ideas_heatmap.png")

# Save
save_data = {
    "model": MODEL_ID, "template": TEMPLATE, "domain": "philosophers_ideas",
    "concept_axes": {k.replace('\n', ' '): v for k, v in CONCEPT_AXES.items()},
    "per_concept": {e: {c: round(v, 4) for c, v in ca.items()} for e, ca in results.items()},
    "axis_attribution": {e: {k.replace('\n', ' '): round(v, 4) for k, v in aa.items()}
                         for e, aa in axis_attribution.items()},
    "axis_differential": {e: {k.replace('\n', ' '): round(v, 4) for k, v in ad.items()}
                          for e, ad in axis_diff.items()},
}
with open("data/circuit_philosophers_ideas_results.json", "w") as f:
    json.dump(save_data, f, indent=2)
print("Saved: data/circuit_philosophers_ideas_results.json")

print("\nDONE")
