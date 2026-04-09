#!/usr/bin/env python3
"""
Activation-space manifold: measure proximity between entity representations
and concept representations directly in the residual stream.

Instead of "what does the model predict next?", this asks:
"How close is Buffett's internal representation to the representation of 'moat'?"

This is the pointer claim directly — the token activates a region in
representation space that is geometrically near associated concepts.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Setup
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "google/gemma-2-2b"
print(f"Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, output_hidden_states=True,
).to(DEVICE)
model.eval()
num_layers = model.config.num_hidden_layers
print(f"Loaded: {num_layers} layers")

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

# Multiple phrasings for each entity to average out context effects
ENTITY_PROMPTS = {
    "Warren Buffett": [
        "Warren Buffett",
        "Think like Warren Buffett",
        "The philosophy of Warren Buffett",
        "As Warren Buffett once said",
        "In the style of Warren Buffett",
    ],
    "George Soros": [
        "George Soros",
        "Think like George Soros",
        "The philosophy of George Soros",
        "As George Soros once said",
        "In the style of George Soros",
    ],
    "Jim Simons": [
        "Jim Simons",
        "Think like Jim Simons",
        "The philosophy of Jim Simons",
        "As Jim Simons once said",
        "In the style of Jim Simons",
    ],
    "a random person": [
        "a random person",
        "Think like a random person",
        "The philosophy of a random person",
        "As a random person once said",
        "In the style of a random person",
    ],
}

# ============================================================
# Extract activations
# ============================================================

def get_activation(text, target_text, layer):
    """Get the hidden state at the last token of target_text within text, at given layer."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    full_tokens = inputs["input_ids"][0].tolist()

    # Find target token position
    pos = len(full_tokens) - 1  # fallback: last token
    for prefix in ["", " "]:
        target_ids = tokenizer.encode(prefix + target_text, add_special_tokens=False)
        for i in range(len(full_tokens) - len(target_ids) + 1):
            if full_tokens[i:i + len(target_ids)] == target_ids:
                pos = i + len(target_ids) - 1
                break

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.hidden_states[layer][0, pos, :].cpu().float().numpy()


def get_concept_activation(concept, layer):
    """Get the activation for a concept word.
    We embed it in a neutral context to get a contextual representation."""
    # Use multiple neutral contexts and average
    contexts = [
        f"The concept of {concept}",
        f"When we talk about {concept}",
        f"The word {concept} refers to",
    ]
    acts = []
    for ctx in contexts:
        acts.append(get_activation(ctx, concept, layer))
    return np.mean(acts, axis=0)


# ============================================================
# Collect activations at multiple layers
# ============================================================

test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]
# That's layers 0, 6, 13, 19, 26

print(f"\nTest layers: {test_layers}")

# Get concept activations
print("\nExtracting concept activations...")
concept_acts = {}  # {layer: {concept: vector}}
for layer in test_layers:
    concept_acts[layer] = {}
    for concept in ALL_CONCEPTS:
        concept_acts[layer][concept] = get_concept_activation(concept, layer)
    print(f"  Layer {layer}: {len(ALL_CONCEPTS)} concepts done")

# Get entity activations (averaged over prompts)
print("\nExtracting entity activations...")
entity_acts = {}  # {layer: {entity: vector}}
for layer in test_layers:
    entity_acts[layer] = {}
    for entity, prompts in ENTITY_PROMPTS.items():
        acts = []
        for prompt in prompts:
            acts.append(get_activation(prompt, entity, layer))
        entity_acts[layer][entity] = np.mean(acts, axis=0)
        print(f"  Layer {layer}, {entity}: averaged {len(prompts)} prompts")

# ============================================================
# Compute entity-concept similarities
# ============================================================

print("\n" + "=" * 60)
print("RESULTS: Entity-Concept Cosine Similarity in Activation Space")
print("=" * 60)

results = {}

for layer in test_layers:
    print(f"\n--- Layer {layer} ---")
    results[layer] = {}

    for entity in ENTITY_PROMPTS:
        e_vec = entity_acts[layer][entity].reshape(1, -1)
        sims = {}
        for concept in ALL_CONCEPTS:
            c_vec = concept_acts[layer][concept].reshape(1, -1)
            sims[concept] = float(cosine_similarity(e_vec, c_vec)[0, 0])

        results[layer][entity] = sims

        # Print top and bottom
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        top5 = sorted_sims[:5]
        bot5 = sorted_sims[-3:]
        print(f"\n  {entity}:")
        print(f"    Closest:  {[(c, f'{s:.3f}') for c, s in top5]}")
        print(f"    Farthest: {[(c, f'{s:.3f}') for c, s in bot5]}")


# ============================================================
# Compute axis-level aggregated similarity (for radar)
# ============================================================

axis_names = list(CONCEPT_AXES.keys())

# For the radar, use the middle layer where representations are richest
radar_layer = num_layers // 2
print(f"\n\nRadar chart layer: {radar_layer}")

axis_similarity = {}  # {entity: {axis: avg_cosine_sim}}
for entity in ENTITY_PROMPTS:
    axis_similarity[entity] = {}
    for axis, concepts in CONCEPT_AXES.items():
        sims = [results[radar_layer][entity][c] for c in concepts]
        axis_similarity[entity][axis] = np.mean(sims)

    print(f"\n{entity}:")
    for ax in axis_names:
        bar = "#" * int(axis_similarity[entity][ax] * 100)
        print(f"  {ax:25s}: {axis_similarity[entity][ax]:.4f}  {bar}")

# Compute differential: entity similarity minus random-person similarity
# This isolates what's EXTRA for the named entity
baseline = "a random person"
axis_differential = {}
for entity in ENTITY_PROMPTS:
    if entity == baseline:
        continue
    axis_differential[entity] = {}
    for ax in axis_names:
        axis_differential[entity][ax] = axis_similarity[entity][ax] - axis_similarity[baseline][ax]

print("\n--- Differential (entity minus random person baseline) ---")
for entity in axis_differential:
    print(f"\n  {entity}:")
    for ax in axis_names:
        val = axis_differential[entity][ax]
        sign = "+" if val > 0 else ""
        print(f"    {ax:25s}: {sign}{val:.4f}")


# ============================================================
# Radar: Raw similarity
# ============================================================

def make_radar(data, personas, title, filename, ylabel="Avg cosine similarity"):
    n_axes = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    radar_config = [
        ("Warren Buffett", '#c0392b', '-', 'o'),
        ("George Soros", '#e67e22', '-', 's'),
        ("Jim Simons", '#2980b9', '-', '^'),
        ("a random person", '#7f8c8d', '--', 'D'),
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor('#fafafa')
    ax.yaxis.grid(True, color='#ccc', linewidth=0.5)
    ax.xaxis.grid(True, color='#ccc', linewidth=0.5)
    ax.spines['polar'].set_visible(False)

    for persona, color, ls, marker in radar_config:
        if persona not in data:
            continue
        vals = [data[persona][ax] for ax in axis_names]
        vals += vals[:1]
        ax.plot(angles, vals, ls, linewidth=2.8, label=persona, color=color,
                marker=marker, markersize=9, markeredgecolor='white',
                markeredgewidth=1.5, zorder=5)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=12, fontweight='bold', color='#333')

    all_vals = [data[p][ax] for p in data for ax in axis_names]
    vmin, vmax = min(all_vals), max(all_vals)
    padding = (vmax - vmin) * 0.15
    ticks = np.linspace(vmin - padding, vmax + padding, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:.3f}' for t in ticks], fontsize=8, color='#666')
    ax.set_ylim(vmin - padding, vmax + padding)

    ax.set_title(title + "\n", fontsize=16, fontweight='bold', pad=25, color='#222')
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08), fontsize=12,
              frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: figures/{filename}")


make_radar(
    axis_similarity,
    list(ENTITY_PROMPTS.keys()),
    f"Activation-Space Proximity to Concept Clusters\n(Layer {radar_layer}, cosine similarity)",
    "activation_manifold_radar.png",
)

# ============================================================
# Radar: Differential (entity - baseline)
# ============================================================

make_radar(
    axis_differential,
    list(axis_differential.keys()),
    f"Differential Proximity vs. Random Person Baseline\n(Layer {radar_layer}, cosine similarity delta)",
    "activation_manifold_differential.png",
)

# ============================================================
# Per-concept bar chart at the radar layer
# ============================================================

fig, ax = plt.subplots(figsize=(18, 7))
entities_to_plot = ["Warren Buffett", "George Soros", "Jim Simons", "a random person"]
colors_bar = ['#c0392b', '#e67e22', '#2980b9', '#7f8c8d']
x = np.arange(len(ALL_CONCEPTS))
width = 0.2

for i, (entity, color) in enumerate(zip(entities_to_plot, colors_bar)):
    sims = [results[radar_layer][entity][c] for c in ALL_CONCEPTS]
    ax.bar(x + i * width - width * 1.5, sims, width, label=entity, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(ALL_CONCEPTS, rotation=70, ha='right', fontsize=8)
ax.set_ylabel("Cosine similarity to entity activation", fontsize=12)
ax.set_title(f"Activation-Space Proximity: Entity → Concept (Layer {radar_layer})",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.2, axis='y')

# Group separators
cumlen = 0
for ax_name, concepts in CONCEPT_AXES.items():
    ax.axvline(x=cumlen - 0.5, color='#999', linestyle='-', alpha=0.4)
    ax.text(cumlen + len(concepts) / 2, ax.get_ylim()[1], ax_name.replace('\n', ' '),
            fontsize=9, fontweight='bold', color='#555', ha='center', va='bottom')
    cumlen += len(concepts)

plt.tight_layout()
plt.savefig("figures/activation_manifold_bars.png", dpi=150, bbox_inches='tight')
print("Saved: figures/activation_manifold_bars.png")

# ============================================================
# Layer progression: how does the manifold emerge?
# ============================================================

fig, axes = plt.subplots(1, len(test_layers), figsize=(5 * len(test_layers), 5))

for ax_plot, layer in zip(axes, test_layers):
    for entity, color in zip(entities_to_plot, colors_bar):
        sims_by_axis = []
        for ax_name, concepts in CONCEPT_AXES.items():
            sims_by_axis.append(np.mean([results[layer][entity][c] for c in concepts]))
        short_labels = [a.split('\n')[0] for a in axis_names]
        ax_plot.bar(np.arange(len(axis_names)) + entities_to_plot.index(entity) * 0.2 - 0.3,
                    sims_by_axis, 0.2, label=entity if layer == test_layers[0] else "",
                    color=color, alpha=0.85)
    ax_plot.set_xticks(range(len(axis_names)))
    ax_plot.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax_plot.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
    ax_plot.set_ylim(0.0, 1.0)
    if layer == test_layers[0]:
        ax_plot.set_ylabel("Avg cosine sim", fontsize=11)
        ax_plot.legend(fontsize=7, loc='upper right')

fig.suptitle("Manifold Structure Across Layers", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/activation_manifold_layers.png", dpi=150, bbox_inches='tight')
print("Saved: figures/activation_manifold_layers.png")

# Save
save_results = {
    "model": MODEL_ID,
    "test_layers": test_layers,
    "radar_layer": radar_layer,
    "axis_similarity": {
        entity: {ax: round(v, 4) for ax, v in sims.items()}
        for entity, sims in axis_similarity.items()
    },
    "axis_differential": {
        entity: {ax: round(v, 4) for ax, v in diffs.items()}
        for entity, diffs in axis_differential.items()
    },
    "per_concept_similarity": {
        str(layer): {
            entity: {c: round(s, 4) for c, s in sims.items()}
            for entity, sims in layer_data.items()
        }
        for layer, layer_data in results.items()
    },
}
with open("data/activation_manifold_results.json", "w") as f:
    json.dump(save_results, f, indent=2)
print("\nSaved: data/activation_manifold_results.json")

print("\n" + "=" * 60)
print("ACTIVATION MANIFOLD EXPERIMENT COMPLETE")
print("=" * 60)
