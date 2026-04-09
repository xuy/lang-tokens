#!/usr/bin/env python3
"""
Discovery-based concept activation: let the model tell us what each
name activates, rather than defining concepts upfront.

For each thinker:
1. Run "The philosophy of {entity} is fundamentally about" forward
2. Read the model's own top predicted tokens (the discovered associations)
3. Run attribution, trace how much of each prediction comes from entity features
4. Compare to baseline ("a random person") to isolate entity-specific activations
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution.attribute import attribute
import json
import warnings
import os
import re
warnings.filterwarnings('ignore')
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"

DEVICE = "mps"
MODEL_ID = "google/gemma-2-2b"
TRANSCODER = "mntss/gemma-scope-transcoders"

print(f"Loading {MODEL_ID}...")
model = ReplacementModel.from_pretrained(MODEL_ID, TRANSCODER, device=DEVICE)
tokenizer = model.tokenizer
print("Loaded.")

# ============================================================
# Stop words to filter out from discovered concepts
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "as", "up", "out", "how", "what", "which", "who", "whom",
    "when", "where", "why", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "too", "very",
    "just", "about", "above", "after", "again", "also", "any", "because",
    "before", "between", "both", "during", "into", "over", "through",
    "under", "until", "while", "one", "two", "three", "first", "new",
    "way", "us", "much", "well", "here", "there", "thing", "things",
    "fact", "something", "nothing", "everything", "anything",
    "", " ", "\n", ".", ",", ":", ";", "!", "?", "-", "(", ")", "'", '"',
}


def is_content_word(token_str):
    """Filter: keep only meaningful content words."""
    clean = token_str.strip().lower()
    if clean in STOP_WORDS:
        return False
    if len(clean) < 3:
        return False
    if not re.match(r'^[a-z]+$', clean):
        return False
    # Filter common non-content words
    if clean in {"ing", "tion", "ness", "ment", "able", "ible", "ous", "ive",
                 "ful", "less", "ship", "dom", "ism", "ist", "ity", "ence",
                 "ance", "ally", "like", "make", "made", "take", "get", "got",
                 "know", "think", "say", "said", "see", "come", "look",
                 "want", "give", "use", "find", "tell", "ask", "try",
                 "need", "let", "keep", "begin", "seem", "help", "show",
                 "hear", "play", "run", "move", "live", "believe", "bring",
                 "happen", "write", "provide", "sit", "stand", "lose", "pay",
                 "meet", "include", "continue", "set", "learn", "change",
                 "lead", "understand", "watch", "follow", "stop", "create",
                 "speak", "read", "allow", "add", "spend", "grow", "open",
                 "walk", "win", "offer", "remember", "love", "consider",
                 "appear", "buy", "wait", "serve", "die", "send", "build",
                 "stay", "fall", "cut", "reach", "kill", "remain", "suggest",
                 "raise", "pass", "sell", "require", "report", "decide",
                 "pull", "develop", "many", "even", "still", "already"}:
        return False
    return True


ENTITIES = {
    "Immanuel Kant": "Immanuel Kant",
    "Friedrich Nietzsche": "Friedrich Nietzsche",
    "Karl Marx": "Karl Marx",
    "Jean-Paul Sartre": "Jean-Paul Sartre",
    "a random person": "a random person",
}

TEMPLATE = "The philosophy of {entity} is fundamentally about"

# ============================================================
# Step 1: Discover what the model predicts for each entity
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Discover model's own predictions per entity")
print("=" * 60)

discovered = {}  # entity -> list of (token_str, prob)

for entity_name, entity_str in ENTITIES.items():
    prompt = TEMPLATE.format(entity=entity_str)
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model.forward(tokens)[0, -1, :]  # last position logits

    probs = torch.softmax(logits, dim=-1).cpu()
    top_k = 200  # look at top 200 to find enough content words
    top_probs, top_ids = torch.topk(probs, top_k)

    content_words = []
    for prob, tid in zip(top_probs, top_ids):
        token_str = tokenizer.decode([tid.item()]).strip()
        if is_content_word(token_str):
            content_words.append((token_str, prob.item()))
        if len(content_words) >= 20:
            break

    discovered[entity_name] = content_words
    print(f"\n  {entity_name}:")
    for word, prob in content_words[:15]:
        print(f"    {word:20s}  P={prob:.4f}")


# ============================================================
# Step 2: Run attribution with default targets (model's own top logits)
#          and trace entity contribution
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Attribution — how much comes from the entity name?")
print("=" * 60)


def find_entity_positions(prompt, entity):
    tokens = tokenizer.encode(prompt)
    for prefix in [" ", ""]:
        entity_toks = tokenizer.encode(prefix + entity, add_special_tokens=False)
        for i in range(len(tokens) - len(entity_toks) + 1):
            if tokens[i:i + len(entity_toks)] == entity_toks:
                return list(range(i, i + len(entity_toks)))
    return list(range(1, min(4, len(tokens))))


entity_attributions = {}  # entity -> {token: attribution_from_entity}

for entity_name, entity_str in ENTITIES.items():
    prompt = TEMPLATE.format(entity=entity_str)
    entity_positions = find_entity_positions(prompt, entity_str)
    print(f"\n{'='*50}")
    print(f"{entity_name} | pos {entity_positions}")

    # Run attribution with model's own top logits (default behavior)
    graph = attribute(prompt, model, max_n_logits=30, desired_logit_prob=0.99, verbose=True)

    adj = graph.adjacency_matrix
    n_feat = graph.active_features.shape[0]
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    logit_start = n_feat + (n_layers * n_pos) + n_pos

    entity_feat_indices = [i for i in range(n_feat)
                           if graph.active_features[i][1].item() in entity_positions]

    # For each logit target, compute entity contribution
    token_attrs = {}
    for li, lt in enumerate(graph.logit_targets):
        logit_row = logit_start + li
        entity_attr = sum(adj[logit_row, fi].item() for fi in entity_feat_indices)
        token_str = lt.token_str.strip()
        if is_content_word(token_str):
            token_attrs[token_str] = entity_attr

    entity_attributions[entity_name] = token_attrs

    sorted_attrs = sorted(token_attrs.items(), key=lambda x: -x[1])
    print(f"\n  Content words from entity features:")
    for tok, attr in sorted_attrs[:10]:
        print(f"    {tok:20s}: {attr:+.4f}")
    if any(v < 0 for _, v in sorted_attrs):
        print(f"  Suppressed:")
        for tok, attr in sorted_attrs[-3:]:
            if attr < 0:
                print(f"    {tok:20s}: {attr:+.4f}")


# ============================================================
# Step 3: Merge discovered predictions + entity attributions
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Final — discovered concepts + entity contribution")
print("=" * 60)

# For each entity, combine: what the model predicts (discovery)
# with how much the entity name contributes (attribution)
final = {}

for entity_name in ENTITIES:
    if entity_name == "a random person":
        continue

    disc = {w: p for w, p in discovered[entity_name]}
    attrs = entity_attributions.get(entity_name, {})
    baseline_disc = {w: p for w, p in discovered["a random person"]}

    # Combine: words discovered for this entity but NOT for random person
    # (or much more probable for this entity)
    unique_concepts = []
    for word, prob in discovered[entity_name]:
        baseline_prob = baseline_disc.get(word, 0)
        lift = prob / max(baseline_prob, 1e-6)
        entity_attr = attrs.get(word, 0)
        if lift > 1.5 or entity_attr > 0.5:
            unique_concepts.append({
                "word": word,
                "prob": prob,
                "baseline_prob": baseline_prob,
                "lift": lift,
                "entity_attribution": entity_attr,
            })

    final[entity_name] = unique_concepts
    print(f"\n  {entity_name} — discovered + entity-attributed concepts:")
    for c in unique_concepts[:12]:
        print(f"    {c['word']:20s}  P={c['prob']:.4f}  lift={c['lift']:.1f}x  "
              f"entity_attr={c['entity_attribution']:+.2f}")


# ============================================================
# Visualization: Constellation diagram
# ============================================================

thinkers = ["Immanuel Kant", "Friedrich Nietzsche", "Karl Marx", "Jean-Paul Sartre"]
short_names = ["Kant", "Nietzsche", "Marx", "Sartre"]
thinker_colors = ['#c0392b', '#e67e22', '#2980b9', '#8e44ad']

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for ax, thinker, short, color in zip(axes, thinkers, short_names, thinker_colors):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central node
    ax.scatter([0], [0], s=800, c=color, zorder=10, edgecolors='white', linewidths=2)
    ax.text(0, 0, short, ha='center', va='center', fontsize=12,
            fontweight='bold', color='white', zorder=11)

    # Discovered concepts as orbiting nodes
    concepts = discovered[thinker][:15]
    baseline_words = {w for w, _ in discovered["a random person"]}
    attrs = entity_attributions.get(thinker, {})

    n = len(concepts)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for i, ((word, prob), angle) in enumerate(zip(concepts, angles)):
        # Distance from center = inverse of attribution (stronger = closer)
        entity_attr = attrs.get(word, 0)
        distance = max(0.4, min(1.3, 1.3 - entity_attr * 0.15))

        x = distance * np.cos(angle)
        y = distance * np.sin(angle)

        # Size by probability
        size = max(8, min(16, prob * 300))

        # Color intensity by entity attribution
        alpha = max(0.3, min(1.0, 0.3 + entity_attr * 0.15))

        # Bold if unique to this thinker (not in random's top)
        is_unique = word not in baseline_words
        weight = 'bold' if is_unique else 'normal'

        # Line from center
        line_alpha = max(0.1, min(0.8, entity_attr * 0.1))
        line_width = max(0.5, min(3.0, entity_attr * 0.5))
        ax.plot([0, x], [0, y], color=color, alpha=line_alpha, linewidth=line_width, zorder=1)

        ax.text(x, y, word, ha='center', va='center', fontsize=size,
                fontweight=weight, color=color, alpha=alpha, zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.3 * alpha, linewidth=0.5))

    ax.set_title(f"What does \"{short}\" activate?", fontsize=14, fontweight='bold', color=color)

fig.suptitle("Discovered Concept Constellations\n"
             "(model's own top predictions, bold = unique to this thinker, "
             "closer = more entity-driven)",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figures/discovered_constellations.png", dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: figures/discovered_constellations.png")

# Also save the simpler network view across all thinkers
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Layout: thinkers on the left, discovered concepts on the right
# Only show concepts unique or strongly attributed to each thinker

thinker_y = {}
for i, (thinker, short, color) in enumerate(zip(thinkers, short_names, thinker_colors)):
    y = (len(thinkers) - 1 - i) * 2.5
    thinker_y[thinker] = y
    ax.text(-0.5, y, short, ha='right', va='center', fontsize=14, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.15, edgecolor=color))

# Collect all unique concepts across thinkers
all_unique = {}  # word -> list of (thinker, prob, entity_attr)
baseline_words = {w: p for w, p in discovered["a random person"]}

for thinker in thinkers:
    attrs = entity_attributions.get(thinker, {})
    for word, prob in discovered[thinker][:15]:
        bp = baseline_words.get(word, 0)
        ea = attrs.get(word, 0)
        if prob > bp * 1.3 or ea > 0.3:
            if word not in all_unique:
                all_unique[word] = []
            all_unique[word].append((thinker, prob, ea))

# Sort concepts by which thinker they're most associated with
concept_list = sorted(all_unique.keys(),
                      key=lambda w: -max(ea for _, _, ea in all_unique[w]))

# Position concepts on the right
n_concepts = len(concept_list)
concept_y = {}
y_span = (len(thinkers) - 1) * 2.5
for i, concept in enumerate(concept_list):
    concept_y[concept] = y_span - (i / max(n_concepts - 1, 1)) * y_span

# Draw connections and concept labels
for concept in concept_list:
    cy = concept_y[concept]
    entries = all_unique[concept]

    # Color by strongest thinker
    best_thinker = max(entries, key=lambda x: x[2])
    best_idx = thinkers.index(best_thinker[0])
    concept_color = thinker_colors[best_idx]

    # Draw concept label
    ax.text(10.5, cy, concept, ha='left', va='center', fontsize=11,
            fontweight='bold', color=concept_color)

    # Draw connections
    for thinker, prob, ea in entries:
        ty = thinker_y[thinker]
        tidx = thinkers.index(thinker)
        color = thinker_colors[tidx]
        alpha = max(0.15, min(0.9, ea * 0.12))
        lw = max(0.5, min(4.0, ea * 0.6))
        ax.plot([0.2, 10.0], [ty, cy], color=color, alpha=alpha, linewidth=lw, zorder=1)

ax.set_xlim(-4, 16)
ax.set_ylim(-1, y_span + 1)
ax.set_title("Discovered Associations: What Each Name Activates\n"
             "(line thickness = attribution from entity features, "
             "color = strongest thinker)",
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/discovered_network.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: figures/discovered_network.png")

# Save data
save_data = {
    "model": MODEL_ID,
    "template": TEMPLATE,
    "discovered_predictions": {
        e: [(w, round(p, 6)) for w, p in concepts]
        for e, concepts in discovered.items()
    },
    "entity_attributions": {
        e: {k: round(v, 4) for k, v in attrs.items()}
        for e, attrs in entity_attributions.items()
    },
    "final_unique_concepts": {
        e: concepts for e, concepts in final.items()
    },
}
with open("data/discovered_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print("Saved: data/discovered_results.json")

print("\nDONE")
