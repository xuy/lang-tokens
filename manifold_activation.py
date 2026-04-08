"""
Experiment: Token Activation Manifold
Does "Warren Buffett" activate an entire distribution over investment concepts,
not just a single fact? Is the distribution shaped and recognizable?

Method:
- Define ~40 investment/finance concepts spanning multiple philosophies
- For each persona (Buffett, Soros, Einstein, random person), measure how much
  the entity token shifts the probability of each concept as the next word
- Compare the resulting distributions: named entities should show structured,
  distinct manifolds; generic descriptors should be flat
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_ID = "google/gemma-2-2b"
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32,
).to(DEVICE)
model.eval()
print("Model loaded.")

# ============================================================
# Investment concept vocabulary — organized by philosophy
# ============================================================

concept_groups = {
    "Value / Fundamental": [
        "value", "intrinsic", "undervalued", "moat", "margin",
        "compounding", "patience", "dividends", "earnings", "book",
        "fundamental", "conservative", "quality", "durable", "ownership",
    ],
    "Macro / Global": [
        "currency", "macro", "sovereign", "geopolitical", "reflexivity",
        "hedge", "global", "bonds", "inflation", "rates",
        "commodities", "emerging", "crisis", "speculation",
    ],
    "Quantitative / Technical": [
        "momentum", "algorithmic", "volatility", "correlation", "arbitrage",
        "quantitative", "derivatives", "options", "leverage", "signals",
        "frequency", "statistical", "systematic",
    ],
    "General / Neutral": [
        "diversification", "portfolio", "risk", "returns", "stocks",
        "market", "capital", "growth", "index", "wealth",
    ],
}

# Flatten and tag
all_concepts = []
concept_to_group = {}
for group, concepts in concept_groups.items():
    for c in concepts:
        all_concepts.append(c)
        concept_to_group[c] = group

print(f"Total concepts: {len(all_concepts)}")

# ============================================================
# Personas and prompt templates
# ============================================================

# Template: the entity name followed by a context that invites concept-related completion
# We use multiple prompt templates to average out template effects

prompt_templates = [
    "{entity} believes the key to investing is",
    "{entity} would advise focusing on",
    "{entity} thinks the most important factor in markets is",
    "When it comes to investing, {entity} emphasizes",
    "The investment philosophy of {entity} centers on",
]

personas = {
    "Warren Buffett": "Warren Buffett",
    "George Soros": "George Soros",
    "Jim Simons": "Jim Simons",
    "Elon Musk": "Elon Musk",
    "Albert Einstein": "Albert Einstein",
    "a random person": "a random person",
    "a financial advisor": "a financial advisor",
}


def get_concept_probabilities(prompt):
    """Get probability of each concept token as the next token after the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last position
    probs = torch.softmax(logits, dim=-1).cpu()

    concept_probs = {}
    for concept in all_concepts:
        # Check multiple tokenizations (with/without space prefix)
        max_prob = 0.0
        for prefix in [" ", ""]:
            token_ids = tokenizer.encode(prefix + concept, add_special_tokens=False)
            if token_ids:
                p = probs[token_ids[0]].item()
                max_prob = max(max_prob, p)
        concept_probs[concept] = max_prob

    return concept_probs


def get_concept_logits(prompt):
    """Get raw logits for concept tokens (more informative than probs for comparison)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].cpu()

    concept_logits = {}
    for concept in all_concepts:
        max_logit = -float('inf')
        for prefix in [" ", ""]:
            token_ids = tokenizer.encode(prefix + concept, add_special_tokens=False)
            if token_ids:
                l = logits[token_ids[0]].item()
                max_logit = max(max_logit, l)
        concept_logits[concept] = max_logit

    return concept_logits


# ============================================================
# Collect data: concept probabilities for each persona × template
# ============================================================

print("\nCollecting concept activations...")
# Store per-persona averaged probabilities and logits
persona_avg_probs = {}
persona_avg_logits = {}
persona_all_probs = {}  # raw per-template data

for persona_name, entity_str in personas.items():
    print(f"\n  {persona_name}:")
    all_probs_for_persona = []
    all_logits_for_persona = []

    for template in prompt_templates:
        prompt = template.format(entity=entity_str)
        print(f"    '{prompt}'")
        cprobs = get_concept_probabilities(prompt)
        clogits = get_concept_logits(prompt)
        all_probs_for_persona.append(cprobs)
        all_logits_for_persona.append(clogits)

    # Average across templates
    avg_probs = {}
    avg_logits = {}
    for concept in all_concepts:
        avg_probs[concept] = np.mean([p[concept] for p in all_probs_for_persona])
        avg_logits[concept] = np.mean([l[concept] for l in all_logits_for_persona])

    persona_avg_probs[persona_name] = avg_probs
    persona_avg_logits[persona_name] = avg_logits
    persona_all_probs[persona_name] = all_probs_for_persona

    # Print top-10 concepts by probability
    sorted_concepts = sorted(avg_probs.items(), key=lambda x: -x[1])
    print(f"    Top-10: {[(c, f'{p:.4f}') for c, p in sorted_concepts[:10]]}")


# ============================================================
# Analysis: Compute "activation manifold" shape metrics
# ============================================================

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

results = {"model": MODEL_ID, "concepts": all_concepts, "concept_groups": concept_groups}

# 1. For each persona, what's the entropy of the concept distribution?
#    Low entropy = concentrated/peaked manifold, high entropy = flat
print("\n--- Distribution entropy (lower = more concentrated) ---")
for persona in personas:
    probs_array = np.array([persona_avg_probs[persona][c] for c in all_concepts])
    # Normalize to a proper distribution
    probs_norm = probs_array / probs_array.sum()
    entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-10))
    max_entropy = np.log(len(all_concepts))
    print(f"  {persona:25s}: entropy={entropy:.3f} / {max_entropy:.3f} = {entropy/max_entropy:.3f} (normalized)")
    results[f"entropy_{persona}"] = round(entropy, 4)
    results[f"entropy_normalized_{persona}"] = round(entropy / max_entropy, 4)

# 2. For each persona, what fraction of probability mass falls in each concept group?
print("\n--- Probability mass by concept group ---")
group_mass = {}
for persona in personas:
    group_mass[persona] = {}
    total = sum(persona_avg_probs[persona][c] for c in all_concepts)
    for group, concepts in concept_groups.items():
        mass = sum(persona_avg_probs[persona][c] for c in concepts) / total if total > 0 else 0
        group_mass[persona][group] = mass

    print(f"\n  {persona}:")
    for group in concept_groups:
        bar = "#" * int(group_mass[persona][group] * 100)
        print(f"    {group:25s}: {group_mass[persona][group]:.3f}  {bar}")

results["group_mass"] = group_mass

# 3. Compute relative lift: how much does each persona elevate concepts vs baseline?
baseline = "a random person"
print(f"\n--- Relative lift vs '{baseline}' ---")
persona_lift = {}
for persona in personas:
    if persona == baseline:
        continue
    lift = {}
    for concept in all_concepts:
        p_persona = persona_avg_probs[persona][concept]
        p_baseline = persona_avg_probs[baseline][concept]
        if p_baseline > 1e-8:
            lift[concept] = p_persona / p_baseline
        else:
            lift[concept] = p_persona / 1e-8
    persona_lift[persona] = lift

    sorted_lift = sorted(lift.items(), key=lambda x: -x[1])
    print(f"\n  {persona} — most elevated vs baseline:")
    for c, l in sorted_lift[:10]:
        group = concept_to_group[c]
        print(f"    {c:20s} ({group:25s}): {l:.1f}x")
    print(f"  {persona} — most suppressed vs baseline:")
    for c, l in sorted_lift[-5:]:
        group = concept_to_group[c]
        print(f"    {c:20s} ({group:25s}): {l:.2f}x")

results["lift_vs_baseline"] = {p: {c: round(l, 2) for c, l in lift.items()}
                                for p, lift in persona_lift.items()}

# ============================================================
# VISUALIZATIONS
# ============================================================

# 1. Heatmap: personas × concepts, colored by probability
print("\n--- Generating heatmap ---")
plot_personas = ["Warren Buffett", "George Soros", "Jim Simons", "a financial advisor", "a random person"]
prob_matrix = np.array([
    [persona_avg_probs[p][c] for c in all_concepts]
    for p in plot_personas
])

# Log-scale for better visibility
log_probs = np.log10(prob_matrix + 1e-8)

fig, ax = plt.subplots(figsize=(22, 7))
im = ax.imshow(log_probs, aspect='auto', cmap='YlOrRd', interpolation='nearest')

ax.set_yticks(range(len(plot_personas)))
ax.set_yticklabels(plot_personas, fontsize=11)
ax.set_xticks(range(len(all_concepts)))
ax.set_xticklabels(all_concepts, rotation=75, ha='right', fontsize=8)

# Add group separators
cumlen = 0
for group, concepts in concept_groups.items():
    mid = cumlen + len(concepts) / 2
    ax.axvline(x=cumlen - 0.5, color='white', linewidth=2)
    ax.text(mid, -1.5, group.split("/")[0].strip(), ha='center', fontsize=9,
            fontweight='bold', color='#333')
    cumlen += len(concepts)

plt.colorbar(im, ax=ax, label='log₁₀(probability)', shrink=0.8)
ax.set_title(f"Concept Activation Manifold by Entity — {MODEL_ID}\n"
             f"(averaged over {len(prompt_templates)} prompt templates)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved: manifold_heatmap.png")

# 2. Radar/spider chart for top personas
print("--- Generating radar chart ---")
# Use concept groups as axes (aggregated probability mass)
radar_personas = ["Warren Buffett", "George Soros", "Jim Simons", "a random person"]
groups_list = list(concept_groups.keys())
n_groups = len(groups_list)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
angles += angles[:1]  # close the loop

colors_radar = ['#e41a1c', '#ff7f00', '#377eb8', '#999999']
for persona, color in zip(radar_personas, colors_radar):
    values = [group_mass[persona][g] for g in groups_list]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=persona, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([g.split("/")[0].strip() for g in groups_list], fontsize=11)
ax.set_title("Concept Group Activation Profile by Entity", fontsize=13, fontweight='bold',
             pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_radar.png", dpi=150, bbox_inches='tight')
print("Saved: manifold_radar.png")

# 3. Lift chart: Buffett vs Soros differential
print("--- Generating differential lift chart ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, persona, title in [
    (axes[0], "Warren Buffett", "Buffett Lift vs Random Person"),
    (axes[1], "George Soros", "Soros Lift vs Random Person"),
]:
    lift = persona_lift[persona]
    sorted_concepts_lift = sorted(all_concepts, key=lambda c: -lift[c])

    colors_lift = []
    group_colors = {
        "Value / Fundamental": '#e41a1c',
        "Macro / Global": '#ff7f00',
        "Quantitative / Technical": '#377eb8',
        "General / Neutral": '#999999',
    }
    for c in sorted_concepts_lift:
        colors_lift.append(group_colors[concept_to_group[c]])

    lift_values = [lift[c] for c in sorted_concepts_lift]
    y_pos = range(len(sorted_concepts_lift))

    ax.barh(y_pos, lift_values, color=colors_lift, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_concepts_lift, fontsize=8)
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Probability lift (×)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_lift.png", dpi=150, bbox_inches='tight')
print("Saved: manifold_lift.png")

# 4. Direct comparison: Buffett vs Soros manifold shape
print("--- Generating Buffett vs Soros comparison ---")
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(all_concepts))
width = 0.25

for i, (persona, color) in enumerate([
    ("Warren Buffett", '#e41a1c'),
    ("George Soros", '#ff7f00'),
    ("a random person", '#999999'),
]):
    probs = [persona_avg_probs[persona][c] for c in all_concepts]
    ax.bar(x + i * width, probs, width, label=persona, color=color, alpha=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels(all_concepts, rotation=75, ha='right', fontsize=8)
ax.set_ylabel("Avg P(concept as next token)", fontsize=11)
ax.set_title(f"Concept Activation: Buffett vs Soros vs Random — {MODEL_ID}", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Group separators
cumlen = 0
for group, concepts in concept_groups.items():
    ax.axvline(x=cumlen - 0.5, color='gray', linestyle='--', alpha=0.3)
    cumlen += len(concepts)

plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/manifold_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: manifold_comparison.png")

# Save results
results["persona_avg_probs"] = {p: {c: round(v, 6) for c, v in probs.items()}
                                 for p, probs in persona_avg_probs.items()}
with open("/Users/x/src/lang-tokens/manifold_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: manifold_results.json")

print("\n" + "="*60)
print("MANIFOLD ACTIVATION EXPERIMENT COMPLETE")
print("="*60)
