"""
Enhanced analysis: redo key plots at middle layers where entity geometry
is more differentiated, plus improved logit lens at middle layers.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(DEVICE)
lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
lm_model.eval()

prompts = {
    "Warren Buffett":   "Think and invest like Warren Buffett",
    "George Soros":     "Think and invest like George Soros",
    "a random person":  "Think and invest like a random person",
    "Albert Einstein":  "Think like Albert Einstein",
    "Elon Musk":        "Think like Elon Musk",
    "a scientist":      "Think like a scientist",
}

def find_entity_end_pos(text, entity_text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    full_tokens = inputs["input_ids"][0].tolist()
    for prefix in ["", " "]:
        entity_tokens = tokenizer.encode(prefix + entity_text, add_special_tokens=False)
        for i in range(len(full_tokens) - len(entity_tokens) + 1):
            if full_tokens[i:i+len(entity_tokens)] == entity_tokens:
                return inputs, i + len(entity_tokens) - 1
    return inputs, len(full_tokens) - 1

def get_all_layer_activations(text, entity_text):
    inputs, pos = find_entity_end_pos(text, entity_text)
    with torch.no_grad():
        outputs = model(**inputs)
    # Return dict: layer_idx -> activation vector
    return {i: outputs.hidden_states[i][0, pos, :].cpu().numpy()
            for i in range(len(outputs.hidden_states))}

# Collect all-layer activations
print("Extracting activations across all layers...")
all_acts = {}
for entity, prompt in prompts.items():
    all_acts[entity] = get_all_layer_activations(prompt, entity)

entities = list(prompts.keys())
short_names = ["Buffett", "Soros", "random person", "Einstein", "Musk", "scientist"]

# ============================================================
# Multi-layer cosine similarity analysis
# ============================================================
print("\n--- Full cosine similarity matrices at layers 1, 6, 9, 12 ---")
results = {}
for layer in [1, 6, 9, 12]:
    act_matrix = np.stack([all_acts[e][layer] for e in entities])
    cos_sim = cosine_similarity(act_matrix)
    results[f"cosine_sim_layer_{layer}"] = {
        f"{entities[i]} vs {entities[j]}": round(float(cos_sim[i,j]), 4)
        for i in range(len(entities)) for j in range(i+1, len(entities))
    }
    print(f"\nLayer {layer}:")
    print(f"{'':>20s}", end="")
    for e in short_names:
        print(f"{e:>15s}", end="")
    print()
    for i, e1 in enumerate(short_names):
        print(f"{e1:>20s}", end="")
        for j in range(len(entities)):
            print(f"{cos_sim[i,j]:15.4f}", end="")
        print()

# ============================================================
# Heatmaps: layer 6 (mid) and layer 12 (final) side by side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, layer, title in [(axes[0], 6, "Layer 6 (Middle)"), (axes[1], 12, "Layer 12 (Final)")]:
    act_matrix = np.stack([all_acts[e][layer] for e in entities])
    cos_sim = cosine_similarity(act_matrix)

    vmin = 0.5 if layer == 6 else 0.95
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_names, fontsize=11)
    for i in range(len(entities)):
        for j in range(len(entities)):
            color = 'white' if cos_sim[i,j] > (0.85 if layer == 6 else 0.99) else 'black'
            ax.text(j, i, f"{cos_sim[i,j]:.3f}", ha='center', va='center', fontsize=10, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{title}", fontsize=13, fontweight='bold')

fig.suptitle("Residual Stream Cosine Similarity — GPT-2 (124M)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_similarity_heatmap.png", dpi=150)
print("\nSaved: entity_similarity_heatmap.png")

# ============================================================
# PCA at layer 6 (where structure is most visible)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, layer, title in [(axes[0], 6, "Layer 6 (Middle)"), (axes[1], 12, "Layer 12 (Final)")]:
    act_matrix = np.stack([all_acts[e][layer] for e in entities])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(act_matrix)

    # Color by domain: investors=red/orange, generic=gray, thinkers=blue/green
    colors = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a', '#666666']
    markers = ['o', 'o', 's', '^', '^', 's']

    for i, (name, coord) in enumerate(zip(short_names, coords)):
        ax.scatter(coord[0], coord[1], c=colors[i], marker=markers[i], s=200, zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(name, (coord[0], coord[1]), fontsize=11, fontweight='bold',
                    xytext=(12, 12), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(f"{title}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if layer == 6:
        results[f"pca_variance_layer_{layer}"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]

fig.suptitle("PCA of Named Entity Activations — GPT-2 (124M)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_pca.png", dpi=150)
print("Saved: entity_pca.png")

# ============================================================
# Logit Lens at layer 6 (middle) — more interesting predictions
# ============================================================
print("\n--- Logit Lens at LAYER 6 ---")
ln_f = lm_model.transformer.ln_f

for entity_name in ["Warren Buffett", "George Soros", "Albert Einstein", "a random person"]:
    act = all_acts[entity_name][6]
    act_tensor = torch.tensor(act, dtype=torch.float32).to(DEVICE)
    normed = ln_f(act_tensor.unsqueeze(0)).squeeze(0)
    logits = (normed @ lm_model.lm_head.weight.T).detach().cpu().numpy()
    top_indices = np.argsort(logits)[-20:][::-1]
    print(f"\n  {entity_name} (layer 6 logit lens):")
    tokens_and_logits = []
    for idx in top_indices:
        token = tokenizer.decode([idx]).strip()
        tokens_and_logits.append((token, float(logits[idx])))
        print(f"    {token:>20s}  {logits[idx]:.2f}")
    results[f"logit_lens_layer6_{entity_name}"] = [(t, round(s,2)) for t,s in tokens_and_logits]

# ============================================================
# Layer-by-layer similarity trajectory (for the report)
# ============================================================
print("\n--- Layer-by-layer pairwise similarities ---")
pair_trajectories = {}
pairs = [
    ("Warren Buffett", "George Soros"),
    ("Warren Buffett", "Albert Einstein"),
    ("Warren Buffett", "a random person"),
    ("Albert Einstein", "Elon Musk"),
    ("a random person", "a scientist"),
]
for e1, e2 in pairs:
    key = f"{e1} vs {e2}"
    pair_trajectories[key] = []
    for layer in range(13):  # 0..12
        sim = float(cosine_similarity(
            [all_acts[e1][layer]], [all_acts[e2][layer]]
        )[0, 0])
        pair_trajectories[key].append(round(sim, 4))

# Plot layer trajectories
fig, ax = plt.subplots(figsize=(10, 6))
styles = ['-o', '-s', '-^', '-D', '-v']
for (key, vals), style in zip(pair_trajectories.items(), styles):
    short_key = key.replace("Warren Buffett", "Buffett").replace("Albert Einstein", "Einstein").replace("a random person", "random").replace("a scientist", "scientist")
    ax.plot(range(13), vals, style, label=short_key, markersize=5)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Cosine Similarity", fontsize=12)
ax.set_title("Entity Similarity Across Layers — GPT-2 (124M)", fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(13))
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_layer_trajectory.png", dpi=150)
print("Saved: entity_layer_trajectory.png")

results["layer_trajectories"] = pair_trajectories

# Save all results
with open("/Users/x/src/lang-tokens/raw_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: raw_results.json")
print("DONE")
