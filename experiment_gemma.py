"""
Named Entity Activation Geometry — Gemma-2-2B
A model in the few-shot regime, where named entities should be richly represented.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
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
    output_hidden_states=True,
).to(DEVICE)
model.eval()

num_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Model loaded: {num_layers} layers, {hidden_dim}-dim hidden state")

# ============================================================
# Setup
# ============================================================

prompts = {
    "Warren Buffett":   "Think and invest like Warren Buffett",
    "George Soros":     "Think and invest like George Soros",
    "a random person":  "Think and invest like a random person",
    "Albert Einstein":  "Think like Albert Einstein",
    "Elon Musk":        "Think like Elon Musk",
    "a scientist":      "Think like a scientist",
}

entities = list(prompts.keys())
short_names = ["Buffett", "Soros", "random person", "Einstein", "Musk", "scientist"]

def find_entity_end_pos(text, entity_text):
    """Find the position of the final token of the entity in the tokenized prompt."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    full_tokens = inputs["input_ids"][0].tolist()

    # Try matching entity tokens with and without space prefix
    for prefix in ["", " ", "▁"]:
        entity_tokens = tokenizer.encode(prefix + entity_text, add_special_tokens=False)
        for i in range(len(full_tokens) - len(entity_tokens) + 1):
            if full_tokens[i:i+len(entity_tokens)] == entity_tokens:
                pos = i + len(entity_tokens) - 1
                tok_str = tokenizer.decode(full_tokens[pos])
                print(f"    '{entity_text}' -> final token pos {pos}: '{tok_str}'")
                return inputs, pos

    # Fallback: decode each token and find by string matching
    decoded = [tokenizer.decode([t]) for t in full_tokens]
    full_text_so_far = ""
    entity_end_char = text.rfind(entity_text) + len(entity_text)
    for i, d in enumerate(decoded):
        full_text_so_far += d
        if len(full_text_so_far.rstrip()) >= entity_end_char:
            print(f"    '{entity_text}' -> final token pos {i} (string match): '{d}'")
            return inputs, i

    print(f"    WARNING: '{entity_text}' not found, using last token")
    return inputs, len(full_tokens) - 1


def get_all_layer_activations(text, entity_text):
    """Extract residual stream activation at entity's final token, all layers."""
    inputs, pos = find_entity_end_pos(text, entity_text)
    with torch.no_grad():
        outputs = model(**inputs)
    # hidden_states: tuple of (num_layers+1) tensors [batch, seq, hidden]
    return {i: outputs.hidden_states[i][0, pos, :].cpu().float().numpy()
            for i in range(len(outputs.hidden_states))}


# ============================================================
# EXPERIMENT 1: Activation Geometry
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Named Entity Activation Geometry")
print("="*60)

print("\nExtracting activations...")
all_acts = {}
for entity, prompt in prompts.items():
    print(f"  {entity}:")
    all_acts[entity] = get_all_layer_activations(prompt, entity)

results = {
    "model": MODEL_ID,
    "num_layers": num_layers,
    "hidden_dim": hidden_dim,
}

# Find the most informative layer (max variance in pairwise cosine sims)
print("\n--- Finding most informative layer ---")
best_layer = 0
best_variance = 0
for layer in range(num_layers + 1):
    act_matrix = np.stack([all_acts[e][layer] for e in entities])
    cs = cosine_similarity(act_matrix)
    # Get upper triangle values
    triu_idx = np.triu_indices(len(entities), k=1)
    pairwise = cs[triu_idx]
    var = np.var(pairwise)
    if var > best_variance:
        best_variance = var
        best_layer = layer

print(f"  Most informative layer: {best_layer} (variance in pairwise sims: {best_variance:.6f})")
results["most_informative_layer"] = best_layer

# Print cosine sim matrices at key layers
mid_layer = num_layers // 2
key_layers = sorted(set([1, mid_layer, best_layer, num_layers]))
print(f"\n--- Cosine similarity at layers {key_layers} ---")

for layer in key_layers:
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
# Heatmaps: best layer and final layer side by side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

display_layers = [(mid_layer, f"Layer {mid_layer} (Middle)"), (num_layers, f"Layer {num_layers} (Final)")]
for ax, (layer, title) in zip(axes, display_layers):
    act_matrix = np.stack([all_acts[e][layer] for e in entities])
    cos_sim = cosine_similarity(act_matrix)
    triu = cos_sim[np.triu_indices(len(entities), k=1)]
    vmin = max(0.0, float(np.min(triu)) - 0.05)
    vmax = min(1.0, float(np.max(cos_sim)) + 0.01)

    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_names, fontsize=11)
    for i in range(len(entities)):
        for j in range(len(entities)):
            thresh = (vmin + vmax) / 2 + 0.1
            color = 'white' if cos_sim[i,j] > thresh else 'black'
            ax.text(j, i, f"{cos_sim[i,j]:.3f}", ha='center', va='center', fontsize=10, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')

fig.suptitle(f"Residual Stream Cosine Similarity — {MODEL_ID}", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_similarity_heatmap.png", dpi=150)
print("\nSaved: entity_similarity_heatmap.png")

# ============================================================
# PCA at best layer and final layer
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a', '#666666']
markers = ['o', 'o', 's', '^', '^', 's']

for ax, (layer, title) in zip(axes, display_layers):
    act_matrix = np.stack([all_acts[e][layer] for e in entities])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(act_matrix)

    for i, (name, coord) in enumerate(zip(short_names, coords)):
        ax.scatter(coord[0], coord[1], c=colors[i], marker=markers[i], s=200, zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(name, (coord[0], coord[1]), fontsize=11, fontweight='bold',
                    xytext=(12, 12), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if layer == best_layer:
        results[f"pca_variance_layer_{layer}"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]

fig.suptitle(f"PCA of Named Entity Activations — {MODEL_ID}", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_pca.png", dpi=150)
print("Saved: entity_pca.png")

# ============================================================
# Layer trajectory plot
# ============================================================
pairs = [
    ("Warren Buffett", "George Soros"),
    ("Warren Buffett", "Albert Einstein"),
    ("Warren Buffett", "a random person"),
    ("Albert Einstein", "Elon Musk"),
    ("a random person", "a scientist"),
]
pair_trajectories = {}
for e1, e2 in pairs:
    key = f"{e1} vs {e2}"
    pair_trajectories[key] = []
    for layer in range(num_layers + 1):
        sim = float(cosine_similarity(
            [all_acts[e1][layer]], [all_acts[e2][layer]]
        )[0, 0])
        pair_trajectories[key].append(round(sim, 4))

fig, ax = plt.subplots(figsize=(12, 6))
styles = ['-o', '-s', '-^', '-D', '-v']
for (key, vals), style in zip(pair_trajectories.items(), styles):
    short_key = (key.replace("Warren Buffett", "Buffett")
                    .replace("Albert Einstein", "Einstein")
                    .replace("a random person", "random")
                    .replace("a scientist", "scientist"))
    ax.plot(range(num_layers + 1), vals, style, label=short_key, markersize=4)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Cosine Similarity", fontsize=12)
ax.set_title(f"Entity Similarity Across Layers — {MODEL_ID}", fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, num_layers + 1, max(1, num_layers // 12)))
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_layer_trajectory.png", dpi=150)
print("Saved: entity_layer_trajectory.png")
results["layer_trajectories"] = pair_trajectories

# ============================================================
# EXPERIMENT 2: Logit Lens at best layer
# ============================================================
print("\n" + "="*60)
print(f"EXPERIMENT 2: Logit Lens at Layer {best_layer}")
print("="*60)

# Get the final RMSNorm and lm_head
final_norm = model.model.norm
lm_head = model.lm_head

for entity_name in ["Warren Buffett", "George Soros", "Albert Einstein", "a random person"]:
    act = all_acts[entity_name][best_layer]
    act_tensor = torch.tensor(act, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        normed = final_norm(act_tensor.unsqueeze(0)).squeeze(0)
        logits = lm_head(normed.unsqueeze(0)).squeeze(0)
    logits_np = logits.cpu().float().numpy()
    top_indices = np.argsort(logits_np)[-25:][::-1]

    print(f"\n  {entity_name} (layer {best_layer} logit lens, top-25):")
    tokens_and_logits = []
    for idx in top_indices:
        token = tokenizer.decode([idx]).strip()
        tokens_and_logits.append((token, float(logits_np[idx])))
        print(f"    {token:>25s}  {logits_np[idx]:.2f}")
    results[f"logit_lens_layer{best_layer}_{entity_name}"] = [(t, round(s,2)) for t,s in tokens_and_logits]

# Also do logit lens at a few other layers for comparison
for check_layer in [mid_layer, num_layers]:
    if check_layer == best_layer:
        continue
    print(f"\n--- Logit Lens at Layer {check_layer} (Buffett only) ---")
    act = all_acts["Warren Buffett"][check_layer]
    act_tensor = torch.tensor(act, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        normed = final_norm(act_tensor.unsqueeze(0)).squeeze(0)
        logits = lm_head(normed.unsqueeze(0)).squeeze(0)
    logits_np = logits.cpu().float().numpy()
    top_indices = np.argsort(logits_np)[-15:][::-1]
    print(f"  Warren Buffett (layer {check_layer}):")
    for idx in top_indices:
        token = tokenizer.decode([idx]).strip()
        print(f"    {token:>25s}  {logits_np[idx]:.2f}")

# ============================================================
# EXPERIMENT 3: Steering Vector
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Steering Vector")
print("="*60)

# Get baseline activation
baseline_prompt = "Think and invest like a random investor"
print(f"\n  Extracting baseline: '{baseline_prompt}'")
baseline_inputs, baseline_pos = find_entity_end_pos(baseline_prompt, "a random investor")
with torch.no_grad():
    baseline_outputs = model(**baseline_inputs)
baseline_act = baseline_outputs.hidden_states[num_layers][0, baseline_pos, :].cpu().float().numpy()
buffett_act = all_acts["Warren Buffett"][num_layers]

steering_vector = buffett_act - baseline_act
steering_norm = np.linalg.norm(steering_vector)
print(f"  Steering vector norm: {steering_norm:.4f}")

generation_prompt = "The best approach to investing is"
print(f"  Generation prompt: '{generation_prompt}'")

def generate_with_steering(prompt, steering_vec, alpha=0.0, max_new_tokens=60,
                           steer_layer=None):
    if steer_layer is None:
        steer_layer = num_layers - 1  # second to last

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    if alpha == 0.0:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    steer_tensor = torch.tensor(steering_vec, dtype=torch.float32).to(DEVICE)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0] + alpha * steer_tensor
            return (hidden,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output + alpha * steer_tensor
        else:
            # BaseModelOutput-like
            output[0] = output[0] + alpha * steer_tensor
            return output

    layer_module = model.model.layers[steer_layer]
    hook = layer_module.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    finally:
        hook.remove()

    return result

steering_results = {}

print("\n--- Without steering (baseline) ---")
baseline_gens = []
for i in range(3):
    torch.manual_seed(42 + i)
    text = generate_with_steering(generation_prompt, steering_vector, alpha=0.0)
    print(f"  [{i+1}] {text}")
    baseline_gens.append(text)

# Try a few alpha values
for alpha in [3.0, 8.0]:
    print(f"\n--- With Buffett steering (alpha={alpha}) ---")
    gens = []
    for i in range(3):
        torch.manual_seed(42 + i)
        text = generate_with_steering(generation_prompt, steering_vector, alpha=alpha)
        print(f"  [{i+1}] {text}")
        gens.append(text)
    steering_results[f"alpha_{alpha}"] = gens

steering_results["baseline"] = baseline_gens
results["steering"] = steering_results

# Save
with open("/Users/x/src/lang-tokens/raw_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: raw_results.json")

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
