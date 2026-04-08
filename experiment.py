"""
Named Entity Activation Geometry Experiments
Using GPT-2 (124M) to investigate how named entities create
geometrically distinct regions in activation space.
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

# Load model
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(DEVICE)
lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
lm_model.eval()
print("Model loaded.")

# ============================================================
# EXPERIMENT 1: Named Entity Activation Geometry
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Named Entity Activation Geometry")
print("="*60)

prompts = {
    "Warren Buffett":   "Think and invest like Warren Buffett",
    "George Soros":     "Think and invest like George Soros",
    "a random person":  "Think and invest like a random person",
    "Albert Einstein":  "Think like Albert Einstein",
    "Elon Musk":        "Think like Elon Musk",
    "a scientist":      "Think like a scientist",
}

def get_activation(text, entity_text, layer=-1):
    """Extract residual stream activation at the final token of the entity."""
    # Tokenize the full prompt
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    # Find entity token positions
    # Tokenize entity alone to find its tokens
    entity_tokens = tokenizer.encode(entity_text, add_special_tokens=False)
    full_tokens = inputs["input_ids"][0].tolist()

    # Find where entity tokens appear in full sequence
    entity_end_pos = None
    for i in range(len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[i:i+len(entity_tokens)] == entity_tokens:
            entity_end_pos = i + len(entity_tokens) - 1

    # Fallback: try with space prefix (GPT-2 tokenizer quirk)
    if entity_end_pos is None:
        entity_tokens = tokenizer.encode(" " + entity_text, add_special_tokens=False)
        for i in range(len(full_tokens) - len(entity_tokens) + 1):
            if full_tokens[i:i+len(entity_tokens)] == entity_tokens:
                entity_end_pos = i + len(entity_tokens) - 1

    if entity_end_pos is None:
        print(f"  WARNING: Could not locate '{entity_text}' tokens precisely, using last token")
        entity_end_pos = len(full_tokens) - 1

    decoded_entity_end = tokenizer.decode(full_tokens[entity_end_pos])
    print(f"  Entity '{entity_text}' -> final token at pos {entity_end_pos}: '{decoded_entity_end}'")

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states is tuple of (n_layers+1) tensors, each [batch, seq, hidden]
    hidden_states = outputs.hidden_states

    # Use specified layer (default: last layer)
    activation = hidden_states[layer][0, entity_end_pos, :].cpu().numpy()
    return activation

# Collect activations for multiple layers
print("\nExtracting activations (final layer)...")
activations = {}
for entity, prompt in prompts.items():
    print(f"  Processing: {entity}")
    activations[entity] = get_activation(prompt, entity, layer=-1)

entities = list(activations.keys())
act_matrix = np.stack([activations[e] for e in entities])

# Cosine similarity matrix
cos_sim = cosine_similarity(act_matrix)
print("\nCosine Similarity Matrix:")
print(f"{'':>20s}", end="")
for e in entities:
    print(f"{e:>18s}", end="")
print()
for i, e1 in enumerate(entities):
    print(f"{e1:>20s}", end="")
    for j, e2 in enumerate(entities):
        print(f"{cos_sim[i,j]:18.4f}", end="")
    print()

# Save key similarities
results = {
    "cosine_similarities": {},
    "model": "GPT-2 (124M, 12 layers, 768 hidden dim)",
    "layer": "final (layer 12)",
}
for i, e1 in enumerate(entities):
    for j, e2 in enumerate(entities):
        if i < j:
            key = f"{e1} vs {e2}"
            results["cosine_similarities"][key] = round(float(cos_sim[i,j]), 4)

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
short_names = ["Buffett", "Soros", "random person", "Einstein", "Musk", "scientist"]
im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(short_names)))
ax.set_yticks(range(len(short_names)))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(short_names, fontsize=12)
for i in range(len(entities)):
    for j in range(len(entities)):
        ax.text(j, i, f"{cos_sim[i,j]:.3f}", ha='center', va='center',
                fontsize=10, color='black' if cos_sim[i,j] < 0.85 else 'white')
plt.colorbar(im, label='Cosine Similarity')
ax.set_title("Residual Stream Cosine Similarity\n(GPT-2, Final Layer, Entity Final Token)", fontsize=14)
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_similarity_heatmap.png", dpi=150)
print("\nSaved: entity_similarity_heatmap.png")

# PCA plot
pca = PCA(n_components=2)
coords = pca.fit_transform(act_matrix)
results["pca_variance_explained"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a', '#999999']
markers = ['o', 'o', 's', '^', '^', 's']
for i, (name, coord) in enumerate(zip(short_names, coords)):
    ax.scatter(coord[0], coord[1], c=colors[i], marker=markers[i], s=200, zorder=5,
               edgecolors='black', linewidths=1)
    ax.annotate(name, (coord[0], coord[1]), fontsize=12, fontweight='bold',
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Draw domain groupings
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
ax.set_title("PCA of Named Entity Activations\n(GPT-2, Final Layer, Residual Stream)", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/entity_pca.png", dpi=150)
print("Saved: entity_pca.png")

# Also try middle layers to see if structure differs
print("\n--- Layer-by-layer analysis (cosine sim: Buffett vs Soros, Buffett vs Einstein) ---")
layer_sims = []
for layer_idx in [1, 3, 6, 9, 12]:  # 12 = last
    acts = {}
    for entity, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # Find last token of entity
        entity_tokens = tokenizer.encode(" " + entity, add_special_tokens=False)
        full_tokens = inputs["input_ids"][0].tolist()
        entity_end_pos = len(full_tokens) - 1
        for i in range(len(full_tokens) - len(entity_tokens) + 1):
            if full_tokens[i:i+len(entity_tokens)] == entity_tokens:
                entity_end_pos = i + len(entity_tokens) - 1
        acts[entity] = outputs.hidden_states[layer_idx][0, entity_end_pos, :].cpu().numpy()

    sim_bs = float(cosine_similarity([acts["Warren Buffett"]], [acts["George Soros"]])[0,0])
    sim_be = float(cosine_similarity([acts["Warren Buffett"]], [acts["Albert Einstein"]])[0,0])
    sim_br = float(cosine_similarity([acts["Warren Buffett"]], [acts["a random person"]])[0,0])
    print(f"  Layer {layer_idx:2d}: Buffett-Soros={sim_bs:.4f}  Buffett-Einstein={sim_be:.4f}  Buffett-random={sim_br:.4f}")
    layer_sims.append({"layer": layer_idx, "buffett_soros": round(sim_bs,4),
                        "buffett_einstein": round(sim_be,4), "buffett_random": round(sim_br,4)})

results["layer_analysis"] = layer_sims

# ============================================================
# EXPERIMENT 2: Co-activated features for "Buffett"
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Logit Lens / Nearest Embedding Neighbors")
print("="*60)

# Get the embedding matrix
embedding_matrix = lm_model.transformer.wte.weight.detach().cpu().numpy()  # [vocab_size, hidden_dim]

def top_k_embedding_neighbors(activation_vec, k=20):
    """Find the k tokens whose embeddings are closest to the activation vector."""
    sims = cosine_similarity([activation_vec], embedding_matrix)[0]
    top_indices = np.argsort(sims)[-k:][::-1]
    results = []
    for idx in top_indices:
        token = tokenizer.decode([idx])
        results.append((token.strip(), float(sims[idx])))
    return results

def logit_lens(activation_vec, k=20):
    """Project activation back through the unembedding to get logits (logit lens)."""
    act_tensor = torch.tensor(activation_vec, dtype=torch.float32).to(DEVICE)
    # Apply final layer norm
    ln_f = lm_model.transformer.ln_f
    normed = ln_f(act_tensor.unsqueeze(0)).squeeze(0)
    # Project through unembedding (which is tied to embedding in GPT-2)
    logits = (normed @ lm_model.lm_head.weight.T).detach().cpu().numpy()
    top_indices = np.argsort(logits)[-k:][::-1]
    results = []
    for idx in top_indices:
        token = tokenizer.decode([idx])
        results.append((token.strip(), float(logits[idx])))
    return results

print("\n--- Logit Lens: Top-20 tokens projected from entity activations ---")
for entity_name in ["Warren Buffett", "George Soros", "Albert Einstein"]:
    act = activations[entity_name]
    top_tokens = logit_lens(act, k=20)
    print(f"\n  {entity_name}:")
    for tok, score in top_tokens:
        print(f"    {tok:>20s}  {score:.2f}")
    results[f"logit_lens_{entity_name}"] = [(t, round(s,2)) for t,s in top_tokens]

print("\n--- Embedding Space Neighbors (cosine sim to activation) ---")
for entity_name in ["Warren Buffett", "George Soros", "Albert Einstein"]:
    act = activations[entity_name]
    neighbors = top_k_embedding_neighbors(act, k=20)
    print(f"\n  {entity_name}:")
    for tok, sim in neighbors:
        print(f"    {tok:>20s}  {sim:.4f}")
    results[f"embedding_neighbors_{entity_name}"] = [(t, round(s,4)) for t,s in neighbors]

# ============================================================
# EXPERIMENT 3: Steering Vector
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Steering Vector")
print("="*60)

# Get activation for "a random investor" as baseline
baseline_prompt = "Think and invest like a random investor"
baseline_act = get_activation(baseline_prompt, "a random investor", layer=-1)
buffett_act = activations["Warren Buffett"]

# Compute steering direction
steering_vector = buffett_act - baseline_act
steering_norm = np.linalg.norm(steering_vector)
print(f"Steering vector norm: {steering_norm:.4f}")

# Now generate with and without steering
generation_prompt = "The best approach to investing is"
print(f"\nGeneration prompt: '{generation_prompt}'")

def generate_with_steering(prompt, steering_vec, alpha=0.0, max_new_tokens=60, layer_to_steer=11):
    """Generate text with optional steering vector added at a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    if alpha == 0.0:
        # Normal generation
        with torch.no_grad():
            output_ids = lm_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # For steering, we use a hook
    steer_tensor = torch.tensor(steering_vec, dtype=torch.float32).to(DEVICE)

    hooks = []
    def make_hook(vec, scale):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                return output + scale * vec
            elif isinstance(output, tuple):
                hidden = output[0] + scale * vec
                return (hidden,) + output[1:]
            else:
                output[0] = output[0] + scale * vec
                return output
        return hook_fn

    # Register hook on the target layer
    layer_module = lm_model.transformer.h[layer_to_steer]
    hook = layer_module.register_forward_hook(make_hook(steer_tensor, alpha))
    hooks.append(hook)

    try:
        with torch.no_grad():
            output_ids = lm_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()

    return result

# Set seed for reproducibility within each run
steering_results = {}

print("\n--- Without steering (baseline) ---")
baseline_gens = []
for i in range(3):
    torch.manual_seed(42 + i)
    text = generate_with_steering(generation_prompt, steering_vector, alpha=0.0)
    print(f"  [{i+1}] {text}")
    baseline_gens.append(text)

print("\n--- With Buffett steering (alpha=5.0) ---")
steered_gens_5 = []
for i in range(3):
    torch.manual_seed(42 + i)
    text = generate_with_steering(generation_prompt, steering_vector, alpha=5.0)
    print(f"  [{i+1}] {text}")
    steered_gens_5.append(text)

print("\n--- With Buffett steering (alpha=15.0) ---")
steered_gens_15 = []
for i in range(3):
    torch.manual_seed(42 + i)
    text = generate_with_steering(generation_prompt, steering_vector, alpha=15.0)
    print(f"  [{i+1}] {text}")
    steered_gens_15.append(text)

steering_results = {
    "baseline": baseline_gens,
    "alpha_5": steered_gens_5,
    "alpha_15": steered_gens_15,
}
results["steering"] = steering_results

# Save raw results as JSON
with open("/Users/x/src/lang-tokens/raw_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: raw_results.json")

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
