"""
Experiment: Causal Tracing (ROME methodology)
Does "Warren Buffett" causally retrieve specific knowledge that "a random person" doesn't?

Method (Meng et al., NeurIPS 2022):
1. Run clean prompt, record all hidden states
2. Corrupt the subject tokens (replace with noise) → model loses the fact
3. Restore individual (layer, position) activations from clean run into corrupted run
4. Measure which restorations recover the correct prediction
5. The resulting heatmap shows WHERE the pointer gets dereferenced

We test multiple factual prompts about Buffett vs generic equivalents.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_ID = "google/gemma-2-2b"
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, output_hidden_states=True,
).to(DEVICE)
model.eval()

num_layers = model.config.num_hidden_layers
print(f"Model loaded: {num_layers} layers")

# ============================================================
# Factual prompts — things we expect the model to "know"
# ============================================================

# Each entry: (prompt, subject_text, expected_completions)
# expected_completions = tokens we check probability for
factual_prompts = [
    {
        "prompt": "Warren Buffett is the CEO of",
        "subject": "Warren Buffett",
        "target_tokens": ["Berkshire", "Berk"],  # Berkshire Hathaway
        "label": "Buffett → Berkshire",
    },
    {
        "prompt": "Warren Buffett lives in",
        "subject": "Warren Buffett",
        "target_tokens": ["Omaha", "Om"],
        "label": "Buffett → Omaha",
    },
    {
        "prompt": "Warren Buffett's investment philosophy emphasizes",
        "subject": "Warren Buffett",
        "target_tokens": ["value", "long", "patience", "fundamental"],
        "label": "Buffett → value investing",
    },
    {
        "prompt": "A random person is the CEO of",
        "subject": "A random person",
        "target_tokens": ["Berkshire", "Berk"],
        "label": "random → Berkshire (control)",
    },
    {
        "prompt": "Albert Einstein is famous for the theory of",
        "subject": "Albert Einstein",
        "target_tokens": ["relat", "general", "special"],
        "label": "Einstein → relativity",
    },
    {
        "prompt": "A random scientist is famous for the theory of",
        "subject": "A random scientist",
        "target_tokens": ["relat", "general", "special"],
        "label": "random scientist → relativity (control)",
    },
]


def find_subject_positions(prompt, subject):
    """Find token positions of the subject in the prompt."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Try with and without space prefix
    for prefix in [" ", ""]:
        subj_ids = tokenizer.encode(prefix + subject, add_special_tokens=False)
        for start in range(len(input_ids) - len(subj_ids) + 1):
            if input_ids[start:start+len(subj_ids)] == subj_ids:
                positions = list(range(start, start + len(subj_ids)))
                tokens_str = [tokenizer.decode([input_ids[p]]) for p in positions]
                print(f"  Subject '{subject}' at positions {positions}: {tokens_str}")
                return positions
    # Fallback: first few tokens after BOS
    print(f"  WARNING: subject not found precisely, using positions 1-3")
    return [1, 2, 3]


def get_target_token_prob(logits, target_tokens):
    """Get the max probability across target token variants."""
    probs = torch.softmax(logits, dim=-1)
    max_prob = 0.0
    best_token = ""
    for target in target_tokens:
        # Find all token IDs that start with the target string
        for tid in range(tokenizer.vocab_size):
            tok_str = tokenizer.decode([tid])
            if tok_str.strip().lower().startswith(target.lower()):
                p = probs[tid].item()
                if p > max_prob:
                    max_prob = p
                    best_token = tok_str
                if p > 0.01:  # Only check high-prob matches
                    break
    return max_prob, best_token


def get_target_prob_fast(logits, target_tokens):
    """Get probability of target tokens efficiently."""
    probs = torch.softmax(logits, dim=-1).cpu()
    max_prob = 0.0
    best_token = ""
    # Check top-100 tokens and see if any match
    top_probs, top_ids = torch.topk(probs, 100)
    for i in range(100):
        tok_str = tokenizer.decode([top_ids[i].item()]).strip().lower()
        for target in target_tokens:
            if tok_str.startswith(target.lower()):
                if top_probs[i].item() > max_prob:
                    max_prob = top_probs[i].item()
                    best_token = tokenizer.decode([top_ids[i].item()])

    # Also directly encode target tokens and check their probs
    for target in target_tokens:
        tids = tokenizer.encode(" " + target, add_special_tokens=False)
        if tids:
            p = probs[tids[0]].item()
            if p > max_prob:
                max_prob = p
                best_token = tokenizer.decode([tids[0]])
        tids = tokenizer.encode(target, add_special_tokens=False)
        if tids:
            p = probs[tids[0]].item()
            if p > max_prob:
                max_prob = p
                best_token = tokenizer.decode([tids[0]])

    return max_prob, best_token


# ============================================================
# Causal Tracing Implementation
# ============================================================

def causal_trace(prompt, subject, target_tokens, noise_std=0.1, n_noise_samples=5):
    """
    Perform causal tracing (Meng et al. 2022).

    Returns a 2D array [layer, position] where each value is the
    probability of the target token when that (layer, position) activation
    is restored from the clean run into the corrupted run.
    """
    # Step 1: Clean run
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        clean_outputs = model(**inputs)
    clean_logits = clean_outputs.logits[0, -1, :]
    clean_hidden_states = [h[0].detach().clone() for h in clean_outputs.hidden_states]  # list of [seq, hidden]

    clean_prob, clean_token = get_target_prob_fast(clean_logits, target_tokens)
    print(f"  Clean prob of target: {clean_prob:.4f} ('{clean_token}')")

    # Show what the model actually predicts
    top5_probs, top5_ids = torch.topk(torch.softmax(clean_logits, dim=-1), 5)
    top5_str = [(tokenizer.decode([tid.item()]).strip(), f"{p.item():.3f}") for tid, p in zip(top5_ids, top5_probs)]
    print(f"  Top-5 predictions: {top5_str}")

    # Step 2: Find subject positions
    subj_positions = find_subject_positions(prompt, subject)

    # Step 3: Corrupted run — add noise to subject token embeddings
    # We hook into the embedding layer to add noise
    corrupted_probs = []
    for _ in range(n_noise_samples):
        noise_hooks = []
        def make_noise_hook(positions, std):
            def hook_fn(module, input, output):
                # output is [batch, seq, hidden] from the embedding layer
                if isinstance(output, torch.Tensor):
                    noisy = output.clone()
                    for pos in positions:
                        noise = torch.randn_like(noisy[0, pos]) * std
                        noisy[0, pos] = noisy[0, pos] + noise
                    return noisy
                return output
            return hook_fn

        embed_layer = model.model.embed_tokens
        hook = embed_layer.register_forward_hook(make_noise_hook(subj_positions, noise_std))
        with torch.no_grad():
            corrupted_outputs = model(**inputs)
        hook.remove()

        corrupted_logits = corrupted_outputs.logits[0, -1, :]
        c_prob, c_token = get_target_prob_fast(corrupted_logits, target_tokens)
        corrupted_probs.append(c_prob)

    avg_corrupted_prob = np.mean(corrupted_probs)
    print(f"  Corrupted prob of target: {avg_corrupted_prob:.4f} (avg over {n_noise_samples} noise samples)")

    if clean_prob < 0.01:
        print(f"  WARNING: Model doesn't know this fact (clean prob < 1%). Skipping detailed trace.")
        return None, clean_prob, avg_corrupted_prob, seq_len, subj_positions

    # Step 4: Restoration — for each (layer, position), restore the clean activation
    # into the corrupted run and measure recovery
    # We only test a subset of layers for speed
    test_layers = list(range(0, num_layers + 1, 1))  # every layer
    restoration_grid = np.zeros((len(test_layers), seq_len))

    for li, layer_idx in enumerate(test_layers):
        for pos in range(seq_len):
            # Hook: add noise to embeddings AND restore hidden state at (layer, pos)
            def make_restore_hook(clean_state, restore_pos):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        restored = output.clone()
                        restored[0, restore_pos] = clean_state[restore_pos]
                        return restored
                    elif isinstance(output, tuple):
                        restored = output[0].clone()
                        restored[0, restore_pos] = clean_state[restore_pos]
                        return (restored,) + output[1:]
                    return output
                return hook_fn

            hooks = []
            # Add noise hook on embeddings
            embed_hook = embed_layer.register_forward_hook(
                make_noise_hook(subj_positions, noise_std)
            )
            hooks.append(embed_hook)

            # Add restore hook on the target layer
            if layer_idx == 0:
                # Restore at embedding output — same as the embed layer
                # This is handled by the clean_hidden_states[0]
                # We need to hook after the embedding but before layer 0
                # Actually, hidden_states[0] IS the embedding output
                # We'll hook the embedding to first add noise then restore
                embed_hook.remove()
                hooks.remove(embed_hook)

                def make_noise_and_restore_hook(positions, std, clean_state, restore_pos):
                    def hook_fn(module, input, output):
                        if isinstance(output, torch.Tensor):
                            noisy = output.clone()
                            for p in positions:
                                noise = torch.randn_like(noisy[0, p]) * std
                                noisy[0, p] = noisy[0, p] + noise
                            # Now restore at the specific position
                            noisy[0, restore_pos] = clean_state[restore_pos]
                            return noisy
                        return output
                    return hook_fn

                h = embed_layer.register_forward_hook(
                    make_noise_and_restore_hook(subj_positions, noise_std,
                                                clean_hidden_states[0], pos)
                )
                hooks.append(h)
            else:
                # Restore at a transformer layer output
                layer_module = model.model.layers[layer_idx - 1]
                restore_hook = layer_module.register_forward_hook(
                    make_restore_hook(clean_hidden_states[layer_idx], pos)
                )
                hooks.append(restore_hook)

            with torch.no_grad():
                restored_outputs = model(**inputs)
            for h in hooks:
                h.remove()

            restored_logits = restored_outputs.logits[0, -1, :]
            r_prob, _ = get_target_prob_fast(restored_logits, target_tokens)
            restoration_grid[li, pos] = r_prob

        print(f"    Layer {layer_idx:2d} done, max restoration: {restoration_grid[li].max():.4f}")

    return restoration_grid, clean_prob, avg_corrupted_prob, seq_len, subj_positions


# ============================================================
# Run causal tracing
# ============================================================
results = {}
all_grids = {}

# Run on key prompts
key_prompts = [0, 1, 3, 4, 5]  # Buffett→Berkshire, Buffett→Omaha, random→Berkshire, Einstein→relativity, random→relativity

for idx in key_prompts:
    fp = factual_prompts[idx]
    print(f"\n{'='*60}")
    print(f"Tracing: {fp['label']}")
    print(f"Prompt: '{fp['prompt']}'")
    print(f"{'='*60}")

    grid, clean_p, corrupt_p, seq_len, subj_pos = causal_trace(
        fp["prompt"], fp["subject"], fp["target_tokens"],
        noise_std=3.0,  # Gemma embeddings have larger norms
        n_noise_samples=3,
    )

    results[fp["label"]] = {
        "clean_prob": round(clean_p, 4),
        "corrupted_prob": round(corrupt_p, 4),
        "prompt": fp["prompt"],
        "subject": fp["subject"],
        "subject_positions": subj_pos,
    }

    if grid is not None:
        all_grids[fp["label"]] = grid
        results[fp["label"]]["max_restoration_prob"] = round(float(grid.max()), 4)
        # Find the peak restoration site
        peak = np.unravel_index(grid.argmax(), grid.shape)
        results[fp["label"]]["peak_layer"] = int(peak[0])
        results[fp["label"]]["peak_position"] = int(peak[1])


# ============================================================
# Plot causal tracing heatmaps
# ============================================================
n_plots = len(all_grids)
if n_plots > 0:
    fig, axes = plt.subplots(1, min(n_plots, 4), figsize=(6 * min(n_plots, 4), 8))
    if n_plots == 1:
        axes = [axes]

    for ax, (label, grid) in zip(axes, list(all_grids.items())[:4]):
        fp = [f for f in factual_prompts if f["label"] == label][0]
        input_ids = tokenizer.encode(fp["prompt"], add_special_tokens=True)
        token_labels = [tokenizer.decode([t]).strip()[:8] for t in input_ids]
        subj_pos = find_subject_positions(fp["prompt"], fp["subject"])

        im = ax.imshow(grid, aspect='auto', cmap='Reds', interpolation='nearest',
                       origin='lower')
        ax.set_xlabel("Token Position", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(0, num_layers + 1, 4))

        # Highlight subject positions
        for sp in subj_pos:
            ax.axvline(x=sp, color='blue', linestyle='--', alpha=0.5, linewidth=1)

        plt.colorbar(im, ax=ax, label='P(target)', shrink=0.8)

    fig.suptitle(f"Causal Tracing — {MODEL_ID}\n(Blue dashes = subject token positions)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/Users/x/src/lang-tokens/causal_tracing.png", dpi=150)
    print("\nSaved: causal_tracing.png")

# Also make a comparison bar chart: clean vs corrupted for each prompt
fig, ax = plt.subplots(figsize=(10, 5))
labels_list = list(results.keys())
clean_probs = [results[l]["clean_prob"] for l in labels_list]
corrupt_probs = [results[l]["corrupted_prob"] for l in labels_list]

x = np.arange(len(labels_list))
width = 0.35
bars1 = ax.bar(x - width/2, clean_probs, width, label='Clean (pointer intact)', color='#2ecc71')
bars2 = ax.bar(x + width/2, corrupt_probs, width, label='Corrupted (pointer destroyed)', color='#e74c3c')
ax.set_ylabel('P(correct fact)', fontsize=12)
ax.set_title('Named Entity as Pointer: Corrupt the Name, Lose the Knowledge', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("/Users/x/src/lang-tokens/causal_bar_chart.png", dpi=150)
print("Saved: causal_bar_chart.png")

# Save results
with open("/Users/x/src/lang-tokens/causal_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: causal_results.json")

print("\n" + "="*60)
print("CAUSAL TRACING COMPLETE")
print("="*60)
