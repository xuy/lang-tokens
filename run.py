#!/usr/bin/env python3
"""
Token as Pointer: Named entities activate structured knowledge manifolds.

Runs three experiments on Gemma-2-2B:
  1. Activation geometry — cosine similarity and PCA across layers
  2. Causal tracing — corrupt the name, lose the knowledge (ROME methodology)
  3. Concept activation manifold — 52 investment concepts, probability distribution per entity

Usage:
  python run.py                  # run all experiments
  python run.py --experiment 1   # run only experiment 1
  python run.py --experiment 2   # run only experiment 2
  python run.py --experiment 3   # run only experiment 3
  python run.py --model google/gemma-2-2b  # specify model (default: gemma-2-2b)

Outputs figures/ and data/ directories.
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# ============================================================
# Config
# ============================================================

PROMPTS = {
    "Warren Buffett":   "Think and invest like Warren Buffett",
    "George Soros":     "Think and invest like George Soros",
    "a random person":  "Think and invest like a random person",
    "Albert Einstein":  "Think like Albert Einstein",
    "Elon Musk":        "Think like Elon Musk",
    "a scientist":      "Think like a scientist",
}

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

MANIFOLD_TEMPLATES = [
    "{entity} believes the key to investing is",
    "{entity} would advise focusing on",
    "{entity} thinks the most important factor in markets is",
    "When it comes to investing, {entity} emphasizes",
    "The investment philosophy of {entity} centers on",
]

MANIFOLD_PERSONAS = {
    "Warren Buffett": "Warren Buffett",
    "George Soros": "George Soros",
    "Jim Simons": "Jim Simons",
    "a random person": "a random person",
}

CAUSAL_PROMPTS = [
    {"prompt": "Warren Buffett is the CEO of", "subject": "Warren Buffett",
     "target_tokens": ["Berkshire", "Berk"], "label": "Buffett → Berkshire"},
    {"prompt": "Warren Buffett lives in", "subject": "Warren Buffett",
     "target_tokens": ["Omaha", "Om"], "label": "Buffett → Omaha"},
    {"prompt": "A random person is the CEO of", "subject": "A random person",
     "target_tokens": ["Berkshire", "Berk"], "label": "random → Berkshire (control)"},
    {"prompt": "Albert Einstein is famous for the theory of", "subject": "Albert Einstein",
     "target_tokens": ["relat", "general", "special"], "label": "Einstein → relativity"},
    {"prompt": "A random scientist is famous for the theory of", "subject": "A random scientist",
     "target_tokens": ["relat", "general", "special"], "label": "random → relativity (control)"},
]


# ============================================================
# Helpers
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id, device):
    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, output_hidden_states=True,
    ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Loaded: {num_layers} layers, {model.config.hidden_size}-dim hidden state")
    return model, tokenizer, num_layers


def find_entity_end_pos(tokenizer, text, entity_text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    full_tokens = inputs["input_ids"][0].tolist()
    for prefix in ["", " "]:
        entity_tokens = tokenizer.encode(prefix + entity_text, add_special_tokens=False)
        for i in range(len(full_tokens) - len(entity_tokens) + 1):
            if full_tokens[i:i + len(entity_tokens)] == entity_tokens:
                return inputs, i + len(entity_tokens) - 1
    return inputs, len(full_tokens) - 1


def get_target_prob(tokenizer, logits, target_tokens):
    probs = torch.softmax(logits, dim=-1).cpu()
    max_prob, best_token = 0.0, ""
    for target in target_tokens:
        for prefix in [" ", ""]:
            tids = tokenizer.encode(prefix + target, add_special_tokens=False)
            if tids:
                p = probs[tids[0]].item()
                if p > max_prob:
                    max_prob = p
                    best_token = tokenizer.decode([tids[0]])
    top_probs, top_ids = torch.topk(probs, 100)
    for i in range(100):
        tok_str = tokenizer.decode([top_ids[i].item()]).strip().lower()
        for target in target_tokens:
            if tok_str.startswith(target.lower()):
                if top_probs[i].item() > max_prob:
                    max_prob = top_probs[i].item()
                    best_token = tokenizer.decode([top_ids[i].item()])
    return max_prob, best_token


# ============================================================
# Experiment 1: Activation Geometry
# ============================================================

def experiment_geometry(model, tokenizer, num_layers, device, out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Activation Geometry")
    print("=" * 60)

    entities = list(PROMPTS.keys())
    short_names = ["Buffett", "Soros", "random person", "Einstein", "Musk", "scientist"]

    # Extract all-layer activations
    all_acts = {}
    for entity, prompt in PROMPTS.items():
        inputs, pos = find_entity_end_pos(tokenizer, prompt, entity, device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_acts[entity] = {
            i: outputs.hidden_states[i][0, pos, :].cpu().float().numpy()
            for i in range(len(outputs.hidden_states))
        }
        print(f"  {entity}: token pos {pos}")

    mid = num_layers // 2
    results = {}

    # Cosine similarity at key layers
    for layer in [mid, num_layers]:
        act_matrix = np.stack([all_acts[e][layer] for e in entities])
        cs = cosine_similarity(act_matrix)
        tag = "mid" if layer == mid else "final"
        results[f"cosine_sim_{tag}_layer{layer}"] = {
            f"{entities[i]} vs {entities[j]}": round(float(cs[i, j]), 4)
            for i in range(len(entities)) for j in range(i + 1, len(entities))
        }
        print(f"\n  Layer {layer} ({tag}):")
        for i, e in enumerate(short_names):
            row = "  ".join(f"{cs[i, j]:.3f}" for j in range(len(entities)))
            print(f"    {e:>15s}: {row}")

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, layer, title in [(axes[0], mid, f"Layer {mid} (Middle)"),
                              (axes[1], num_layers, f"Layer {num_layers} (Final)")]:
        act_matrix = np.stack([all_acts[e][layer] for e in entities])
        cs = cosine_similarity(act_matrix)
        triu = cs[np.triu_indices(len(entities), k=1)]
        vmin = max(0.0, float(np.min(triu)) - 0.05)
        im = ax.imshow(cs, cmap='RdYlBu_r', vmin=vmin, vmax=1.0)
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(short_names, fontsize=11)
        for i in range(len(entities)):
            for j in range(len(entities)):
                color = 'white' if cs[i, j] > (vmin + 1.0) / 2 + 0.1 else 'black'
                ax.text(j, i, f"{cs[i, j]:.3f}", ha='center', va='center', fontsize=10, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=13, fontweight='bold')
    fig.suptitle("Residual Stream Cosine Similarity", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/entity_similarity_heatmap.png", dpi=150)
    print(f"\n  Saved: {out_dir}/entity_similarity_heatmap.png")

    # PCA
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a', '#666666']
    markers = ['o', 'o', 's', '^', '^', 's']
    for ax, layer, title in [(axes[0], mid, f"Layer {mid} (Middle)"),
                              (axes[1], num_layers, f"Layer {num_layers} (Final)")]:
        act_matrix = np.stack([all_acts[e][layer] for e in entities])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(act_matrix)
        for i, (name, coord) in enumerate(zip(short_names, coords)):
            ax.scatter(coord[0], coord[1], c=colors[i], marker=markers[i], s=200,
                       zorder=5, edgecolors='black', linewidths=1.5)
            ax.annotate(name, (coord[0], coord[1]), fontsize=11, fontweight='bold',
                        xytext=(12, 12), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    fig.suptitle("PCA of Named Entity Activations", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/entity_pca.png", dpi=150)
    print(f"  Saved: {out_dir}/entity_pca.png")

    # Layer trajectory
    pairs = [
        ("Warren Buffett", "George Soros"), ("Warren Buffett", "Albert Einstein"),
        ("Warren Buffett", "a random person"), ("Albert Einstein", "Elon Musk"),
        ("a random person", "a scientist"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for (e1, e2), style in zip(pairs, ['-o', '-s', '-^', '-D', '-v']):
        sims = [float(cosine_similarity([all_acts[e1][l]], [all_acts[e2][l]])[0, 0])
                for l in range(num_layers + 1)]
        label = (e1.replace("Warren Buffett", "Buffett").replace("Albert Einstein", "Einstein")
                 .replace("a random person", "random").replace("a scientist", "scientist")
                 + " vs "
                 + e2.replace("Warren Buffett", "Buffett").replace("Albert Einstein", "Einstein")
                 .replace("a random person", "random").replace("a scientist", "scientist"))
        ax.plot(range(num_layers + 1), sims, style, label=label, markersize=4)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Entity Similarity Across Layers", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/entity_layer_trajectory.png", dpi=150)
    print(f"  Saved: {out_dir}/entity_layer_trajectory.png")

    return results


# ============================================================
# Experiment 2: Causal Tracing
# ============================================================

def experiment_causal(model, tokenizer, num_layers, device, out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Causal Tracing")
    print("=" * 60)

    embed_layer = model.model.embed_tokens
    results = {}
    all_grids = {}

    for fp in CAUSAL_PROMPTS:
        prompt, subject = fp["prompt"], fp["subject"]
        target_tokens, label = fp["target_tokens"], fp["label"]
        print(f"\n  Tracing: {label}")
        print(f"  Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Find subject positions
        _, subj_end = find_entity_end_pos(tokenizer, prompt, subject, device)
        subj_tokens = tokenizer.encode(" " + subject, add_special_tokens=False)
        subj_start = subj_end - len(subj_tokens) + 1
        subj_positions = list(range(max(1, subj_start), subj_end + 1))

        # Clean run
        with torch.no_grad():
            clean_out = model(**inputs)
        clean_logits = clean_out.logits[0, -1, :]
        clean_hidden = [h[0].detach().clone() for h in clean_out.hidden_states]
        clean_prob, clean_tok = get_target_prob(tokenizer, clean_logits, target_tokens)
        print(f"  Clean: P={clean_prob:.4f} ('{clean_tok}')")

        # Corrupted run
        noise_std = 3.0
        corrupted_probs = []
        for _ in range(3):
            def noise_hook(module, input, output, positions=subj_positions, std=noise_std):
                if isinstance(output, torch.Tensor):
                    noisy = output.clone()
                    for p in positions:
                        noisy[0, p] += torch.randn_like(noisy[0, p]) * std
                    return noisy
                return output
            h = embed_layer.register_forward_hook(noise_hook)
            with torch.no_grad():
                c_out = model(**inputs)
            h.remove()
            c_prob, _ = get_target_prob(tokenizer, c_out.logits[0, -1, :], target_tokens)
            corrupted_probs.append(c_prob)
        avg_corrupt = np.mean(corrupted_probs)
        print(f"  Corrupted: P={avg_corrupt:.4f}")

        results[label] = {
            "clean_prob": round(clean_prob, 4),
            "corrupted_prob": round(avg_corrupt, 4),
            "prompt": prompt, "subject": subject,
        }

        if clean_prob < 0.01:
            print(f"  Model doesn't know this fact. Skipping detailed trace.")
            continue

        # Restoration grid
        grid = np.zeros((num_layers + 1, seq_len))
        for layer_idx in range(num_layers + 1):
            for pos in range(seq_len):
                hooks = []

                if layer_idx == 0:
                    def combo_hook(module, input, output, positions=subj_positions,
                                   std=noise_std, clean=clean_hidden[0], rpos=pos):
                        if isinstance(output, torch.Tensor):
                            noisy = output.clone()
                            for p in positions:
                                noisy[0, p] += torch.randn_like(noisy[0, p]) * std
                            noisy[0, rpos] = clean[rpos]
                            return noisy
                        return output
                    hooks.append(embed_layer.register_forward_hook(combo_hook))
                else:
                    def noise_hook_2(module, input, output, positions=subj_positions, std=noise_std):
                        if isinstance(output, torch.Tensor):
                            noisy = output.clone()
                            for p in positions:
                                noisy[0, p] += torch.randn_like(noisy[0, p]) * std
                            return noisy
                        return output
                    hooks.append(embed_layer.register_forward_hook(noise_hook_2))

                    def restore_hook(module, input, output, clean=clean_hidden[layer_idx], rpos=pos):
                        if isinstance(output, torch.Tensor):
                            r = output.clone()
                            r[0, rpos] = clean[rpos]
                            return r
                        elif isinstance(output, tuple):
                            r = output[0].clone()
                            r[0, rpos] = clean[rpos]
                            return (r,) + output[1:]
                        return output
                    hooks.append(model.model.layers[layer_idx - 1].register_forward_hook(restore_hook))

                with torch.no_grad():
                    r_out = model(**inputs)
                for hook in hooks:
                    hook.remove()
                r_prob, _ = get_target_prob(tokenizer, r_out.logits[0, -1, :], target_tokens)
                grid[layer_idx, pos] = r_prob
            print(f"    Layer {layer_idx:2d}: max restoration {grid[layer_idx].max():.4f}")

        all_grids[label] = grid
        results[label]["peak_layer"] = int(np.unravel_index(grid.argmax(), grid.shape)[0])
        results[label]["peak_position"] = int(np.unravel_index(grid.argmax(), grid.shape)[1])

    # Heatmaps
    n = len(all_grids)
    if n > 0:
        fig, axes = plt.subplots(1, min(n, 4), figsize=(6 * min(n, 4), 8))
        if n == 1:
            axes = [axes]
        for ax, (label, grid) in zip(axes, list(all_grids.items())[:4]):
            fp = [f for f in CAUSAL_PROMPTS if f["label"] == label][0]
            toks = tokenizer.encode(fp["prompt"], add_special_tokens=True)
            tok_labels = [tokenizer.decode([t]).strip()[:8] for t in toks]
            im = ax.imshow(grid, aspect='auto', cmap='Reds', interpolation='nearest', origin='lower')
            ax.set_xlabel("Token Position", fontsize=10)
            ax.set_ylabel("Layer", fontsize=10)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(tok_labels)))
            ax.set_xticklabels(tok_labels, rotation=60, ha='right', fontsize=8)
            plt.colorbar(im, ax=ax, label='P(target)', shrink=0.8)
        fig.suptitle("Causal Tracing — Corrupt the Name, Lose the Knowledge", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/causal_tracing.png", dpi=150)
        print(f"\n  Saved: {out_dir}/causal_tracing.png")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_list = list(results.keys())
    x = np.arange(len(labels_list))
    ax.bar(x - 0.175, [results[l]["clean_prob"] for l in labels_list], 0.35,
           label='Clean (pointer intact)', color='#2ecc71')
    ax.bar(x + 0.175, [results[l]["corrupted_prob"] for l in labels_list], 0.35,
           label='Corrupted (pointer destroyed)', color='#e74c3c')
    ax.set_ylabel('P(correct fact)', fontsize=12)
    ax.set_title('Named Entity as Pointer: Corrupt the Name, Lose the Knowledge', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/causal_bar_chart.png", dpi=150)
    print(f"  Saved: {out_dir}/causal_bar_chart.png")

    return results


# ============================================================
# Experiment 3: Concept Activation Manifold
# ============================================================

def experiment_manifold(model, tokenizer, device, out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Concept Activation Manifold")
    print("=" * 60)

    def get_concept_probs(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu()
        out = {}
        for concept in ALL_CONCEPTS:
            max_p = 0.0
            for prefix in [" ", ""]:
                tids = tokenizer.encode(prefix + concept, add_special_tokens=False)
                if tids:
                    max_p = max(max_p, probs[tids[0]].item())
            out[concept] = max_p
        return out

    # Collect
    persona_avg = {}
    for name, entity_str in MANIFOLD_PERSONAS.items():
        print(f"\n  {name}:")
        all_probs = []
        for tmpl in MANIFOLD_TEMPLATES:
            prompt = tmpl.format(entity=entity_str)
            print(f"    {prompt}")
            all_probs.append(get_concept_probs(prompt))
        persona_avg[name] = {c: np.mean([p[c] for p in all_probs]) for c in ALL_CONCEPTS}
        top = sorted(persona_avg[name].items(), key=lambda x: -x[1])[:5]
        print(f"    Top-5: {[(c, f'{p:.4f}') for c, p in top]}")

    # Compute axis mass
    axis_names = list(CONCEPT_AXES.keys())
    axis_mass = {}
    for persona in MANIFOLD_PERSONAS:
        total = sum(persona_avg[persona][c] for c in ALL_CONCEPTS)
        axis_mass[persona] = {
            ax: sum(persona_avg[persona][c] for c in concepts) / total if total > 0 else 0
            for ax, concepts in CONCEPT_AXES.items()
        }

    # Lift vs baseline
    baseline = "a random person"
    lift_data = {}
    for persona in MANIFOLD_PERSONAS:
        if persona == baseline:
            continue
        lift_data[persona] = {}
        for c in ALL_CONCEPTS:
            bp = persona_avg[baseline][c]
            lift_data[persona][c] = persona_avg[persona][c] / bp if bp > 1e-8 else 0
        top_lift = sorted(lift_data[persona].items(), key=lambda x: -x[1])[:5]
        print(f"\n  {persona} top lift vs random: {[(c, f'{l:.0f}x') for c, l in top_lift]}")

    # Radar chart
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
        vals = [axis_mass[persona][ax] for ax in axis_names] + [axis_mass[persona][axis_names[0]]]
        ax.plot(angles, vals, ls, linewidth=2.8, label=persona, color=color,
                marker=marker, markersize=9, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=12, fontweight='bold', color='#333')
    max_val = max(axis_mass[p][ax] for p in MANIFOLD_PERSONAS for ax in axis_names)
    ticks = np.arange(0.05, max_val + 0.1, 0.05)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:.0%}' for t in ticks], fontsize=8, color='#666')
    ax.set_ylim(0, max_val + 0.05)
    ax.set_title("Where Does the Probability Mass Land?\nConcept activation profile by entity name\n",
                 fontsize=16, fontweight='bold', pad=25, color='#222')
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08), fontsize=12,
              frameon=True, fancybox=True, edgecolor='#ccc', facecolor='white')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/manifold_radar.png", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n  Saved: {out_dir}/manifold_radar.png")

    # Lift bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    group_colors = {}
    palette = ['#c0392b', '#e67e22', '#2980b9', '#27ae60', '#8e44ad', '#7f8c8d']
    for i, ax_name in enumerate(axis_names):
        for c in CONCEPT_AXES[ax_name]:
            group_colors[c] = palette[i]

    for ax, persona in [(axes[0], "Warren Buffett"), (axes[1], "George Soros")]:
        sorted_c = sorted(ALL_CONCEPTS, key=lambda c: -lift_data[persona][c])
        colors = [group_colors[c] for c in sorted_c]
        vals = [lift_data[persona][c] for c in sorted_c]
        ax.barh(range(len(sorted_c)), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_c)))
        ax.set_yticklabels(sorted_c, fontsize=8)
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Probability lift (×)", fontsize=11)
        ax.set_title(f"{persona} vs Random Person", fontsize=12, fontweight='bold')
        ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[i], label=ax_name.replace('\n', ' '))
                       for i, ax_name in enumerate(axis_names)]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/manifold_lift.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_dir}/manifold_lift.png")

    return {
        "axis_mass": axis_mass,
        "persona_avg_probs": {p: {c: round(v, 6) for c, v in probs.items()}
                              for p, probs in persona_avg.items()},
        "lift_vs_baseline": {p: {c: round(l, 2) for c, l in lift.items()}
                             for p, lift in lift_data.items()},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Token as Pointer experiments")
    parser.add_argument("--model", default="google/gemma-2-2b", help="HuggingFace model ID")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], help="Run only this experiment")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, num_layers = load_model(args.model, device)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    all_results = {"model": args.model, "num_layers": num_layers}
    run_all = args.experiment is None

    if run_all or args.experiment == 1:
        all_results["geometry"] = experiment_geometry(model, tokenizer, num_layers, device, "figures")

    if run_all or args.experiment == 2:
        all_results["causal"] = experiment_causal(model, tokenizer, num_layers, device, "figures")

    if run_all or args.experiment == 3:
        all_results["manifold"] = experiment_manifold(model, tokenizer, device, "figures")

    with open("data/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: data/results.json")
    print("\nDone. Figures in figures/, data in data/.")


if __name__ == "__main__":
    main()
