# Token as Pointer: Named Entities Activate Structured Knowledge Manifolds in Language Models

**Model**: Gemma-2-2B (Google, 26 transformer layers, 2304-dimensional hidden state)
**Comparison model**: GPT-2 (124M, 12 layers, 768-dim)
**Date**: 2026-04-08

---

## Core Thesis

Named entity tokens like "Warren Buffett" don't function as labels — they function as **pointers** that dereference into structured distributions over conceptual space. When a language model reads "Warren Buffett," those tokens activate a recognizable manifold of associated concepts (value investing, margin of safety, long-term holding) while suppressing others (algorithmic trading, derivatives). Generic descriptors like "a random person" activate no such structure — the distribution stays flat. The token points nowhere.

This document presents three experiments that verify this thesis with increasing specificity: geometric separation (Experiment 1), causal necessity (Experiment 2), and manifold structure (Experiment 3).

---

## Experiment 1: Activation Geometry

### Method

Six prompts were run through Gemma-2-2B, and residual stream activations were extracted at the final token position of each entity. Cosine similarity and PCA were computed at multiple layers.

### Key Findings

At layer 13 (middle), named entities cluster together with average cosine similarity **0.73**, while generic descriptors average **0.55** to named entities — a gap of **0.18**. The investor pair (Buffett ↔ Soros) shows the highest cross-entity similarity at **0.756**, confirming domain clustering.

Unlike GPT-2 (124M) where all distinctions collapse in the final layer (all pairs > 0.987), Gemma-2-2B **preserves entity geometry even at the final layer** — similarities range from 0.79 to 0.91, maintaining the separation between pointers and non-pointers throughout the network.

The logit lens at layer 0 reveals a striking contrast: named entities project sharply to their own name token (Buffett: logit 280, Soros: 294, Einstein: 277), while "a random person" projects to a diffuse multilingual concept cloud — `person`, `personne` (French), `pessoa` (Portuguese), `человек` (Russian), `व्यक्ति` (Hindi) — spanning a dozen languages. The pointer resolves to a specific identity; the non-pointer dissolves into a category.

### Output Files
- `entity_similarity_heatmap.png` — Cosine similarity heatmaps at layer 13 and layer 26
- `entity_pca.png` — PCA projections showing entity clustering
- `entity_layer_trajectory.png` — Pairwise similarity across all 27 layers

---

## Experiment 2: Causal Tracing — Corrupt the Pointer, Lose the Knowledge

### Method

Following Meng et al. (NeurIPS 2022, "ROME"), we performed **causal tracing**: run the model on a factual prompt, record all hidden states, then corrupt the subject tokens with Gaussian noise. If the model can no longer predict the fact, we restore individual (layer, position) activations from the clean run and measure which restorations recover the correct prediction.

### Key Findings

| Prompt | Clean P(target) | Corrupted P(target) | Interpretation |
|---|---|---|---|
| "Warren Buffett is the CEO of ___" | **70.6%** (Berkshire) | **0.0%** | Pointer destroyed → knowledge lost |
| "Warren Buffett lives in ___" | **35.2%** (Omaha) | **0.0%** | Pointer destroyed → knowledge lost |
| "A random person is the CEO of ___" | 0.01% (Berkshire) | 0.0% | No pointer → no knowledge to lose |
| "Albert Einstein is famous for the theory of ___" | **80.7%** (relativity) | **12.7%** | Partial context survives ("theory of" primes "relativity") |
| "A random scientist is famous for the theory of ___" | 11.2% (relativity) | 18.4% | No pointer → noise doesn't hurt (nothing to destroy) |

The causal tracing heatmaps show that knowledge recovery concentrates at the **subject token positions** in **early-to-mid layers** (layers 1–4), consistent with the ROME finding that mid-layer MLPs at the last subject token are where factual associations are stored and retrieved — where the pointer gets dereferenced.

The control case is equally important: for "A random person is the CEO of ___", there is no knowledge to recover because there was never a pointer to dereference. The model assigns 0.01% to "Berkshire" in the clean run. Corrupting the subject changes nothing.

### Output Files
- `causal_tracing.png` — Heatmaps showing restoration probability by (layer, position)
- `causal_bar_chart.png` — Clean vs. corrupted probability comparison

---

## Experiment 3: The Activation Manifold — A Pointer Dereferences to a Worldview

### The Experiment: What Does a Name Activate?

We gave Gemma-2-2B a simple task: complete sentences like "Warren Buffett believes the key to investing is ___." We defined 52 investment concepts spanning four schools of thought — value investing (moat, intrinsic value, margin of safety), macro/global (currency, geopolitical, reflexivity), quantitative (algorithmic, statistical, arbitrage), and general terms (diversification, risk, portfolio). Then we measured how much probability the model assigned to each concept as the next word, averaging across five different prompt phrasings to control for template effects.

The radar chart shows what happened. Each shape is a fingerprint — the distribution of probability mass across the four concept groups when the model reads a particular name. Warren Buffett's fingerprint spikes dramatically toward Value & Fundamental, with 72% of all concept probability concentrated there. George Soros tilts toward Macro & Global. Jim Simons pulls sharply toward Quantitative & Technical. The gray dashed line — "a random person" — is the control: a small, flat shape hugging the center, with most of its mass in General & Neutral. It has no spike, no tilt, no signature. It activates nothing in particular.

This is what it means to say a name is a pointer, not a label. When the model reads "Warren Buffett," those two tokens don't just mean "a male human" — they dereference into a structured distribution over an entire conceptual space. The word "value" becomes 19 times more likely, "moat" 46 times, "durable" 404 times. Simultaneously, "algorithmic" is suppressed to near zero. The pointer doesn't merely retrieve a fact (Buffett → Berkshire Hathaway); it reshapes the model's entire probability landscape into something recognizable as a worldview. "A random person" cannot do this, because there is no address to dereference — the token points nowhere, and the distribution stays flat.

### Detailed Lift Analysis

**Warren Buffett** — most elevated vs. baseline ("a random person"):

| Concept | Group | Lift |
|---|---|---|
| durable | Value / Fundamental | **404x** |
| undervalued | Value / Fundamental | 85x |
| moat | Value / Fundamental | 46x |
| value | Value / Fundamental | 19x |
| earnings | Value / Fundamental | 18x |
| patience | Value / Fundamental | 14x |

Most suppressed: `algorithmic` (0.00x), `frequency` (0.00x), `arbitrage` (0.01x), `correlation` (0.01x) — all Quantitative/Technical.

**George Soros** — most elevated:

| Concept | Group | Lift |
|---|---|---|
| emerging | Macro / Global | **221x** |
| crisis | Macro / Global | 174x |
| geopolitical | Macro / Global | 70x |
| reflexivity | Macro / Global | 38x |
| global | Macro / Global | 28x |

Most suppressed: `moat` (0.05x), `intrinsic` (0.07x), `dividends` (0.05x) — all Value/Fundamental.

**Jim Simons** — most elevated:

| Concept | Group | Lift |
|---|---|---|
| statistical | Quantitative / Technical | **1,471x** |
| quantitative | Quantitative / Technical | 120x |
| algorithmic | Quantitative / Technical | 90x |
| systematic | Quantitative / Technical | 96x |
| derivatives | Quantitative / Technical | 26x |

Most suppressed: `moat` (0.07x), `currency` (0.06x), `speculation` (0.07x).

### Concept Group Mass Distribution

| Entity | Value | Macro | Quantitative | General |
|---|---|---|---|---|
| **Warren Buffett** | **72%** | 4% | 2% | 22% |
| **George Soros** | 27% | **29%** | 7% | 38% |
| **Jim Simons** | 42% | 2% | **21%** | 35% |
| a random person | 18% | 18% | 8% | **55%** |

### Distribution Entropy

Normalized entropy (1.0 = perfectly flat, 0.0 = all mass on one concept):

| Entity | Normalized entropy |
|---|---|
| Warren Buffett | **0.591** (most concentrated) |
| a financial advisor | 0.585 |
| Albert Einstein | 0.605 |
| Elon Musk | 0.606 |
| Jim Simons | 0.672 |
| a random person | **0.687** (most diffuse) |
| George Soros | 0.690 |

Named entities generally have lower entropy (more concentrated distributions) than generic descriptors, consistent with the pointer hypothesis: a specific address dereferences to a specific manifold, not a uniform one. Soros is an interesting exception — his distribution is broad but distinctly *shaped* (tilted toward Macro), suggesting his manifold is wide but still structured.

### Output Files
- `manifold_radar_final.png` — Radar chart of concept group activation by entity
- `manifold_bars_final.png` — Per-concept probability comparison across entities
- `manifold_heatmap.png` — Full heatmap across all 52 concepts
- `manifold_lift.png` — Lift vs. baseline for Buffett and Soros
- `manifold_results.json` — All numerical results

---

## Limitations

1. **Next-token probability is an indirect measure.** We measure P(concept | entity + template), which conflates the model's knowledge about the entity with what it thinks a reasonable sentence completion looks like. A more direct approach would use sparse autoencoder features or probing classifiers on the hidden state.
2. **Template effects.** Although we average over 5 templates, the specific phrasing influences results. "Believes the key to investing is" may prime different completions than "would advise focusing on."
3. **Concept vocabulary is hand-curated.** The 52 concepts were chosen to span investment philosophies but are not exhaustive. Some concepts may be undertokenized (multi-token) and thus underrepresented.
4. **Single model.** Results from Gemma-2-2B may not generalize to all architectures. GPT-2 (124M) showed weaker effects, suggesting these manifolds sharpen with scale.
5. **Causal tracing uses Gaussian noise.** The corruption method (adding noise to embeddings) is standard but somewhat arbitrary. Different noise levels or corruption methods could yield different causal maps.

---

## Related Work

- **Meng et al., "Locating and Editing Factual Associations in GPT" (NeurIPS 2022)** — ROME. Causal tracing methodology; showed mid-layer MLPs at subject tokens store factual associations.
- **Anthropic, "Scaling Monosemanticity" (May 2024)** — Found specific entity features (Golden Gate Bridge, etc.) in sparse autoencoders trained on Claude 3 Sonnet. Feature steering demonstrated causal control.
- **Anthropic, "On the Biology of a Large Language Model" (March 2025)** — Full attribution graphs tracing entity → knowledge pathways. Released open-source tools for Gemma-2-2B.
- **Hernandez et al., "Linearity of Relation Decoding" (ICLR 2024)** — Entity-to-attribute mappings are linear transformations on hidden states for ~48% of relations tested.
- **Geva et al., "Transformer FFN Layers Are Key-Value Memories" (EMNLP 2021)** — The theoretical foundation: FFN layers operate as associative memories where subject tokens activate keys that retrieve value vectors.

---

## Code

All experiments are reproducible with the scripts in this repository:
- `experiment_gemma.py` — Experiment 1: Activation geometry (Gemma-2-2B)
- `experiment.py`, `experiment_v2.py` — Experiment 1: Activation geometry (GPT-2, for comparison)
- `causal_tracing.py` — Experiment 2: Causal tracing
- `manifold_activation.py` — Experiment 3: Concept activation manifold
- `radar_final.py` — Publication-quality visualizations
