# Token as Pointer

Named entity tokens don't function as labels — they function as **pointers** that dereference into structured distributions over conceptual space.

When a language model reads "Warren Buffett," those tokens activate a recognizable manifold of associated concepts (value investing, moat, patience, margin of safety) while suppressing others (algorithmic trading, derivatives). "A random person" activates no such structure — the distribution stays flat. The token points nowhere.

![Radar chart showing distinct concept activation profiles for Buffett, Soros, Simons, and a random person](figures/manifold_radar.png)

## Experiments

Three experiments on **Gemma-2-2B** verify this with increasing specificity:

### 1. Activation Geometry
Residual stream activations at named entity tokens cluster by identity and domain. Named entities average cosine similarity 0.73 to each other vs. 0.55 to generic descriptors — a gap of 0.18.

### 2. Causal Tracing ([Meng et al. 2022](https://arxiv.org/abs/2202.05262))
Corrupt the entity token with noise → the model loses the associated knowledge. "Warren Buffett is the CEO of ___" drops from 70.6% → 0.0% for "Berkshire." Destroy the pointer, lose the knowledge.

### 3. Concept Activation Manifold
52 investment concepts across 6 dimensions (value analysis, macro, quantitative, etc.), measured across 5 prompt templates. Each entity activates a distinct, recognizable distribution:

| Entity | Top axis | Mass |
|---|---|---|
| Warren Buffett | Value Analysis | 42% |
| George Soros | Macro & Geopolitics | 22% |
| Jim Simons | Quantitative Methods | 17% |
| a random person | Growth & Markets (generic) | 32% |

Buffett elevates "durable" by 404x, "moat" by 46x, "value" by 19x vs. baseline — while suppressing "algorithmic" to 0.00x.

## Running

```bash
pip install torch transformers scikit-learn matplotlib numpy

# Run all experiments (~30 min on Apple Silicon, ~10 min on GPU)
python run.py

# Run a single experiment
python run.py --experiment 1   # geometry
python run.py --experiment 2   # causal tracing
python run.py --experiment 3   # manifold

# Use a different model
python run.py --model google/gemma-2-2b
```

Outputs go to `figures/` (PNGs) and `data/` (JSON).

## Related Work

- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) — ROME, Meng et al. (NeurIPS 2022)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Anthropic (2024)
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) — Anthropic (2025)
- [Linearity of Relation Decoding](https://arxiv.org/abs/2308.09124) — Hernandez et al. (ICLR 2024)
