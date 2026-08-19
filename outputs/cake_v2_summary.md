# Fixing CAKE's paraphrase-generalization and locality weaknesses

**Result: both weaknesses fixed.** On CounterFact single-edit (GPT-2-XL, EasyEdit's
native token-accuracy metric), a semantic-key redesign lifts CAKE from a combined score of
**0.33 to 0.90** — now clearly above ROME (0.67), the previous best on this benchmark.

| method | ES (efficacy) | PS (generalization) | NS (locality) | **score** |
|---|---|---|---|---|
| CAKE v1 (raw hₗ) | 1.00 | **0.15** | **0.69** | 0.329 |
| ROME (prior best) | 0.955 | 0.43 | 0.88 | 0.665 |
| **CAKE v2 (semantic key)** | **1.00** | **0.86** | **0.86** | **0.902** |

*v2 reported held-out: firing gate selected on a 100-record tune split, metrics reported on a
disjoint 100-record test split — the number is not fit on its own evaluation set.*

## Diagnosis — why v1 failed

CAKE v1 addressed its memory with the **raw last-token hidden state hₗ** of the written prompt.
A geometry probe over 40 CounterFact records measured what that key actually does, comparing the
edit prompt against its paraphrases (should be CLOSE) and its neighborhood prompts (same relation,
different subject — should be FAR):

| addressing key | paraphrase sim ↑ | neighbor sim ↓ | separation |
|---|---|---|---|
| **raw hₗ (v1)** | 0.735 | 0.770 | **−0.035** |
| MiniLM sentence-embed | 0.615 | 0.327 | **+0.288** |

The v1 key is **worse than useless**: it rates same-relation *neighbors* (0.77) as *more* similar
than genuine paraphrases (0.735) — a negative separation. No firing threshold can distinguish a real
hit from a neighbor on that geometry. That single fact caused **both** symptoms at once:
- paraphrases land far from the written key → miss the slot → **low PS (0.15)**
- neighbors land close to the written key → falsely trip the gate → **imperfect NS (0.69)**

## The fix — a semantic addressing key (still zero-gradient)

Change *what CAKE keys on*, nothing else:

1. **Key = sentence embedding of the prompt** (`all-MiniLM-L6-v2`, 384-d, normalized), which maps
   paraphrases together and different subjects apart.
2. **Project to the residual dim** with a fixed seeded Johnson–Lindenstrauss matrix (random
   Gaussian, no training) — preserves cosine geometry so the product-key table works unchanged.
3. **Everything else is identical to v1:** GPT-2 stays frozen, the value is still the answer-token
   unembedding direction, injection is still the logit-space bias / token playback. **No gradients
   anywhere** — the method remains a memory write, not a learning step.

The gate now operates in a well-separated cosine space, so it becomes a clean, tunable
**efficacy↔locality knob** (panel b): raising it trades generalization for locality along a smooth
curve. The tune split picked 0.45; at that point the held-out test gives ES 1.00 / PS 0.86 / NS 0.86.

## What moved, precisely

- **Generalization PS: 0.15 → 0.86** (+0.71). The decisive fix — paraphrases now address the same
  slot as the written prompt.
- **Locality NS: 0.69 → 0.86** (+0.17). Neighbors no longer trip the gate, because they are now
  genuinely far from the written key in the semantic space.
- **Efficacy ES: 1.00 → 1.00.** Unchanged — addressing on the exact prompt was never the problem.
- **Combined score: 0.33 → 0.90**, past ROME's 0.67.

## Honest caveats

- **A subject-only key scored higher in the probe (+0.36 separation) but is not deployable:** it
  needs the query's subject entity extracted at read time (NER on every question). The prompt key
  needs only the query text, so that is the reported method.
- **The value path still stores an answer-token direction, not full passage semantics** — the
  original v1 limitation for rich multi-fact chunks is unchanged; this fix targets *addressing*,
  which was the measured bottleneck.
- **These are GPT-2-XL / CounterFact numbers.** The mechanism is model-agnostic (the encoder is
  external), but zsRE and larger models are not yet run.
- The write cost rises modestly (a sentence-encoder forward pass per chunk) but stays
  gradient-free and far below any fine-tune / ROME edit.

## Bottom line

The paraphrase-generalization and locality weaknesses were a single root cause — a bad addressing
key — and swapping in a semantic key fixes both without adding any training. On CounterFact
single-edit, CAKE v2 now leads ROME on the combined metric while keeping CAKE's defining properties:
gradient-free, non-destructive (zero weight change), reversible.

## Files
- `gpt2_memory_semkey.py` — CAKE v2 (`GPT2WithSemanticMemory`)
- `eval_cf_semkey.py` — gate-sweep eval; `eval_cf_semkey_split.py` — held-out tune/test protocol
- `cake_v2.json` / `cake_v2.csv` — all numbers incl. sweep + geometry probe
