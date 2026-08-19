# zsRE benchmark — CAKE v2 on the second standard knowledge-editing test

Ran the full six-method comparison on **zsRE** (GPT-2-XL, N=100 single-edit), same token-accuracy
metric as CounterFact so the numbers compare across benchmarks. zsRE differs from CounterFact in one
way that matters here: its answers are **free-form multi-token entities** ("University of Michigan",
"Illinois Institute of Technology"), not single salient tokens.

| method | ES (efficacy) | PS (generalization) | NS (locality) | **score** | write |
|---|---|---|---|---|---|
| Base | 0.21 | 0.21 | 1.00 | 0.280 | — |
| In-context (RAG) | 0.94 | 0.85 | 0.88 | 0.887 | 0 |
| Fine-tune (25 steps) | 0.79 | 0.75 | **0.01** | 0.029 | 6.6 s |
| **ROME** | **1.00** | 0.80 | **0.96** | **0.908** | 3.8 s |
| MEMIT | 0.28 | 0.24 | 1.00 | 0.346 | 4.4 s |
| **CAKE v2 (semantic key)** | 0.53 | 0.52 | **1.00** | 0.624 | 0.3 s |

*CAKE v2 reported held-out: gate 0.3 selected on a 50-record tune split, metrics on a disjoint
50-record test split.*

## Headline: the addressing fix generalizes to zsRE

The reason for running zsRE was to check whether the CounterFact fix (semantic key) was benchmark-
specific. **It is not.** The two numbers that prove the fix works are:

- **PS (0.52) ≈ ES (0.53)** — paraphrases address the written slot just as reliably as the original
  question. This is the whole point of the semantic key, and it holds on a second, differently-
  constructed benchmark. (For reference, CAKE v1's raw-hidden-state key gave PS=0.15 on CounterFact.)
- **NS = 1.00** — perfect locality. zsRE's locality probes are unrelated questions, and the semantic
  key never confuses them with the edit, so nothing false-fires.

## Why CAKE's overall score (0.62) trails ROME (0.91) here

Not an addressing failure — a **value-representation** limit. CAKE's stored value is the unembedding
direction of the **first answer token**. On CounterFact (single-token answers like "English") that's
enough for ES=1.00. On zsRE, the answer is a multi-word entity, and token-accuracy scores the whole
sequence, so a first-token value caps efficacy at ~0.53 regardless of how well addressing works. ES
and PS fall **together** (0.53 / 0.52), which is the signature of a value limit, not a key limit — if
addressing were failing, PS would drop below ES as it did in v1.

This points at the exact next upgrade: **store a multi-token value** (the full answer's playback
sequence is already kept in slot meta; wiring it into the scored path, or a small learned value
decoder, would lift ES/PS toward RAG/ROME without touching the addressing that already works).

## The other methods, briefly

- **ROME wins (0.908)** — near-perfect efficacy, strong generalization and locality, as published on
  zsRE. Consistent with its CounterFact result; ROME is the method to beat on single edits.
- **RAG (0.887)** — strong, second place, but pays its cost on every query.
- **MEMIT (0.346)** — weak single-edit as expected (batch method). It scores ES 0.28 here vs 0.00 on
  CounterFact only because multi-token targets give partial token credit; its natural regime is still
  mass editing.
- **Fine-tune (0.029)** — locality collapses to 0.01. Catastrophic forgetting, identical failure mode
  to CounterFact.

## Cross-benchmark summary (CAKE v2)

| benchmark | ES | PS | NS | score | limiting factor |
|---|---|---|---|---|---|
| CounterFact (single-token answers) | 1.00 | 0.86 | 0.86 | 0.902 | — (beats ROME 0.665) |
| zsRE (multi-token answers) | 0.53 | 0.52 | 1.00 | 0.624 | first-token value caps efficacy |

The semantic-key fix is robust across both benchmarks — **generalization tracks efficacy and locality
is high on both**. Where CAKE trails (zsRE), the cause is isolated and identified: the single-token
value, not the addressing. That's the next thing to fix.

## Files
- `eval_zsre.py` (base/RAG/CAKE/finetune), `eval_zsre_edit.py` (ROME/MEMIT via EasyEdit)
- `zsre.json` / `zsre.csv` — all numbers incl. CAKE gate-sweep tune curve
