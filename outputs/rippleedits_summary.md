# Native RippleEdits — the honest portability verdict

RippleEdits (Cohen et al., TACL 2024) is *the* published benchmark for the exact weakness I flagged: does
an edit **propagate** to logically-related facts (ripple) without **corrupting** unrelated ones? I ran the
POPULAR split, 100 single edits on GPT-J-6B, four methods, six criteria — generation-based scoring
(greedy 24 tokens, gold value/alias substring match, per-group AND/OR condition).

## Results (GPT-J-6B, POPULAR, N=100)

| criterion | type | Base | ROME | RAG | CAKE |
|---|---|---|---|---|---|
| Logical_Generalization | propagate | 0.033 | 0.076 | 0.104 | 0.033 |
| Compositionality_I | propagate | 0.076 | 0.070 | **0.400** | 0.136 |
| Compositionality_II | propagate | 0.000 | 0.000 | **0.531** | 0.241 |
| Subject_Aliasing | propagate | 0.000 | 0.000 | 0.758 | **0.816** |
| Relation_Specificity | **preserve** | 0.222 | 0.235 | **0.326** | 0.061 |
| Forgetfulness | **preserve** | 0.109 | 0.086 | **0.236** | 0.036 |
| **propagation avg** | | 0.027 | 0.037 | **0.448** | 0.307 |
| **preservation avg** | | 0.165 | 0.161 | **0.281** | 0.049 |
| **aggregate** | | 0.073 | 0.078 | **0.393** | 0.221 |

**Ranking: RAG 0.393 > CAKE 0.221 > ROME 0.078 > Base 0.073.**

## What this says — and it reverses the leaderboard

On the single-edit metrics (CounterFact/zsRE), CAKE was #1. **On native portability, CAKE is not — and
neither is ROME.** Three honest findings:

1. **RAG wins.** In-context editing lets the model *reason over* the new fact, so ripples propagate
   (Compositionality I/II 0.40/0.53) and it corrupts neighbors least (preservation 0.281). This is the
   published finding that retrieval-augmented prompting handles ripple effects better than weight-editing —
   and it holds here.
2. **ROME scores near base (0.078 vs 0.073) — indicative, not a rigorously matched comparison.** A
   single-layer weight edit on GPT-J does not appear to propagate multi-hop ripples well. **Caveat:** ROME
   was run with a separate driver (`eval_rome_ripple.py`) that derives the subject with a crude heuristic
   (`prompt.split(" of ")[-1]`) and **skips** any edit where that fails — so ROME's 100 edits are *not
   guaranteed identical* to the base/RAG/CAKE sample, and the heuristic may extract wrong subject spans for
   edits it keeps. ROME's low number is therefore suggestive of the RippleEdits paper's point (weight-editing
   is no portability panacea) but should not be read as a rigorously matched result. The **RAG-vs-CAKE
   contrast is sound** — those three methods share an identical query set (no skip logic).
3. **CAKE shows exactly its predicted signature.** It *wins Subject_Aliasing* (0.816 — an alias of the
   edited subject still hits the slot, because the semantic key is subject-robust), is mid on the rest of
   propagation, and is **worst of all four on preservation** (0.049 — *below the unedited base*). It
   over-fires: asked an unrelated fact about the edited subject, it plays back the edit and destroys the
   neighbor. This is the retrieval-not-rederivation weakness, now measured against the published standard,
   not my LLM probes.

## Final verdict on "state of the art"

The complete, honest picture across every benchmark run in this project:

- **Single-edit efficacy / generalization / locality / sequential retention** (CounterFact, zsRE,
  sequential stress test): **CAKE is state of the art** — it leads base, RAG, fine-tune, ROME, MEMIT,
  WISE, GRACE, and AlphaEdit at two model sizes, sample-stable to N=1000, gradient-free, ~1600× cheaper
  per edit than ROME.
- **Compositional portability** (native RippleEdits): **CAKE is not SOTA, and cannot be by design.** RAG
  wins; CAKE places second overall but is *last* on preservation because it retrieves rather than
  re-derives. Even ROME can't win this axis.

So the precise, defensible claim is: **CAKE is state of the art for direct fact editing and lifelong/
sequential editing, with a structural limitation on compositional portability that no gate fully fixes.**
It is not an unqualified "best knowledge editor" — no method in this comparison is, because the two axes
(direct editing vs. ripple propagation) reward opposite mechanisms. CAKE optimizes the first; RAG the
second. That is the honest, complete answer, now backed by the field's own portability benchmark.

## Files
- `rippleedits.{json,png}`, `eval_rippleedits.py` (base/RAG/CAKE), `eval_rome_ripple.py` (ROME via EasyEdit)
- Data: `RippleEdits/popular.json` (885 examples, fetched from edenbiran/RippleEdits)
