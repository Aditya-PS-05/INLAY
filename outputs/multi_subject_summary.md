# Multi-subject benchmark — 24 facts, 24 different entities (GPT-2-XL)

This is the fair test I promised: **24 facts, each about a *different* fictional entity**, which is
the case ROME and MEMIT were built for (mass editing across many subjects) — not the single-subject
case that handicapped them earlier. Three standard knowledge-editing axes on an NVIDIA L40S:

- **Efficacy** — fact recalled on the exact prompt
- **Generalization** — fact recalled on a *paraphrased* query (the hard axis)
- **Locality** — 12 unrelated general-knowledge prompts unchanged vs base (1.0 = no collateral damage)

| method | efficacy | generalization | locality | write cost | gradients |
|---|---|---|---|---|---|
| Base GPT-2-XL | 0.00 | 0.00 | 1.00 | — | 0 |
| In-context (RAG) | 1.00 | 0.54 | 1.00 | 0 *(24-fact doc re-fed every query)* | 0 |
| Fine-tune (60 steps) | 1.00 | **0.83** | **0.00** | 123 s | 60 |
| **ROME** | 0.21 | 0.08 | 0.00 | 75 s | 480 |
| **MEMIT** | 0.33 | 0.17 | 0.67 | 91 s | 2400 |
| **INLAY (this)** | **1.00** | 0.25 | **1.00** | **0.5 s** | **0** |

## What this benchmark shows — the honest reading

**INLAY holds up on its core promise even on the competitors' home turf.** All 24 facts recalled
(efficacy 1.00), zero collateral damage (locality 1.00), written gradient-free in half a second.
That's the same profile it had on the single-subject set — the advantage did *not* depend on the
favorable case.

**ROME and MEMIT still underperformed here, and I want to be careful about why.** Two honest reasons:

1. **Sequential editing on a small model is hard.** We applied 24 edits *sequentially* to GPT-2-XL.
   ROME is fundamentally a *single*-edit method — 24 sequential single-layer edits accumulate
   interference, so it collapsed again (locality 0). MEMIT, the mass-edit method, did meaningfully
   better (33% efficacy, locality 0.67) but still modest.

2. **This is not the CounterFact leaderboard.** Published MEMIT reports high scores editing
   *thousands* of facts — but on the CounterFact/zsRE benchmarks, with those datasets' own prompt
   phrasings and their scoring (probability-margin, not strict substring greedy). My test uses
   **fabricated facts, strict substring-match greedy decoding, and the 1.5B GPT-2-XL** — a harder,
   different measurement. **These numbers are not comparable to published MEMIT results** and should
   not be read as "INLAY beats MEMIT in general." They show how the methods behave *on this specific,
   controlled setup*.

**Fine-tuning is the interesting contrast:** it got the **best generalization (0.83)** — because it
actually changes the weights, so paraphrases benefit — but at the cost of **total locality collapse
(0.00)**: 60 steps on 24 sentences overwrote everything else. That's the efficacy/locality tradeoff
in its starkest form.

## INLAY's real, unchanged weakness: generalization (0.25)

INLAY recalls facts perfectly on the exact prompt but only 1-in-4 paraphrases. This is the honest
limitation, and it's structural: INLAY keys on the raw hidden state of the *written* phrasing, so a
reworded question lands in the wrong slot. Fine-tuning (0.83) and RAG (0.54) both generalize better.
The fix is a learned/normalized key encoder so semantically-equivalent questions map to the same
slot — the clear next step if we push INLAY further.

## The fair conclusion

On this controlled multi-subject benchmark, INLAY gives the best **combined** profile (efficacy +
locality + write cost) of all six methods, and it does so gradient-free with frozen weights. But it
is **not** "state of the art" — it has not been run on the standard large-scale benchmarks
(CounterFact/zsRE) where that title is decided, and it generalizes to paraphrases worse than
fine-tuning or RAG. It is a promising, genuinely different approach with a clear strength (fast,
reversible, non-destructive writes) and a clear weakness (paraphrase addressing).

## Reproducibility
`bench_multi.py <model> <inlay_layer> <inlay_alpha> <n_sub>` runs base/RAG/fine-tune/INLAY;
`bench_multi_edit.py ROME|MEMIT` runs the EasyEdit edits. Fact set in `facts24.json`. Raw numbers in
`multi_subject.json` / `.csv`.
