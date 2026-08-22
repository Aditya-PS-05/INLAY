# INLAY vs ROME, MEMIT, fine-tuning, and in-context — measured

All six methods on **the same model (GPT-2-XL, 1.5B)**, same 5 fabricated facts (efficacy)
and 12 general-knowledge control prompts (locality), on an NVIDIA L40S.
ROME/MEMIT run via EasyEdit (the reference implementation).

| method | efficacy | locality | write cost | gradients | what it changes |
|---|---|---|---|---|---|
| Base GPT-2-XL | 0/5 | 1.00 | — | 0 | nothing |
| In-context (RAG) | 5/5 | 1.00 | 0 *(re-fed every query)* | 0 | nothing |
| Fine-tune (60 steps) | 5/5 | **0.25** | 34.7 s | 60 | all 1.5B params |
| **ROME** | 1/5 | **0.00** | 15 s | 20 | rank-1, 1 MLP layer |
| **MEMIT** | 2/5 | 0.92 | 10,729 s* | 100 | 5 MLP layers (13–17) |
| **INLAY (this)** | **5/5** | **1.00** | **0.15 s** | **0** | external table (weights frozen) |

\*MEMIT's write time is dominated by a **one-time** Wikipedia covariance precompute (~3 h for
5 layers, cached to disk); the edit itself is seconds. All other write times are per-edit.

**Efficacy** = facts answered correctly (greedy). **Locality** = unrelated control prompts whose
answer is unchanged from base (1.0 = no collateral damage).

## What the numbers say

- **INLAY and in-context (RAG) are the only two methods in the ideal top-right corner** — full
  efficacy AND full locality. INLAY gets there with a 0.15 s gradient-free write and frozen weights;
  RAG gets there by re-feeding the document on every query (cost grows with the corpus).

- **ROME collapsed (1/5, locality 0).** This is the important, honest finding: **all 5 facts
  describe the same subject** ("Zorvax reactor"), and ROME keys its edit on the subject's
  last-token representation. Editing 5 attributes of one subject sequentially makes each edit
  overwrite the same location — they collide, and the model degenerates to repeating one answer
  ("Rurik Tolan") for every prompt, which also wrecks unrelated knowledge. ROME is designed for
  editing *many different* subjects; this benchmark is its adversarial case.

- **MEMIT held locality (11/12) but only reached 2/5 efficacy** — it spreads the update across 5
  layers with a covariance correction, which protects unrelated knowledge far better than ROME or
  fine-tuning, but the same-subject collision still limits how many of the 5 attributes it can
  install cleanly. It recalled "helium" and "Rurik Tolan"; the rest came out mangled.

- **Fine-tuning learned all 5 facts but corrupted 75% of unrelated knowledge** (locality 0.25) — the
  catastrophic-forgetting failure mode, worse on gpt2-xl than gpt2 because 60 full-model steps on 5
  sentences overfits hard.

## The fair reading

This benchmark is deliberately the **hardest case for subject-keyed weight editors** (many
attributes of one entity) and the **natural case for addressable memory** (one slot per fact). INLAY
wins here because it keys on the full context, not the subject token, so 5 facts about one entity
land in 5 disjoint slots with no collision.

The honest flip side, unchanged from before: INLAY's advantage narrows when facts span *many*
subjects (ROME/MEMIT's home turf, where published MEMIT scales to thousands of edits), and INLAY
still addresses paraphrased queries imperfectly (2/5). A complete evaluation would add a
multi-subject fact set — that's the benchmark where ROME/MEMIT are expected to close the gap.

## Reproducibility

`compare_methods.py <model> <inlay_layer> <inlay_alpha> <n_sub>` runs base/RAG/fine-tune/INLAY and
prints locality 1.00 for INLAY using the min_score=0.9 firing gate (the code reproduces the reported
number). `run_rome_memit.py ROME|MEMIT` runs the EasyEdit edits. Raw numbers in
`comparison_full.json`.
