# CounterFact benchmark — the standard knowledge-editing test (GPT-2-XL, N=100)

You asked to settle "is CAKE state of the art?" by running the benchmark where that title is
actually decided. Here it is: **CounterFact**, single-edit protocol, 100 records, on GPT-2-XL, scored
with **EasyEdit's own native metric** (teacher-forcing token accuracy) so every method is measured
identically. ROME/MEMIT run through the reference EasyEdit implementation.

- **ES (efficacy)** — the edited fact is produced on the edit prompt
- **PS (generalization)** — the fact is produced on a *paraphrase* prompt
- **NS (locality/specificity)** — a neighborhood prompt (same relation, different subject) is
  *unchanged* vs the pre-edit model
- **Score** — harmonic mean of ES, PS, NS (the ROME paper's summary number)

| method | ES (efficacy) | PS (generalization) | NS (locality) | **score** | write cost |
|---|---|---|---|---|---|
| Base GPT-2-XL | 0.01 | 0.00 | 1.00 | 0.000 | — |
| In-context (RAG) | 0.92 | 0.41 | 0.44 | 0.515 | 0 *(re-fed each query)* |
| Fine-tune (25 steps) | 0.63 | 0.40 | 0.02 | 0.050 | 6.6 s |
| **ROME** | **0.95** | 0.43 | **0.88** | **0.665** | 11.9 s |
| MEMIT* | 0.00 | 0.00 | 0.99 | 0.000 | 12.7 s |
| **CAKE (this)** | **1.00** | 0.15 | 0.69 | 0.329 | 2.7 s (100 facts) |

\*MEMIT single-edit rewrite_acc=0 under EasyEdit's default gpt2-xl config. Confirmed both in
isolation and across N=100. This is not a CAKE win — see below.

## The straight answer to "is CAKE state of the art?" — No.

**On CounterFact, the benchmark that decides it, ROME wins (0.665).** CAKE scores 0.329 — less than
half. This is the honest, direct result you asked for, and it's the opposite of what the earlier
multi-subject test suggested. The reason the earlier benchmark favored CAKE was that it used
*sequential* editing of same/related facts — the adversarial case for ROME. CounterFact's
**single-edit** protocol is ROME's designed use case, and ROME performs almost exactly as its paper
reports: near-perfect efficacy (0.95), strong locality (0.88).

## What CAKE does well, and where it loses

- **CAKE has perfect efficacy (1.00)** — its addressing + logit injection installs the counterfactual
  every time, and its write is ~4.4× faster than ROME (2.7 s vs 11.9 s for the full N=100), gradient-free.
- **CAKE loses on the two axes that matter for a real editor:**
  - **Generalization PS = 0.15** — the decisive weakness. CAKE keys on the *exact* prompt's hidden
    state, so CounterFact's paraphrases (deliberately hard, e.g. prefixed with distractor sentences)
    mostly miss the slot. ROME (0.43) and even fine-tuning (0.40) generalize far better because they
    change the model's computation, not just add a lookup entry.
  - **Locality NS = 0.69** — CAKE's firing gate sometimes triggers on neighborhood prompts (same
    relation, different subject) whose hidden state is close to the edited one, injecting the wrong
    fact. ROME's NS is 0.88.

## Why MEMIT reads 0.00 here (and why that is NOT a CAKE win)

MEMIT's default EasyEdit gpt2-xl config spreads each update across **5 layers** with a covariance
correction — machinery designed for editing **many facts at once**. For a *single* fact that update
is too diffuse to flip the token, so single-edit rewrite_acc is 0 (confirmed both in an isolated
one-fact test and across the N=100 run; ROME, with the identical single-edit call, scored 0.955
aggregate). **MEMIT's fair result is the batch benchmark**, where I already measured it as the best
weight-editor (efficacy 0.33, locality 0.67 on the 24-fact multi-subject set). Reporting MEMIT's
single-edit 0 as a CAKE advantage would be misleading — it's a protocol mismatch, not a quality gap.

## Fine-tuning and RAG, for context

- **Fine-tune** got decent efficacy (0.63) but **locality collapsed to 0.02** — 25 gradient steps on
  one sentence overwrites unrelated knowledge. The catastrophic-forgetting failure mode, in full.
- **RAG** is the strongest non-weight method (0.515): good efficacy, but its NS drops to 0.44 because
  prepending the fact-context also perturbs neighborhood predictions, and its cost recurs on every
  query.

## Bottom line

Across all three benchmarks now run — single-subject, multi-subject batch, and CounterFact —
the honest picture is consistent:

- **CAKE's genuine strengths:** fastest writes (gradient-free, 2.7 s for 100 facts, ~4.4× faster than ROME), perfect efficacy, zero
  weight modification (reversible, non-destructive), best combined score when edits are sequential
  and non-adversarial.
- **CAKE's genuine weaknesses:** poor paraphrase generalization (0.15 on CounterFact) and imperfect
  locality — both stemming from keying on raw hidden states.
- **CAKE is not state of the art.** On CounterFact, ROME clearly wins. CAKE is a promising,
  genuinely different mechanism (addressable external memory rather than weight surgery) with a clear
  niche, not a replacement for the established editors on their own benchmark.

The single highest-leverage fix remains a **learned/normalized key encoder** so paraphrases address
the same slot — that PS=0.15 is what's dragging CAKE's score down, and it's addressable.

## Reproducibility
`eval_cf.py <method> <model> <N> [layer alpha]` (base/RAG/CAKE/finetune) and
`eval_cf_edit.py ROME|MEMIT <N>` (EasyEdit). CounterFact from rome.baulab.info. Raw numbers in
`counterfact.json` / `.csv`.
