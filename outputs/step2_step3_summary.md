# INLAY vs the sequential/lifelong editors, and the portability weakness

Two questions remained for the "state of the art" claim after the GPT-J-6B port: (step 2) how does
INLAY compare to the *sequential/lifelong* editors WISE and GRACE — its real conceptual rivals — and
(step 3) does INLAY have a genuine weakness on the one axis where a retrieval method should be weak,
**portability** (does an edit propagate to related facts rather than being blindly played back)?

Both were run on GPT-J-6B. The answer to step 2 is that INLAY still leads. The answer to step 3 is
**yes, INLAY has a real weakness** — it over-fires on same-subject/different-relation queries — and a
relation-aware gate *partially* fixes it at a measured cost.

## Step 2 — WISE and GRACE (CounterFact single-edit, GPT-J-6B, N=100)

| method | ES | PS | NS | **score** | write |
|---|---|---|---|---|---|
| Base | 0.02 | 0.00 | 1.00 | 0.000 | — |
| RAG (in-context) | 0.84 | 0.42 | 0.45 | 0.515 | 0 |
| GRACE* | 0.00 | 0.00 | 1.00 | 0.000 | 17.7 s |
| WISE | 1.00 | 0.385 | 1.00 | 0.653 | 15.1 s |
| ROME | 0.995 | 0.755 | 0.60 | 0.751 | 8.6 s |
| **INLAY v3** | **1.00** | **0.85** | **0.828** | **0.886** | ~0 |

**Ranking: INLAY 0.886 > ROME 0.751 > WISE 0.653 > RAG 0.515 > Base/GRACE.**

- **WISE** has the profile its design predicts: perfect efficacy and perfect locality (its side-memory
  is non-destructive, like INLAY) but weak paraphrase generalization (PS 0.385) — it stores the exact
  edit and doesn't generalize to reworded questions. INLAY's semantic key gives it much better PS (0.85).
- **GRACE** shows ES 0.0 on CounterFact single-edit. GRACE is a codebook that only intervenes when a
  runtime activation lands within `eps` of a stored key; under EasyEdit's default GPT-J config that gate
  does not activate on CounterFact prompts. It is also scored by EasyEdit with a *generation* token-match
  metric rather than the teacher-forcing metric used for the others, so the 0.0 is not strictly
  comparable — it means "GRACE did not install the edit under this configuration," a protocol note, not
  a INLAY win.

INLAY beats the sequential-editing rivals on this benchmark. But WISE/GRACE are built for *long streams*
of edits — the fair comparison there is the sequential stress test already in this project (where INLAY
held retention flat to 400 edits while ROME collapsed by ~50).

## Step 3 — the portability weakness (and a partial fix)

**The probe.** Standard ES/PS/NS never test whether an edit *propagates* to a logically-related but
surface-different query. I built 30 compositional probes (LLM-generated from CounterFact edits): for each
edit, a **same-relation reworded** query (correct answer = target_new) and a **same-subject
different-relation** query (correct answer is NOT target_new — emitting target_new here is a failure).

**The weakness (measured):**

| method | propagation (want high) | over-fire (want low) |
|---|---|---|
| Base | 0.03 | 0.00 |
| RAG | 0.63 | 0.00 |
| **INLAY (rel_gate 0)** | **1.00** | **0.97** |

INLAY propagates perfectly but **over-fires 97% of the time** — ask "Singled Out was hosted by ___" after
editing its debut network to CBS, and INLAY plays back "CBS". RAG doesn't (the model still reasons about
which attribute is asked). **This is the structural signature of retrieval-not-rederivation** and is
INLAY's genuine remaining weakness — exactly where a stored-answer method is expected to be weak.

**Root cause (measured).** INLAY's gate keys on a MiniLM embedding of the whole prompt, which is dominated
by the **subject**. Gate cosine for genuine "same-relation" probes (mean 0.816) overlaps heavily with
"different-relation same-subject" probes (mean 0.749, max 0.956) — no threshold on the prompt embedding
separates them.

**The fix — a relation gate.** The subject is known at *write* time (you are writing a specific fact about
a specific entity), so store the prompt and subject embeddings in the slot. At read time, form the query's
**relation-residual** (query embedding with the slot's stored subject direction projected out) and compare
it to the slot's relation-residual; fire only if it clears `rel_gate`. No read-time NER, still
zero-gradient. This raises the same/different-relation separation from 0.07 (full embedding) to 0.18
(relation-residual).

**Result — a genuine partial fix with a measured cost:**

| rel_gate | comp over-fire | comp propagation | CounterFact PS |
|---|---|---|---|
| 0.0 (off) | 0.97 | 1.00 | 0.87 |
| **0.2** | **0.67** | **0.97** | **0.66** |
| 0.3 | 0.53 | 0.83 | 0.49 |
| 0.4 | 0.37 | 0.77 | (not measured) |

(CounterFact PS was swept 0.0–0.3, decreasing monotonically 0.87→0.82→0.76→0.72→0.66→0.60→0.49; the
comp over-fire/propagation sweep extends to 0.5. The CF-PS cost at rel_gate>0.3 was not run.)

At the chosen operating point **rel_gate 0.2**, over-firing drops from 0.97 to 0.67 while compositional
propagation stays high (0.97). But CounterFact paraphrase generalization falls from 0.87 to 0.66
(full held-out CF: score 0.886 → 0.806).

**Why it's a trade, not a free win.** A reworded paraphrase and a different-relation query are *both* moves
away from the exact written prompt. Any gate that rejects the latter also partly rejects the former. The
relation gate improves the balance but cannot eliminate the tension — it is a real, honestly-bounded
mitigation. rel_gate defaults to 0 (off); turn it up when specificity against same-subject queries matters
more than paraphrase coverage.

## Bottom line for "state of the art"

- INLAY leads every method measured on single-edit CounterFact at both model sizes, **including the
  sequential-editing rivals WISE and GRACE.**
- INLAY has a **real, characterized weakness**: it over-fires on same-subject/different-relation queries
  (retrieval, not re-derivation). The relation gate cuts this by a third at a paraphrase-coverage cost.
- The remaining honest gates: full-benchmark N (vs 100), a native RippleEdits/portability dataset (vs
  LLM-generated probes), and AlphaEdit. INLAY is the strongest method *on the axes measured here*, with a
  named weakness and a partial fix — not an unqualified SOTA claim.

## Files
- `eval_cf_wise_grace.py` (WISE/GRACE), `eval_comp_portability.py` (compositional probe, inlay/base/RAG),
  `eval_relgate_sweep.py` (relation-gate sweep), `gpt2_memory_semkey.py` (v5, relation gate added)
- `gptj_all_methods.{json,png}`, `portability_weakness.{json,png}`, `comp_probes.json`
