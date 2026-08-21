# There is no routing headroom on AKEW — and why that matters more than the router

**2026-08-21. This is the most consequential result in the project and it
requires revising the central claim of `akew_reliability_head_results.md`.**

## What was measured

For every query in a 1,689-row sample spanning all three datasets and all
three input modes, each candidate router action was **actually executed and
scored**: `REJECT` (answer with no context), `REASON` (answer over the
retrieved card), and `DIRECT` (recite the stored fact, structured mode only).
That yields, per query, the ground truth of which actions would have produced
a correct answer — the thing a router is trying to guess.

## The result

| cell | n | static policy | oracle | headroom |
|---|---|---|---|---|
| CounterFact/structured | 250 | 1.0000 | 1.0000 | **+0.0000** |
| CounterFact/unstructured | 250 | 0.9040 | 0.9040 | **+0.0000** |
| CounterFact/extracted | 250 | 0.8400 | 0.8400 | **+0.0000** |
| WikiUpdate/structured | 250 | 0.9960 | 0.9960 | **+0.0000** |
| WikiUpdate/unstructured | 250 | 0.4280 | 0.4280 | **+0.0000** |
| WikiUpdate/extracted | 250 | 0.4800 | 0.4800 | **+0.0000** |
| MQuAKE-CF/structured | 63 | 1.0000 | 1.0000 | **+0.0000** |
| MQuAKE-CF/unstructured | 63 | 0.8095 | 0.8095 | **+0.0000** |
| MQuAKE-CF/extracted | 63 | 0.8571 | 0.8571 | **+0.0000** |
| **POOLED** | **1689** | **0.7874** | **0.7874** | **+0.0000** |

"Static policy" is one line of code with no model, no training and no
inference cost: **DIRECT where it is legal, otherwise REASON.** "Oracle" is a
router that picks the best available action on every single query — an upper
bound no real system can exceed.

**They are identical, to four decimal places, in every cell.**

**The maximum possible gain of ANY per-query routing method over a fixed
one-line policy, on this entire benchmark suite, is 0.00 points.**

## The mechanism: REJECT is never uniquely correct

| REJECT success rate | structured | unstructured | extracted |
|---|---|---|---|
| CounterFact | 0.0% | 0.4% | 0.8% |
| WikiUpdate | 2.0% | 2.0% | 2.4% |
| MQuAKE-CF | 0.0% | 0.0% | 0.0% |

Checked directly against the raw labels rather than inferred: REJECT succeeded
on **19 of 1,689** queries, and on **all 19** of those, REASON or DIRECT
succeeded too. **REJECT is the sole winning action zero times.** (The labels
are not stuck at zero — both classes are present, which is why the 19 exist.)

The reason is structural and, in hindsight, obvious. These are
**counterfactual** edit benchmarks. The eval question asks for the *post-edit*
answer. Answering from parametric knowledge alone returns the *pre-edit*
value — which is, by construction, the wrong answer. **Abstention cannot be
correct on a benchmark where every question targets an edited fact.**

## What this does to the reliability head's claims

**What survives, unchanged.** The measured gains are real: adaptive routing
beat fixed gating by +15.9 points on two MQuAKE-CF cells (p=0.0019,
Holm-corrected), with byte-identical decisions where fixed gating already
worked. Those numbers were correctly measured and stand.

**What was wrong: the explanation.** The write-up framed the head as *adapting
its decision per query to the local regime*. It does not, because there is no
per-query decision worth making. What the head actually does is **suppress the
REJECT and DIRECT gates in the regimes where they misfire** — that is, it
converges toward the static policy. Its gains are the size of the damage the
fixed gates were doing, not the size of any insight it adds.

That reframing is unflattering and it is the correct one. A single line —
"never REJECT; DIRECT only in structured mode" — matches the head's
performance everywhere, with no training data, no 5x cross-encoder cost, and
no OOD generalisation argument required.

The head's other properties remain factually true and now look beside the
point: 0.956 OOD AUROC at predicting retrieval correctness, 71.5% of
coefficient mass on margin features. It is a good predictor of a quantity that
turns out not to drive the decision.

## The finding underneath, which is the actual contribution

The benchmark suite contains **no out-of-scope queries**. Every evaluation
question targets a fact that was edited. There is no query for which "this
edit does not apply; answer from what you already know" is correct.

This has a consequence beyond this project. **Scope classification is a
load-bearing component of the SERAC-descendant design family** — the entire
premise is a learned decision about whether a stored edit applies to the
incoming query. But on AKEW-style benchmarks that decision is degenerate:
the answer is always "yes, it applies." A benchmark with no negatives cannot
measure a classifier's ability to reject, and any REJECT path evaluated on it
can only ever lose points.

So the field's standard benchmarks **structurally cannot reward the abstention
machinery that its standard architectures are built around.** Reported routing
gains on these benchmarks are, at best, measurements of how much harm a gate
was doing.

## The experiment this demands

The missing condition is constructible. Ask eval questions against an index
that does **not** contain the corresponding edit — a query whose correct
behaviour genuinely is "no relevant edit exists, answer from base knowledge."
On such queries REJECT becomes the correct action, the oracle should pull
away from the static policy, and routing should acquire real headroom for the
first time.

That measures the thing every scope-classifier paper claims to measure, and
this suite currently cannot. Building it next.

## Reproducing

```
python3 src/akew_outcome_labels.py <dataset> <mode> <limit>   # per-action ground truth
python3 src/akew_headroom.py "outputs/outcome_labels_*.json"  # the table above
```
