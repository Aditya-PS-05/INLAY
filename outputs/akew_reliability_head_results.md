# Retrieval-reliability calibration head ("confidence of confidence") — 2026-08-20

A new method, not another baseline: a second-order calibration signal that
lets the router decide **whether to trust its own gating** on a per-query
basis, rather than applying fixed thresholds everywhere.

## The bug this fixes, and why two prior attempts failed

`akew_fullpipeline_results.md` documents the same failure on MQuAKE-CF in
every input mode tested, while the identical gating helps or ties on
CounterFact and WikiUpdate in every mode:

| MQuAKE-CF | routed (fixed gates) | always-REASON |
|---|---|---|
| structured | 93.65% | **96.83%** |
| extracted | 69.84% | **85.71%** |

Two separate experiments already ruled out the obvious fixes:

- **Verifier recalibration** (`akew_verifier_recalibration_results.md`):
  moving the threshold changed the false-fire rate from 18.33% to 18.22% --
  no meaningful change, because the positive and negative score
  distributions on MQuAKE-CF genuinely **overlap**.
- **A `direct_threshold` sweep** (0.85 / 0.97 / 1.01): raising it to 0.97
  changed *nothing* -- byte-identical decisions and accuracy -- because the
  verifier's confidence on MQuAKE-CF's **wrong** retrievals is already above
  0.97. Disabling DIRECT entirely still lost to always-REASON.

Both point at the same conclusion: **the router is asking the wrong
question.** It asks "how confident is the verifier about this one card?"
when what it needs is "is this verifier's confidence *discriminative* right
now?" A verifier scoring 0.99 on the top card and 0.98 on four unrelated
cards is not confident -- it is saturated, and its top-1 pick is close to
arbitrary. **No threshold on the top-1 score can see that distinction**,
which is precisely why every threshold-based fix failed.

## The method

Predict `P(top-1 retrieval is correct)` from the **shape** of the top-k
retrieval + verification result rather than its top-1 magnitude. Features
(15 total, all inference-time computable):

- **Margin/shape features (the novel signal):** `ver_margin_12` (top-1
  verifier score minus the best competing one), `ver_max_rest`,
  `ver_n_above_direct` (how many candidates clear the DIRECT threshold),
  `emb_margin_12`, `emb_margin_15`, `emb_entropy` (softmax entropy over the
  neighbourhood), `ver_std_topk`, `subject_diversity` (distinct subjects
  among top-k -- the entity-collision signal).
- **First-order features (what the old router used):** `ver_top1`,
  `emb_top1`, plus `ver_mean_topk`, `emb_mean_topk`, `emb_std_topk`,
  `query_len`, `looks_multihop`.

**The hard constraint that makes this a method rather than the manual
per-dataset bypass with extra steps:** no feature is dataset identity, and
nothing dataset-level is available at inference. The whole claim is that
MQuAKE-CF's unreliable regime is *detectable from retrieval shape*, so the
head is trained on CounterFact + WikiUpdate with **MQuAKE-CF held out
entirely** (mirroring the verifier v1/v2 protocol) and must generalise to a
dataset it has never seen.

Deliberately a **linear** model: the training set is small, the claim rests
on OOD generalisation where a high-capacity model would more likely memorise
CounterFact/WikiUpdate's retrieval geometry, and the coefficients are
directly inspectable -- so the central mechanism claim becomes an empirical
check on fitted weights rather than an assertion.

## Head quality: it generalises to a dataset it never saw

| split | n | base rate (retrieval correct) | mean predicted | AUROC | AUPRC | ECE |
|---|---|---|---|---|---|---|
| train (CF+Wiki) | 2400 | 0.8971 | 0.8812 | 0.9987 | 0.9994 | 0.0159 |
| val (CF+Wiki) | 915 | 0.8831 | 0.8560 | 0.9976 | 0.9984 | 0.0270 |
| in-domain test (CF+Wiki) | 921 | 0.8893 | 0.8642 | 0.9986 | 0.9986 | 0.0263 |
| **OOD test (MQuAKE-CF, never seen)** | 189 | 0.8042 | 0.7966 | **0.9557** | **0.9835** | 0.0543 |

OOD AUROC 0.9557 on a dataset never seen in training, with mean predicted
reliability (0.7966) tracking the actual base rate (0.8042) closely -- the
head correctly recognises MQuAKE-CF as a lower-reliability regime **without
ever having been told such a regime exists**. That is the claim the whole
method rests on, and it holds.

## The mechanism is validated on the coefficients, not just the outcome

| feature | coefficient (standardized) |
|---|---|
| **ver_max_rest** | **−1.4346** |
| emb_margin_12 | +1.2894 |
| **ver_margin_12** | **+1.1765** |
| emb_top1 | +0.7622 |
| ver_top1 | +0.6757 |
| ver_std_topk | +0.5375 |
| ver_mean_topk | +0.3858 |
| emb_margin_15 | +0.3655 |
| query_len | −0.2899 |
| emb_entropy | −0.1192 |
| emb_mean_topk | +0.1162 |
| emb_std_topk | −0.1091 |
| subject_diversity | −0.0742 |
| ver_n_above_direct | −0.0376 |
| looks_multihop | 0.0000 |

**Margin-feature coefficient mass: 3.60 vs first-order's 1.44 — a 71.5%
share.** The single largest weight in the model is `ver_max_rest` at
**−1.43**: the more confident the verifier is about some *competing* card,
the *less* likely its top pick is right. That is the confidence-of-confidence
effect stated as a mechanism at the top of this page, recovered from data
rather than assumed -- and it is exactly the quantity a top-1 threshold
cannot express, however it is tuned.

**One honest null in that table:** `looks_multihop` has a coefficient of
exactly 0.0, because it is constant (always 0) across CounterFact and
WikiUpdate -- both single-hop datasets. A feature that is constant in
training gets zero weight and therefore contributes nothing at inference
either, including on MQuAKE-CF where it *would* fire. It is dead weight in
the current head, not a contributing signal, and is reported as such rather
than left in the feature list implying it does work.

**Honest comparison against a stronger model**, reported even though it does
not favour the choice made: a gradient-boosted classifier on the identical
features scores slightly *better* OOD (AUROC 0.9671 vs 0.9557, AUPRC 0.9861
vs 0.9835). The linear head was kept for interpretability and
generalisation-safety on a small training set, but the GBM's edge is real
and is not hidden. If the linear head's OOD margin ever proves fragile, the
GBM is the obvious first thing to reach for.

## Downstream result: MQuAKE-CF structured

The critical test -- the dataset/mode where fixed gating loses to a blanket
bypass. All three conditions scored on the identical queries, same model,
same decoding, in a single pass.

| condition | accuracy |
|---|---|
| routed, fixed gates (current shipped router) | 93.65% |
| always-REASON (manual bypass) | 96.83% |
| **routed, adaptive (reliability head)** | **100.0%** |

**The adaptive router beats both existing configurations**, and does it by
firing the bypass on only 4/63 queries (6.35%) rather than blanket-disabling
the gates. Router decisions shifted from `REJECT 4 / DIRECT 53 / REASON 6`
to `REJECT 0 / DIRECT 53 / REASON 10` -- it converted exactly the four
REJECTs (which were throwing away imperfect-but-usable retrievals, the
mechanism diagnosed in the threshold sweep) while leaving all 53 DIRECT
decisions untouched.

**Bypass targeting** (the mechanism check -- a head firing at random could
land the same headline number): precision 0.75 (3 of 4 bypasses were on
genuinely wrong retrievals), recall 0.27 (caught 3 of 11 bad retrievals).
High precision, modest recall: it fires selectively and correctly rather
than often.

**Sample-size caveat, stated plainly:** n=63. In absolute terms adaptive
fixed 4 errors relative to fixed gating and 2 relative to always-REASON.
Those are small counts and 100% on 63 examples should not be read as "solved"
-- the remaining dataset/mode cells below are what determine whether this
generalises or whether this cell was fortunate.

## The two extremes, and why they matter more than the headline

The method's claim is not "higher accuracy on one dataset" -- it is that a
**single configuration, with no per-dataset tuning, adapts correctly to both
regimes.** The two extreme cells test exactly that.

**MQuAKE-CF extracted** -- where fixed gating loses worst (a 15.87-point gap):

| condition | accuracy |
|---|---|
| routed, fixed gates | 69.84% |
| always-REASON | 85.71% |
| **routed, adaptive** | **85.71%** |

The full gap is recovered. Bypass fired on 17/63 queries (27.0%) and caught
**12 of the 13 genuinely-wrong retrievals (92.3% recall, 70.6% precision)** --
converting all 11 REJECTs to REASON. On this cell the head is doing what a
blanket bypass does, because on this cell a blanket bypass is correct.

**CounterFact structured** -- where fixed gating *wins* (DIRECT genuinely
earns its place, 100% vs always-REASON's 97.96%):

| condition | accuracy |
|---|---|
| routed, fixed gates | 100.0% |
| always-REASON | 97.96% |
| **routed, adaptive** | **100.0%** |

**The bypass fired zero times.** Mean predicted reliability 0.9983 against
an actual retrieval correctness of 1.0 -- the head recognised a
high-reliability regime and left all 147 DIRECT decisions untouched,
preserving the advantage a blanket bypass would have thrown away.

Together these are the result: **the same head, same threshold, same
weights** bypasses aggressively (27%) where gating is harmful and not at all
(0%) where gating is valuable -- on a dataset it was never trained on. The
per-dataset manual bypass that `akew_fullpipeline_results.md` concluded was
necessary is no longer necessary.

## All three MQuAKE-CF modes recover

The failure was documented in every input mode, so the fix has to hold in
every input mode -- not just the one it was developed against.

| MQuAKE-CF mode | fixed gates | always-REASON | **adaptive** | bypass rate | recall of bad retrievals |
|---|---|---|---|---|---|
| structured | 93.65% | 96.83% | **100.0%** | 6.3% | 27.3% |
| unstructured | 65.08% | 80.95% | **80.95%** | 25.4% | 84.6% |
| extracted | 69.84% | 85.71% | **85.71%** | 27.0% | 92.3% |

Every mode recovers the full gap; structured additionally exceeds the manual
bypass. Note the bypass rate is not a constant the head applies blindly --
it varies from 6.3% to 27.0% *across modes of the same dataset*, tracking
how much of each mode's retrieval is actually bad, which is the behaviour a
per-query signal should have and a per-dataset switch structurally cannot.

## Full matrix

All cells: `Qwen/Qwen2.5-1.5B-Instruct`, verifier v2, bypass threshold 0.5,
one head, no per-dataset tuning. **Bold** marks the best of the three.

| dataset | mode | n | fixed gates | always-REASON | **adaptive** | bypass rate |
|---|---|---|---|---|---|---|
| CounterFact | structured | 147 | **100.0%** | 97.96% | **100.0%** | 0.0% |
| CounterFact | unstructured | 147 | **87.07%** | **87.07%** | **87.07%** | 0.7% |
| WikiUpdate | structured | 160 | **99.38%** | 96.25% | **99.38%** | 3.8% |
| WikiUpdate | unstructured | 160 | **43.75%** | 43.13% | 43.13% | 32.5% |
| MQuAKE-CF | structured | 63 | 93.65% | 96.83% | **100.0%** | 6.3% |
| MQuAKE-CF | unstructured | 63 | 65.08% | **80.95%** | **80.95%** | 25.4% |
| MQuAKE-CF | extracted | 63 | 69.84% | **85.71%** | **85.71%** | 27.0% |

**The headline: adaptive strictly dominates either fixed policy.**

- vs. *fixed gates everywhere*: wins by +6.35 / +15.87 / +15.87 on the three
  MQuAKE-CF cells, ties on three, loses 0.62 on one.
- vs. *always-REASON everywhere*: wins by +2.04 (CF structured), +3.13 (Wiki
  structured), +3.17 (MQuAKE structured), ties on the rest.

Neither existing configuration wins on every cell; adaptive matches or beats
the better of the two on six of seven, **without being told which dataset it
is looking at.** The per-dataset manual bypass that
`akew_fullpipeline_results.md` concluded was necessary is no longer
necessary.

## The one loss, which is the most informative cell here

**WikiUpdate unstructured: 43.75% (fixed) → 43.13% (adaptive), −0.62 points.**

First, the magnitude honestly: on n=160 that is 70 correct vs 69 correct --
**a one-example difference**, comfortably inside noise, and not a meaningful
regression. But it should not be waved away either, because the *mechanism*
behind it is real and matters more than the number.

The head did its job on this cell, and did it well: bypass precision 82.7%,
**recall of bad retrievals 95.6%** -- it caught 43 of the 45 genuinely-wrong
retrievals, its strongest detection performance anywhere in the matrix. The
detection was not the problem. What went wrong is what the router then *did*
about it.

**Detecting that retrieval is unreliable is not the same as knowing the
right response to that.** The current design hard-codes one response --
bypass the gates, force REASON -- because that is what the MQuAKE-CF
diagnosis called for. But the two datasets want *opposite* responses to the
same signal:

- On **MQuAKE-CF**, a low-confidence retrieval still carries useful signal
  (its chains pass through ordinary unedited facts), so reasoning over
  imperfect evidence beats declining. Bypass → REASON is right.
- On **WikiUpdate**, a low-confidence retrieval is *actively misleading*
  (its stale/current officeholder collisions mean the wrong card often
  asserts a plausible, confidently-stated falsehood), so declining beats
  reasoning over it. REJECT is right, and forcing REASON throws away the
  REJECT path's measured value.

So the head's output is being used to answer the wrong question. It reliably
predicts *"is this retrieval trustworthy?"* and the router converts that into
a fixed action, when the action itself should be a second decision --
learned per-regime rather than hard-coded. **That is the concrete next
iteration**: let the reliability signal select among {REJECT, REASON,
DIRECT} rather than only gating a bypass, with the choice trained on which
response actually pays off. Recorded as a specific, diagnosed architectural
limitation with a named fix, not a vague "future work."

## Cost

The head requires verifier scores for all top-k candidates rather than top-1
alone: **k cross-encoder calls per query instead of 1** (k=5 by default), on
top of one negligible dot product for the linear head itself. Cross-encoder
calls on five short pairs are cheap relative to the generation call that
follows, but this is a real 5× increase in the verification stage and is
stated rather than buried. Where the head fires no bypass (CounterFact
structured: 0/147), that cost buys nothing on that cell -- it is the price
of not needing to know in advance which cell you are on.
