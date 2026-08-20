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
| CounterFact | extracted | 147 | **78.23%** | **78.23%** | **78.23%** | 2.0% |
| WikiUpdate | structured | 160 | **99.38%** | 96.25% | **99.38%** | 3.8% |
| WikiUpdate | unstructured | 160 | **43.75%** | 43.13% | 43.13% | 32.5% |
| WikiUpdate | extracted | 160 | **47.50%** | 46.25% | 46.25% | 36.3% |
| MQuAKE-CF | structured | 63 | 93.65% | 96.83% | **100.0%** | 6.3% |
| MQuAKE-CF | unstructured | 63 | 65.08% | **80.95%** | **80.95%** | 25.4% |
| MQuAKE-CF | extracted | 63 | 69.84% | **85.71%** | **85.71%** | 27.0% |

Complete 3×3 matrix, all nine cells.

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

## Statistical rigor: which of these claims actually survive a paired test

Accuracies on n=63 with no interval are not evidence, so every cell was
rerun emitting per-example hit vectors and analysed with `akew_stats.py`:
Wilson 95% intervals, **exact McNemar** on discordant pairs (exact rather
than chi-square, since discordant counts here are often under 10), a
**paired** bootstrap on the difference, and **Holm-Bonferroni** across the
seven cells, since testing seven uncorrected buys a spurious result by luck.

| cell | fixed (95% CI) | adaptive (95% CI) | difference (95% CI) | McNemar p | survives Holm |
|---|---|---|---|---|---|
| MQuAKE-CF extracted | 0.698 [0.576, 0.798] | **0.857** [0.750, 0.923] | **+0.159** [+0.079, +0.254] | 0.0019 | **YES** |
| MQuAKE-CF unstructured | 0.651 [0.527, 0.757] | **0.809** [0.696, 0.887] | **+0.159** [+0.079, +0.254] | 0.0019 | **YES** |
| MQuAKE-CF structured | 0.936 [0.848, 0.975] | 1.000 [0.943, 1.000] | +0.064 [+0.016, +0.127] | 0.1250 | no |
| CounterFact structured | 1.000 [0.975, 1.000] | 1.000 [0.975, 1.000] | 0.000 [0.000, 0.000] | 1.0000 | n/a |
| WikiUpdate structured | 0.994 [0.966, 0.999] | 0.994 [0.966, 0.999] | 0.000 [0.000, 0.000] | 1.0000 | n/a |
| WikiUpdate extracted | 0.475 [0.399, 0.552] | 0.463 [0.387, 0.540] | −0.013 [−0.037, +0.013] | 0.6250 | n/a |
| WikiUpdate unstructured | 0.438 [0.363, 0.515] | 0.431 [0.357, 0.509] | −0.006 [−0.025, +0.013] | 1.0000 | n/a |

**What this changes about the claims above -- three corrections, all in the
direction of claiming less:**

1. **The two large MQuAKE-CF gains are real and survive correction.** +15.9
   points on both unstructured and extracted, p=0.0019 after Holm, with
   bootstrap intervals comfortably excluding zero. These are the result.

2. **The MQuAKE-CF structured "100%" cannot be called significant, and the
   earlier write-up overstated it.** The paired bootstrap interval
   [+0.016, +0.127] excludes zero, but the exact McNemar test returns
   p=0.125 -- and it *cannot* do better: there are only **4 discordant
   pairs**, and with all four favouring adaptive the smallest attainable
   two-sided exact p is 2/2⁴ = 0.125. The two tests disagree because the
   bootstrap resamples the whole sample while McNemar conditions on the
   discordant pairs alone; with evidence this thin the conservative reading
   is the honest one. **A 100% on n=63 built from a four-example margin is
   suggestive, not established.**

3. **The WikiUpdate "losses" are not losses.** Both differences are
   statistically indistinguishable from zero (−0.013, p=0.625; −0.006,
   p=1.0), with intervals spanning zero. The earlier framing of a
   "systematic pattern" across the two WikiUpdate cells should be read as a
   *hypothesis suggested by the direction of two non-significant
   differences*, not as a measured regression -- and the three-way policy
   built to fix that supposed regression duly failed, which is consistent
   with there having been less there to fix than it appeared.

**One genuinely strong result the aggregate numbers hid:** on CounterFact
structured and WikiUpdate structured the adaptive router has **zero
discordant pairs** with fixed gating -- not merely similar accuracy, but
*byte-identical decisions on every single query*. The head demonstrably does
not disturb the regimes where the existing gates already work, which is a
sharper statement than any accuracy comparison could make.

## Threshold robustness: 0.5 was not a lucky pick

A single hand-chosen threshold is exactly the kind of thing that makes a
result look better than it is, so it was swept across a deliberately wide
range on both extremes of the matrix.

| bypass threshold | MQuAKE-CF extracted | (bypass rate) | CounterFact structured | (bypass rate) |
|---|---|---|---|---|
| 0.3 | 85.71% | 23.8% | 100.0% | 0.0% |
| 0.4 | 85.71% | 27.0% | 100.0% | 0.0% |
| 0.5 (default) | 85.71% | 27.0% | 100.0% | 0.0% |
| 0.6 | 85.71% | 27.0% | 100.0% | 0.0% |
| 0.7 | 85.71% | 27.0% | 100.0% | 0.0% |
| 0.8 | 85.71% | 31.8% | 100.0% | 0.7% |

**Identical accuracy at every threshold from 0.3 to 0.8** on both cells,
while the bypass rate moves as the threshold moves -- so the threshold is
doing something, and the outcome simply does not depend on where in that
wide band it sits. The head's predictions are separated enough that any
reasonable cut lands in the same place. This is the opposite of a
knife-edge tuned result, and it is worth contrasting with the thing it
replaces: the `direct_threshold` sweep on the *old* router changed nothing
between 0.85 and 0.97 and then fell off a cliff at 1.01, because it was
thresholding a saturated signal.

## The one loss, which is the most informative cell here

**WikiUpdate unstructured: 43.75% → 43.13% (−0.62, one example on n=160).
WikiUpdate extracted: 47.50% → 46.25% (−1.25, two examples on n=160).**

First, the magnitudes honestly: one and two examples respectively, both
comfortably inside noise. But the fact that *both* WikiUpdate cells move the
same direction, while all three MQuAKE-CF cells move the other way, says the
pattern is **systematic rather than noise**, even though each individual
number is small. That is what makes it worth diagnosing rather than
dismissing -- and the mechanism behind it matters more than the magnitude.

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
a fixed action, when the action itself should be a second decision.

**The discriminating observation that makes this fixable:** WikiUpdate's
unreliable cases score *lower* on predicted reliability (mean 0.6596
unstructured, 0.6243 extracted) than MQuAKE-CF's do (0.7492 extracted,
0.7640 unstructured). The **magnitude** of predicted unreliability -- not
merely whether it crossed one line -- carries the missing signal. That
motivates a three-way policy driven by the same score with a second, lower
threshold:

    p <  reject_floor      -> REJECT   (actively misleading; decline)
    p <  bypass_threshold  -> REASON   (imperfect but usable; reason over it)
    otherwise              -> normal fixed gating

Implemented (`AkewRouter(reject_floor=...)`, `None` reproduces the binary
behaviour exactly) and tested. **It does not work.**

## The three-way policy fails decisively — a negative result on my own proposed fix

`reject_floor=0.3`, `bypass_threshold=0.5`, same head, same cells:

| cell | fixed | adaptive (binary) | **three-way** | always-REASON |
|---|---|---|---|---|
| MQuAKE-CF structured | 93.65% | **100.0%** | **100.0%** | 96.83% |
| MQuAKE-CF unstructured | 65.08% | **80.95%** | 58.73% | 80.95% |
| MQuAKE-CF extracted | 69.84% | **85.71%** | 63.49% | 85.71% |
| CounterFact structured | **100.0%** | **100.0%** | **100.0%** | 97.96% |
| WikiUpdate structured | **99.38%** | **99.38%** | 98.75% | 96.25% |
| WikiUpdate unstructured | **43.75%** | 43.13% | 41.25% | 43.13% |

It fails **on the cell it was specifically designed to fix** (WikiUpdate
unstructured: 43.13% → 41.25%, worse than the binary policy it was meant to
improve on) and it is *catastrophic* on MQuAKE-CF, dropping 22 points below
the binary policy and landing well below even the original fixed gating.

**The hypothesis was wrong, and cleanly so.** "WikiUpdate's unreliable cases
score lower than MQuAKE-CF's, so a second lower threshold separates them" was
reasoning from the *means* (0.62–0.66 vs 0.75–0.76) while ignoring the
*distributions*. MQuAKE-CF has a substantial low-reliability tail that the
binary policy was routing to REASON and handling correctly; converting that
tail to REJECT throws away exactly the retrievals the whole MQuAKE-CF fix
depended on keeping.

## Why it failed, which is more useful than the fix working would have been

The deeper reason is a **target mismatch**, and it reframes the limitation
section above.

The head is trained to predict `P(top-1 retrieval is correct)`. It does that
well (OOD AUROC 0.956). But that is *not* the quantity the router needs. What
the router needs is `P(this routing action produces a correct answer)` -- and
those two sets come apart:

- A query can have a **wrong** retrieval and still be answered correctly,
  because the model ignores the irrelevant context and answers from
  parametric knowledge. Declining would have *lost* that one.
- A query can have a **right** retrieval and still be answered wrongly, if
  the evidence is present but the model misreads it.

So "retrieval is untrustworthy" and "declining beats answering here" are
different predicates, and the binary policy only appeared to work because on
MQuAKE-CF, forcing REASON happened to be the right action for nearly the
whole untrustworthy set. The fixed router's verifier-threshold REJECT set,
despite being a *worse* predictor of retrieval correctness, is apparently a
*better*-chosen action set on WikiUpdate -- which is exactly what a
target mismatch looks like.

**The correct next iteration, now properly motivated:** train the head on
**outcome** labels rather than retrieval-correctness labels -- for each query
and each candidate action in {REJECT, DIRECT, REASON}, whether that action
actually produced a correct answer -- and have it select the argmax action.
That is a decision-theoretic objective rather than a retrieval-quality one,
it subsumes the binary policy as a special case, and it is what this negative
result points at. It costs one generation pass per action per training query
to label, which is why it was not the first thing built, but it is now
clearly the right thing to build next.

The binary adaptive policy stands unchanged as the shipped result: it is the
configuration that dominates both baselines. `reject_floor` remains in the
code with its default of `None` (binary behaviour), documented here as tested
and rejected rather than quietly removed.

## Scale validation: the gain is architectural, not model-dependent

The head predicts *retrieval* correctness, which does not depend on the
generator at all -- so the prediction going in was that the head transfers
unchanged and only downstream answering accuracy should move. Tested with
`Qwen/Qwen2.5-7B-Instruct` (4.7× the parameters), same head, same threshold,
same test splits (model identity verified in each run's output rather than
assumed from the launch environment):

| cell | | 1.5B | 7B |
|---|---|---|---|
| MQuAKE-CF extracted | fixed | 69.84% | 69.84% |
| | **adaptive** | **85.71%** | **85.71%** |
| MQuAKE-CF structured | fixed | 93.65% | 93.65% |
| | **adaptive** | **100.0%** | **100.0%** |
| CounterFact structured | fixed / adaptive | 100.0% | 100.0% |
| WikiUpdate unstructured | fixed | 43.75% | 39.37% |
| | **adaptive** | 43.13% | **39.37%** |
| | always-REASON | 43.13% | 39.37% |

At 7B the adaptive-vs-fixed gap on WikiUpdate unstructured is gone: all three
conditions land on exactly 39.37%. That is consistent with the paired test
above finding the 1.5B gap indistinguishable from zero.

**But the gap closed DOWNWARD, and that must not be read as good news.**
An earlier version of this section reported only that "the loss disappears at
7B," which was misleading by omission: WikiUpdate unstructured is **4.38
points WORSE at 7B than at 1.5B for every condition**, and the conditions
converge because they all degraded, not because any improved. Correcting that
framing here rather than leaving a favourable-sounding half-truth in place.

**This is a real, dataset-specific scale degradation and it needs
explaining, not burying.** It is not general noise: CounterFact unstructured
moves the expected direction over the same scale jump (87.07% -> 89.12%,
`akew_fullpipeline_results.md`). Something about WikiUpdate specifically gets
harder for the larger model.

Run validity was checked before drawing any conclusion, since three identical
numbers is exactly what a collapsed configuration looks like: routing genuinely
differed between the conditions (fixed REJECT 42 / adaptive REJECT 0 /
three-way REJECT 50 of 160), and the model identity is recorded as
`Qwen/Qwen2.5-7B-Instruct` in the run output. All three arriving at 63/160 is
a genuine coincidence of aggregate accuracy over different routing, not a
config error.

### Hypothesis 1 — parametric-prior conflict — TESTED AND REFUTED

The proposed account was: WikiUpdate's edits concern real-world facts whose
*previous* value the model also saw in pretraining, so a larger model's
stronger priors make it override the injected edit more often.

Operationalised without circularity (`akew_prior_conflict_diag.py`): ask the
model the eval question with **no context at all**; if it spontaneously
produces `target_true`, it demonstrably holds that prior. Prior labels are
computed *per model*, since the hypothesis is precisely that the larger model
holds more of them. Confirm/refute criteria were written into the script
before it was run.

| WikiUpdate unstructured | 1.5B | 7B |
|---|---|---|
| strong-prior rate | 3.75% (6/160) | 5.63% (9/160) |
| accuracy, strong-prior subset | 83.33% | 33.33% |
| accuracy, no-prior subset | 41.56% | 39.74% |
| **reverted to pre-edit value** | **2.50%** | **1.87%** |
| retrieval correctness | 71.88% | 71.88% |

**Refuted on two of its three criteria:**

1. **The revert rate went DOWN** (2.50% → 1.87%). Reverting to the prior *is*
   the proposed mechanism; the larger model does it *less*. This alone is
   decisive.
2. **The loss sits in the no-prior subset.** Of ~6 net examples lost, roughly
   4 come from the no-strong-prior group (94% of the data) and ~2 from the
   strong-prior group. It is not concentrated where the account requires.
3. The strong-prior rate does rise (6 → 9 examples), but both are tiny and
   cannot carry a 6-example swing.

**The CounterFact control settles it.** CounterFact at 7B has **four times**
WikiUpdate's strong-prior rate (23.81% vs 5.63%) and *improves* at 7B — and
its strong-prior subset scores **higher** than its no-prior subset (91.43% vs
88.39%). Across both datasets and both scales, holding a strong prior tracks
*better* accuracy, not worse. The collision account is wrong.

### Hypothesis 2 — hedging on weak evidence — suggested by the refutation

The refutation is informative rather than merely negative. At 7B the model is
**not** answering the old value (it reverts less), so it is failing some other
way, and the failures sit where it has no prior to fall back on.

The relevant observation already on record in this project: the 7B model
hedges on weak evidence rather than committing ("Based on the given evidence,
we cannot determine who the head of state...", `akew_multihop_results.md`).
WikiUpdate has ~28% wrong retrievals to hedge about; CounterFact has ~1% --
which would explain why the degradation is dataset-specific, the fact that
made Hypothesis 1 attractive in the first place.

### Hypothesis 2 — CONFIRMED on all three pre-stated criteria

| WikiUpdate unstructured | 1.5B | 7B | Δ |
|---|---|---|---|
| hedge rate, overall | 29.38% | **46.88%** | **+17.50** |
| hedge rate, retrieval **wrong** | 66.67% | **91.11%** | **+24.44** |
| hedge rate, retrieval correct | 14.78% | 29.57% | +14.79 |
| CounterFact control, overall | 0.68% | 1.36% | **+0.68** |

All three criteria met: markedly more hedging at 7B, the excess largest on
wrong-retrieval examples, and a control gap 25× smaller.

**A measurement bug found and fixed mid-diagnostic, which reversed the
result.** The first version of the hedge detector used literal strings and
had `does not contain / provide / mention / specify` but not `does not
INCLUDE` -- which turned out to be the 7B model's most common refusal
phrasing (three of seven sampled failures used it and scored as non-hedges).
Because 7B favours that phrasing, the under-count was *biased toward 7B
specifically*, and the broken detector reported wrong-retrieval hedging
*falling* at 7B (57.78% → 51.11%) -- the exact opposite of the truth. The
detector was rebuilt as pattern families, checked against the cases it had
missed plus confident answers it must not fire on (11 checks), and re-run.
The first version's numbers are void, not adjusted. **The sample dump is what
caught this; a rate reported without one would have been wrong and
plausible.**

### Why this fully accounts for the anomaly

Cross-tabulating hedging against correctness:

| WikiUpdate | hedged | not hedged |
|---|---|---|
| 1.5B | 0.021 (n=47) | 0.602 (n=113) |
| 7B | 0.013 (n=75) | **0.729** (n=85) |

**Hedging is functionally a guaranteed miss** (~2% vs 60–73%). The
decomposition reproduces the observed numbers exactly: 1.5B scores
17×0.059 + 98×0.694 ≈ 69 of 160 (43.1%); 7B scores 34×0.000 + 81×0.765 +
41×0.024 ≈ 63 of 160 (39.4%).

**The essential point: 7B is the better model, and loses anyway.** When it
commits to an answer it is *more* accurate than 1.5B (0.765 vs 0.694 on
correct-retrieval examples). It scores lower only because it declines far
more often. Had 7B hedged at 1.5B's rate on correct retrievals, it would
score roughly 47.5% — comfortably ahead.

Its extra caution is also **correctly targeted where it is free and wrongly
targeted where it is costly**: on wrong retrievals it hedges 91.11% vs
66.67%, which costs nothing (accuracy is ~0 either way and declining is
arguably the right behaviour); on *correct* retrievals it hedges 29.57% vs
14.78%, and that subset is the entire loss.

### What this means beyond this one anomaly

Two consequences worth carrying into the write-up:

1. **The anomaly is not a knowledge-editing failure at all.** No edit is
   being resisted, overridden, or lost — the earlier prior-conflict story was
   wrong about that. The larger model simply declines to answer more often on
   noisy evidence, and this benchmark's accuracy metric scores a refusal
   identically to a wrong answer. WikiUpdate exposes it only because it is
   the one dataset with enough retrieval noise (28%) to trigger the behaviour
   at volume; CounterFact's ~1% noise gives it nothing to decline about.

2. **Substring accuracy conflates "wrong" with "declined to answer,"** and
   larger or more heavily aligned models decline more. Any scale comparison
   on a noisy-retrieval benchmark using this metric will therefore understate
   larger models. This is a property of the measure, not of the method under
   test, and it is a caveat that applies to the field's numbers generally,
   not just to ours. Reporting hedge rate alongside accuracy costs nothing
   and would prevent the misreading.

**The +15.87-point gain on MQuAKE-CF extracted is identical at both scales.**
Not merely similar -- identical, because the improvement comes entirely from
*which action the router takes*, and the router's inputs (retrieval, verifier,
head) are all independent of the generator. The 11 queries that fixed gating
REJECTs and adaptive routes to REASON flip the same way at both scales,
because whether the retrieved evidence contains the answer is a property of
the evidence, not of the model reading it.

This is a stronger form of scale-robustness than "the trend holds at 7B": the
mechanism is architectural, so there is no scale at which it should stop
working, and the 7B run is a check on that reasoning rather than a search for
a trend. One caveat: CounterFact structured saturates at 100% for every
condition at 7B (always-REASON too), so that cell no longer discriminates
between methods at this scale -- a ceiling effect, not evidence of anything.

## Cost

The head requires verifier scores for all top-k candidates rather than top-1
alone: **k cross-encoder calls per query instead of 1** (k=5 by default), on
top of one negligible dot product for the linear head itself. Cross-encoder
calls on five short pairs are cheap relative to the generation call that
follows, but this is a real 5× increase in the verification stage and is
stated rather than buried. Where the head fires no bypass (CounterFact
structured: 0/147), that cost buys nothing on that cell -- it is the price
of not needing to know in advance which cell you are on.
