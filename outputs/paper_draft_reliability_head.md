# Knowing When Not To Trust Yourself: Second-Order Confidence Calibration for Retrieval-Based Knowledge Editing

**Draft — workshop paper (4–8 pages). Status: experiments complete, prose draft.**

---

## Abstract

Retrieval-based knowledge editing routes a query to one of several answering
strategies based on a verifier's confidence that a retrieved fact is
relevant. We show this design has a failure mode that no threshold on that
confidence can fix: on datasets with high entity collision, the verifier is
*confidently wrong* — its score distributions for correct and incorrect
retrievals overlap, so the top-1 score carries no usable signal about whether
to trust it. We demonstrate this concretely: recalibrating the threshold
moves the false-fire rate by 0.1 points, and raising it from 0.85 to 0.97
leaves every routing decision byte-identical.

We propose predicting retrieval reliability from the *shape* of the top-k
retrieval and verification result rather than its top-1 magnitude — a
second-order signal ("is this confidence discriminative?") that a first-order
threshold structurally cannot express. A 15-feature linear head trained on
two datasets and evaluated on a **held-out third it never saw** reaches 0.956
AUROC, and 71.5% of its fitted coefficient mass sits on margin features, with
the single largest weight being negative on the best *competing* candidate's
score — the mechanism recovered from data rather than assumed.

Used to gate the router adaptively, one configuration with no per-dataset
tuning strictly dominates both fixed policies across a 3×3 dataset × input-mode
matrix: +15.9 points (p=0.0019, Holm-corrected) where fixed gating fails,
while making *byte-identical decisions* where it already works. The gain is
identical at 1.5B and 7B parameters, because it is architectural. We also
report a refinement we proposed and then refuted, which diagnoses the method's
real limitation: the head predicts retrieval correctness, but the router needs
to know which *action* pays off, and those are different predicates.

---

## 1. Introduction

**The setting.** Knowledge editing asks how to update a fact in a language
model without retraining it. Weight-editing methods (ROME, MEMIT, AlphaEdit)
modify parameters directly; retrieval-based methods keep the model frozen and
supply the new fact as context at inference. The latter avoids catastrophic
interference and makes edits revertible and auditable, at the cost of a new
failure surface: the system must decide, per query, whether it has retrieved
something worth trusting.

**The standard design and its assumption.** A typical router scores the
top-1 retrieved candidate with a cross-encoder verifier and thresholds that
score: below one threshold, decline to answer (REJECT); above another,
recite the stored fact directly (DIRECT); otherwise reason over the
retrieved evidence (REASON). This encodes an assumption that is rarely
stated: **that the verifier's confidence is informative about whether it is
right.**

**Where the assumption breaks.** We find this assumption fails on datasets
with high entity collision. On MQuAKE-CF, gating is *net-negative in every
input mode we tested* — the routed pipeline underperforms simply disabling
the gates and always reasoning, by up to 15.9 points. Two natural fixes both
fail, and their failure is diagnostic:

- Recalibrating the verifier threshold moves the false-fire rate from 18.33%
  to 18.22% — no meaningful change, because the positive and negative score
  distributions genuinely overlap.
- Raising the DIRECT threshold from 0.85 to 0.97 changes *nothing*: identical
  decisions, identical accuracy. The verifier's confidence on the *wrong*
  retrievals is already above 0.97.

The router is asking the wrong question. It asks *"how confident is the
verifier about this candidate?"* when what it needs is *"is this confidence
discriminative right now?"* A verifier scoring 0.99 on the top candidate and
0.98 on four unrelated ones is not confident; it is saturated, and its top
pick is close to arbitrary. **No threshold on a top-1 score can see that
distinction, which is precisely why every threshold-based fix failed.**

**Contributions.**

1. We characterise a failure mode of confidence-thresholded routing and show
   empirically that it is not fixable by recalibration (§3).
2. We propose a second-order reliability signal computed from the *shape* of
   the top-k retrieval/verification result, and show it generalises to a
   held-out dataset (0.956 AUROC) with the mechanism validated on the fitted
   coefficients rather than asserted (§4).
3. We show an adaptively gated router strictly dominates both fixed policies
   across a 3×3 matrix, with statistically significant gains where gating
   fails and provably identical behaviour where it works (§5).
4. We report a refinement we proposed, tested, and refuted, and use its
   failure to diagnose the approach's real limitation and the next step (§6).

---

## 2. Related Work

*(To be expanded — placeholder structure with the honest positioning.)*

**Weight editing.** ROME and MEMIT locate and modify factual associations in
MLP layers; AlphaEdit constrains updates to a null space to reduce
interference. These operate on a clean `(prompt, target)` pair and have no
mechanism for unstructured evidence — a structural limitation rather than a
tuning gap, which we quantify in §5.

**Retrieval-based and in-context editing.** SERAC introduces a scope
classifier deciding whether an edit applies; IKE demonstrates that few-shot
override demonstrations teach a model to prefer injected facts over
parametric belief. **We build directly on SERAC's scope-classifier idea and
IKE's demonstration mechanism; neither is claimed as novel here.** Our
contribution is orthogonal: a signal for when the scope decision itself
should be trusted.

**Calibration and selective prediction.** Selective prediction studies when a
model should abstain, typically from its own output distribution. We apply a
related idea one level up — calibrating a *routing component's* reliability
rather than a predictor's — and find the standard first-order approach
insufficient for the reason above.

**Benchmarks.** We evaluate on AKEW, which supplies three input conditions
(structured triples, unstructured evidence prose, LLM-extracted triples)
across CounterFact, WikiUpdate, and MQuAKE-CF, allowing us to separate
retrieval-regime effects from input-format effects.

---

## 3. The Failure Mode

**Setup.** Router with REJECT/DIRECT/REASON gates on cross-encoder verifier
confidence; dense retrieval over the full card pool; subject-disjoint splits.

**Observation.** Gating helps on CounterFact and WikiUpdate but is
net-negative on MQuAKE-CF in every input mode:

| MQuAKE-CF | fixed gates | always-REASON | gap |
|---|---|---|---|
| structured | 93.65% | 96.83% | −3.18 |
| unstructured | 65.08% | 80.95% | −15.87 |
| extracted | 69.84% | 85.71% | −15.87 |

**Why thresholds cannot fix it.** Table of the threshold sweep (0.85 / 0.97 /
1.01) showing identical decisions at 0.85 and 0.97, plus the recalibration
result (18.33% → 18.22%). Both are consistent with overlapping score
distributions rather than a misplaced cut point.

**Diagnosis.** Retrieval correctness on MQuAKE-CF is 79–83%, versus 99% on
CounterFact — and the verifier assigns high confidence to the wrong
retrievals as readily as the right ones. What distinguishes the two cases is
not the top-1 score but whether that score *stands out* from its competitors.

---

## 4. Method: A Retrieval-Reliability Head

**Target.** Predict `P(top-1 retrieval is correct)` per query.

**Features (15).** Computed from the top-k (k=5) retrieval and verification
result. The novel group is second-order:

- `ver_margin_12` — top-1 verifier score minus the best competitor's
- `ver_max_rest`, `ver_n_above_direct`, `ver_std_topk`
- `emb_margin_12`, `emb_margin_15`, `emb_entropy` (softmax entropy over the
  neighbourhood), `subject_diversity` (distinct subjects among top-k)

plus first-order/contextual: `ver_top1`, `emb_top1`, `ver_mean_topk`,
`emb_mean_topk`, `emb_std_topk`, `query_len`, `looks_multihop`.

**The design constraint that makes this a method rather than a per-dataset
switch:** no feature is dataset identity, and no dataset-level statistic is
available at inference. The claim is that the unreliable regime is
*detectable from retrieval shape*; the head is therefore trained on
CounterFact + WikiUpdate with **MQuAKE-CF held out entirely** and must
generalise to a dataset it has never seen.

**Model.** Logistic regression with standardisation. Chosen over a
higher-capacity model deliberately: the training set is small, the claim
rests on OOD generalisation where capacity invites memorising the training
datasets' retrieval geometry, and the coefficients are inspectable so the
mechanism claim is checkable. *We report that a gradient-boosted model on the
same features scores slightly better OOD (0.967 vs 0.956 AUROC) — the linear
head's advantage is interpretability and safety, not fit.*

**Results.**

| split | n | base rate | AUROC | AUPRC | ECE |
|---|---|---|---|---|---|
| in-domain test | 921 | 0.889 | 0.9986 | 0.9986 | 0.026 |
| **OOD (MQuAKE-CF, unseen)** | 189 | 0.804 | **0.9557** | 0.9835 | 0.054 |

**Mechanism validated on the weights.** Margin-feature coefficient mass is
3.60 versus 1.44 for first-order features — a **71.5% share**. The single
largest weight is `ver_max_rest` at **−1.43**: the more confident the
verifier is about a *competing* candidate, the less likely its top pick is
correct. This is the second-order effect stated in §3, recovered from data.

*Honest null:* `looks_multihop` receives exactly 0.0 weight because it is
constant across both training datasets (both single-hop) — dead weight in the
current head, reported rather than left implying it contributes.

---

## 5. Adaptive Routing

**Policy.** If predicted reliability < τ (0.5), bypass both gates and force
REASON; otherwise apply the existing gates unchanged.

**Main result.** Complete 3×3 matrix, one head, one threshold, no per-dataset
tuning. Adaptive matches or beats the better of the two fixed policies in six
of seven evaluated cells.

| | fixed | always-REASON | **adaptive** |
|---|---|---|---|
| MQuAKE-CF extracted | 0.698 | 0.857 | **0.857** |
| MQuAKE-CF unstructured | 0.651 | 0.810 | **0.810** |
| MQuAKE-CF structured | 0.936 | 0.968 | **1.000** |
| CounterFact structured | **1.000** | 0.980 | **1.000** |
| WikiUpdate structured | **0.994** | 0.963 | **0.994** |
| WikiUpdate unstructured | 0.438 | 0.431 | 0.431 |
| WikiUpdate extracted | 0.475 | 0.463 | 0.463 |

**Statistics.** Exact McNemar on discordant pairs (exact, not chi-square —
discordant counts are often <10), paired bootstrap on the difference, Wilson
intervals, Holm-Bonferroni across cells:

- MQuAKE-CF unstructured and extracted: **+15.9 points each, p=0.0019,
  survive Holm correction.**
- MQuAKE-CF structured (+6.4): **not significant.** Only 4 discordant pairs;
  the smallest attainable two-sided exact p is 2/2⁴ = 0.125. Suggestive only.
- WikiUpdate differences: **not distinguishable from zero** (p=0.625, p=1.0).
- CounterFact structured and WikiUpdate structured: **zero discordant
  pairs** — the adaptive router makes identical decisions on every query,
  a stronger statement than matched accuracy.

**Behavioural adaptation.** The bypass fires at 27.0% on MQuAKE-CF extracted
and **0.0%** on CounterFact structured — same head, same threshold. It also
varies *across modes of the same dataset* (6.3% to 27.0% within MQuAKE-CF),
tracking how much of each mode's retrieval is actually bad, which a
per-dataset switch structurally cannot do.

**Threshold robustness.** Accuracy is identical for every τ from 0.3 to 0.8
on both extremes, while the bypass rate moves with τ — the result does not
sit on a knife edge.

**Scale.** The +15.87-point gain on MQuAKE-CF extracted is *identical* at
1.5B and 7B. The router's inputs are generator-independent, so the mechanism
is architectural; the 7B run tests that reasoning rather than searching for a
trend.

**A scale anomaly, diagnosed.** On WikiUpdate unstructured every condition is
4.38 points *worse* at 7B than at 1.5B (43.75% → 39.37% for fixed gating),
while CounterFact unstructured improves over the same jump (87.07% → 89.12%).
Our head neither causes nor addresses this — it is present in the fixed
baseline — but leaving it unexplained invites a wrong reading, so we
diagnosed it (§6.1).

**Cost.** k cross-encoder calls per query instead of 1 (k=5), plus a
negligible dot product. Cheap relative to the generation call that follows,
but a real 5× increase in the verification stage.

---

## 6.1 The Scale Anomaly: Refusal, Not Edit Failure

The 7B degradation on WikiUpdate invites the obvious knowledge-editing
reading — that a larger model's stronger parametric priors resist the
injected edit. **We tested that and refuted it.** Operationalising "holds a
strong prior" as spontaneously producing the pre-edit value with no context,
labelled per model: the rate of reverting to the pre-edit value *fell* at 7B
(2.50% → 1.87%), the loss sat in the *no*-prior subset, and the CounterFact
control has 4× the strong-prior rate while improving. Across both datasets
and scales, holding a prior tracks *better* accuracy.

The actual cause is refusal. At 7B the model declines to commit far more
often on noisy evidence:

| WikiUpdate | 1.5B | 7B |
|---|---|---|
| refusal rate, overall | 29.38% | **46.88%** |
| refusal rate, retrieval wrong | 66.67% | **91.11%** |
| refusal rate, retrieval correct | 14.78% | 29.57% |
| CounterFact control | 0.68% | 1.36% |

Refusal is functionally a guaranteed miss under substring accuracy (2% vs
60–73% when the model commits), and the decomposition reproduces both
observed accuracies exactly. **The larger model is more accurate whenever it
commits** (0.765 vs 0.694 on correct-retrieval examples); it scores lower
purely because it declines more. At 1.5B's refusal rate it would score ~47.5%.
Its extra caution is well-placed where retrieval is wrong (free — accuracy is
~0 either way) and costly where retrieval is correct, which is the entire gap.

Two implications extend beyond this cell. First, the anomaly is **not an
editing failure**: no edit is resisted or lost. WikiUpdate exposes it only
because it is the one dataset with enough retrieval noise (28%) to trigger
refusal at volume. Second, **substring accuracy conflates "wrong" with
"declined,"** and larger or more heavily aligned models decline more — so any
scale comparison on a noisy-retrieval benchmark using this metric will
understate larger models. This is a property of the measure rather than of any
method under test, and applies to the field's numbers generally. Reporting
refusal rate alongside accuracy is nearly free and prevents the misreading.

*Methodological note:* our first refusal detector used literal strings and
omitted "does not include" — the 7B model's most common refusal phrasing.
Because the omission was model-specific, it under-counted 7B and reported the
wrong-retrieval trend *backwards*. Inspecting sampled outputs caught it; the
detector was rebuilt as pattern families and validated against the missed
cases plus confident answers it must not fire on. We report this because a
refusal rate published without an audited sample would have been wrong and
entirely plausible.

## 6.2 A Refinement We Refuted

Observing that WikiUpdate's unreliable cases score lower on predicted
reliability (mean 0.62–0.66) than MQuAKE-CF's (0.75–0.76), we hypothesised
that the *magnitude* of predicted unreliability distinguishes "misleading,
decline" from "imperfect, reason over it," and implemented a three-way policy
with a second lower threshold.

**It failed decisively** — on the cell it was designed to fix (43.13% →
41.25%) and catastrophically on MQuAKE-CF (80.95% → 58.73%, below even fixed
gating). The hypothesis reasoned from means while ignoring distributions:
MQuAKE-CF has a substantial low-reliability tail that the binary policy was
correctly routing to REASON.

**The failure is more informative than success would have been.** It exposes
a **target mismatch**. The head predicts `P(retrieval correct)`, but the
router needs `P(this action yields a correct answer)`, and those predicates
come apart: a wrong retrieval can still be answered correctly from parametric
knowledge (declining would lose it), and a right retrieval can be misread.
Notably, the fixed router's verifier-threshold REJECT set — a *worse*
predictor of retrieval correctness — is a *better*-chosen action set on
WikiUpdate. That is what a target mismatch looks like.

**Next step, now properly motivated:** train on **outcome** labels — for each
query and each action in {REJECT, DIRECT, REASON}, whether that action
produced a correct answer — and select the argmax. This is decision-theoretic
rather than retrieval-quality-based, subsumes the binary policy as a special
case, and costs one generation pass per action per training query to label.

---

## 7. Limitations

- **Small cells.** MQuAKE-CF slices are n=63; the structured-mode result
  cannot reach significance and is reported as suggestive only.
- **One retrieval stack.** Single encoder (MiniLM) and one cross-encoder
  verifier; sensitivity to those choices is untested.
- **Two training datasets.** OOD generalisation is demonstrated on one
  held-out dataset, not many.
- **Binary policy only.** The three-way extension failed; the outcome-labelled
  version is proposed, not built.
- **Not a weight-editing replacement.** On clean structured single-fact edits,
  weight editors remain competitive; our claim concerns robustness across
  input conditions and retrieval regimes, not single-edit accuracy.

## 8. Conclusion

Confidence-thresholded routing fails when confidence is uninformative, and no
recalibration of that confidence can detect the failure. Measuring whether
confidence is *discriminative* — from the shape of the candidate set rather
than the magnitude of the top score — supplies the missing signal, transfers
to an unseen dataset, and yields a single configuration that dominates both
fixed policies without per-dataset tuning.

---

## Appendix / Reproducibility

- Code: `akew_reliability.py`, `akew_reliability_train.py`,
  `akew_adaptive_router_eval.py`, `akew_stats.py` (+ 55 self-tests across
  `akew_reliability_test.py` and `akew_stats_test.py`).
- Splits: subject-disjoint, seed 0, held-out dataset never touched in training.
- Full experimental records: `akew_reliability_head_results.md`,
  `akew_fullpipeline_results.md`, `akew_weightedit_baseline_results.md`.
