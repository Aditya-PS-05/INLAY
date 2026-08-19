# Verifier threshold recalibration test — 2026-08-19

Tests whether the verifier's MQuAKE-CF miscalibration (`akew_verifier_results.md`:
false-fire rate 1.5% on val -> 17.85% zero-shot on MQuAKE-CF) is a cheap
threshold-selection problem, or a genuine model-quality gap requiring
retraining. Split MQuAKE-CF's own cards subject-disjointly into a small
calibration slice (20%, 1,215 rows) and a held-out test slice (80%, 5,325
rows), picked a new F1-maximizing threshold on the calibration slice only,
and compared it against the original val-calibrated threshold on the SAME
held-out test slice.

| | threshold | precision | recall | F1 | false-fire rate |
|---|---|---|---|---|---|
| before (CounterFact+WikiUpdate threshold) | 0.65 | 0.5767 | 0.9991 | 0.7313 | 18.33% |
| after (MQuAKE-CF-recalibrated threshold) | 0.70 | 0.5783 | 0.9991 | 0.7325 | 18.22% |

**Result: recalibration barely moves anything.** False-fire rate improves by
0.11 percentage points -- noise, not a fix. This decisively rules out the
"just recalibrate the threshold per dataset" hypothesis from
`akew_verifier_results.md`.

## Why: this is a distribution-overlap problem, not a threshold problem

Recall stays essentially saturated (99.91%) regardless of which threshold is
picked -- true positives score high no matter what. What's stuck is
precision, at ~58% across the entire threshold sweep the calibration slice
tested. That specific pattern (recall saturated, precision stuck regardless
of threshold) means the negative examples' score distribution genuinely
overlaps with the positive examples' score distribution on MQuAKE-CF -- no
single scalar cutoff can separate two distributions that overlap that much,
which is exactly why moving the threshold from 0.65 to 0.70 changed almost
nothing. This is consistent with the Stage 1 retrieval finding
(`akew_schema_validation.md`): MQuAKE-CF's entities and relations recur far
more than CounterFact's or WikiUpdate's, so many genuinely-wrong candidates
look nearly as relevant as the genuinely-right one under the verifier's
current features.

## What this means for the fix

The verifier's training data (CounterFact + WikiUpdate hard negatives) simply
does not contain the specific kind of confusion MQuAKE-CF creates --
same-relation, entity-dense collisions from a multi-hop chain structure that
neither training dataset has. **The fix is retraining with MQuAKE-CF-shaped
hard negatives added to the mix**, not a cheaper recalibration step. This is
the harder of the two branches flagged in `akew_verifier_results.md`, now
confirmed necessary by directly ruling out the easier one with a controlled
experiment, rather than assuming it and skipping straight to a retrain.

Not yet done: retraining the verifier with a MQuAKE-CF-shaped hard-negative
slice included, and rerunning the zero-shot-style test to see whether the
false-fire rate genuinely drops or whether the multi-hop structure resists
even a targeted retrain.
