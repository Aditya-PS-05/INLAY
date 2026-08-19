# Verifier v2: retrained with MQuAKE-CF hard negatives — 2026-08-19

Retrained the verifier with a small MQuAKE-CF hard-negative slice (1,215
rows, the same subject-disjoint 20% calibration split used in the
recalibration experiment) added to the CounterFact+WikiUpdate training mix.
Evaluated on the SAME held-out 80% MQuAKE-CF test slice (5,325 rows) used
throughout, for a fair three-way comparison against v1 zero-shot and v1
threshold-recalibrated.

| | false-fire rate | recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| v1, zero-shot (never saw MQuAKE-CF) | 17.85% | 99.9% | 0.7365 | 0.9235 | 0.6083 |
| v1, threshold-recalibrated only | 18.22% | 99.9% | 0.7325 | 0.9193 | 0.5940 |
| v2, retrained + MQuAKE hard negs, at v1's threshold (0.65) | 17.93% | 99.7% | 0.7347 | 0.9314 | 0.6469 |
| v2, retrained + MQuAKE hard negs, at own optimal threshold (0.90) | **15.94%** | 96.6% | 0.7422 | 0.9314 | 0.6469 |

## Honest reading

Real, measurable improvement — AUROC and AUPRC both genuinely improved
(0.9235->0.9314, 0.6083->0.6469), confirming the retrain taught the verifier
something real about MQuAKE-CF's confusion patterns, not nothing. **But this
is not the fix.** At its best operating point, false-fire only drops from
~18% to ~16%, and that gain costs real recall (99.9% -> 96.6%, meaning the
retrained verifier now genuinely misses ~3.4% of true matches it used to
catch, trading one error type for a partial reduction in the other).

This is a small-data result, worth being precise about: only 1,215 rows
(243 positive) of MQuAKE-CF-specific signal were added, against 21,285 rows
of CounterFact+WikiUpdate. The verifier learned *some* of MQuAKE-CF's
pattern from that slice, but not enough to fully close a gap this large --
consistent with the recalibration experiment's finding that the negative and
positive distributions genuinely overlap on this dataset, a harder problem
than a small amount of targeted data alone resolves.

## What this means going forward

This is now a well-characterized, three-attempt-deep finding, not a vague
"needs more work": (1) recalibrating the threshold alone does nothing
(ruled out cleanly); (2) a small amount of targeted retraining helps
modestly but incompletely; (3) the honest open question is whether scaling
up the MQuAKE-CF training slice, or adding structural features beyond raw
text similarity (e.g. explicit entity/relation graph density, which is the
actual property driving MQuAKE-CF's collisions per the Stage 1 retrieval
finding), would close the remaining gap. Not tested in this pass -- a
genuine, specific, defensible direction for future work rather than a loose
end.
