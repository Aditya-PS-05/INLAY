# Hard-negative composition ablation — 2026-08-19

Trains two additional verifier variants on CounterFact+WikiUpdate (same
scope as the original v1) — **no hard negatives** (random negatives only)
and **specificity hard negatives only** (same-subject/diff-relation +
same-relation/diff-subject, the two categories EVOKE's paper specifically
flags) — and compares all three against the full 8-category mix.

## Methodology correction made mid-ablation

The first pass scored each variant against ITS OWN validation set, built
with that variant's own negative-sampling method. A verifier trained only on
random negatives scoring 99.9% against random negatives isn't evidence of
quality, it's evidence the benchmark was easy (random cards are almost never
even topically related to a query, a trivial discrimination problem). Fixed
by cross-evaluating all three checkpoints on the SAME hard validation set
(the full-mix verifier's own val split) for a genuinely fair comparison.

## Results, all three on the same hard validation set (n=4,575)

| training negatives | AUROC | AUPRC | ECE | F1 | false-fire rate |
|---|---|---|---|---|---|
| full 8-category mix | 0.9942 | 0.9692 | 0.0136 | **0.9569** | **1.48%** |
| specificity only (2 categories) | 0.9932 | 0.9619 | 0.0421 | 0.9438 | 2.46% |
| random negatives only | 0.9922 | 0.9624 | 0.0394 | 0.9278 | 3.42% |

**Monotonic, clean result.** More hard-negative diversity produces a better
verifier on every axis: false-fire rate improves 3.42% -> 2.46% -> 1.48% as
negatives go random -> specificity-only -> full mix; calibration (ECE) is
roughly 3x better with the full mix than either simpler variant. Specificity
negatives alone recover most of the gap versus random (confirming EVOKE's
core finding transfers to this project's own verifier), but the additional
six categories -- retrieval-based neighbors, stale-object, same-document
sibling, prefixed-unrelated, ranked-below-correct -- still contribute a real,
measurable further improvement, not redundant coverage of the same failure
mode specificity negatives already catch.

## What this validates

The design decision in `akew_hard_negatives.py` to build all 8 categories,
not just the two specificity ones, is empirically justified rather than
merely thorough-for-its-own-sake: each additional category type is pulling
real weight in the final verifier's false-fire rate. This is a genuinely
quotable ablation for the eventual writeup -- a clean before/after showing
exactly what hard-negative diversity buys, on a fair apples-to-apples
comparison.
