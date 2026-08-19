# AKEW scope-verifier: training and zero-shot generalization — 2026-08-19

Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, fine-tuned) trained on
CounterFact + WikiUpdate (all 3 input modes each), subject-disjoint 70/15/15
train/val/test split within each dataset (brief section 9). MQuAKE-CF held
out **entirely** from training and threshold calibration -- used only for the
zero-shot generalization test, matching the brief's suggested alternative
validation experiment.

Training data: 21,285 rows (4,257 positive, 17,028 negative, 1:4 ratio, capped
and spread across the 8 hard-negative categories per positive rather than
letting the numerically dominant categories drown out the rarer ones).
2 epochs, 98s total training time on one GPU. Threshold (0.65) selected by
sweeping F1 on the validation split only, never on test data.

## Held-out validation (CounterFact + WikiUpdate, subject-disjoint)

| metric | value |
|---|---|
| AUROC | 0.9939 |
| AUPRC | 0.9675 |
| ECE | 0.0136 |
| Precision | 0.9417 |
| Recall | 0.9716 |
| F1 | 0.9564 |
| False-fire rate | 1.50% |
| False-reject rate | 2.84% |

Strong discrimination and good calibration on data drawn from the same
distributions the verifier trained on.

## Zero-shot: MQuAKE-CF (never touched during training or calibration)

| metric | value |
|---|---|
| AUROC | 0.9235 |
| AUPRC | 0.6083 |
| ECE | 0.1446 |
| Precision | 0.5832 |
| Recall | 0.9992 |
| F1 | 0.7365 |
| False-fire rate | 17.85% |
| False-reject rate | 0.08% |

**Honest reading, not glossed over:** ranking ability transfers reasonably
(AUROC only drops 0.994 -> 0.924) and the verifier essentially never misses a
true positive on MQuAKE-CF (recall 99.9%), but the **fixed threshold
calibrated on CounterFact+WikiUpdate is badly miscalibrated for MQuAKE-CF
specifically** -- false-fire rate jumps from 1.5% to 17.85%, nearly 12x worse,
and AUPRC collapses from 0.968 to 0.608. ECE more than 10x worse (0.014 ->
0.145) confirms this is a genuine calibration failure, not just a harder
ranking problem.

This lines up directly with the Stage 1 retrieval pilot's own finding
(`akew_schema_validation.md`): MQuAKE-CF was built for multi-hop chains, so
its entities and relations recur across records far more than CounterFact's
or WikiUpdate's largely-independent facts do. A verifier trained where
same-subject/same-relation collisions are comparatively rare learns a
threshold tuned for that regime, and that threshold is simply too permissive
once applied to a benchmark where those collisions are structurally common.
**The fix is not "train longer" or "tune harder" -- it's that the threshold
itself needs to be recalibrated per-dataset, or the verifier needs training
examples that include MQuAKE-CF-style entity-dense hard negatives**, which
were entirely absent from its training distribution.

## What this does NOT establish

This validates the verifier as a standalone classifier (given a query and one
candidate card, does it correctly predict match/no-match). It does NOT yet
validate the full router-in-context (Stage 1 retrieval -> Stage 2 verifier
-> REJECT/DIRECT/REASON decision) end to end, and it does NOT yet touch
answer generation at all (brief section 5). Those are the next two phases.

## Known issue found and fixed during this run

`CrossEncoder.fit(..., output_path=..., save_best_model=True)` did NOT
persist model weights in the installed sentence-transformers version (5.6.1,
which routes CrossEncoder training through an HF-Trainer-backed
implementation internally) -- confirmed by direct inspection: after the first
training run completed cleanly with real metrics logged, the output
directory contained only an `eval/` results CSV, no `config.json` or
`model.safetensors` anywhere on disk, and the trained weights were
unrecoverable (lost with the process). Fixed by adding an explicit
`model.save(output_path)` call plus a hard assertion that the expected files
actually landed, so a future silent-non-save regression fails loudly instead
of discovering it after the fact. Retrained from scratch (98s, cheap) with
the fix; the results above are from the properly-persisted checkpoint.
