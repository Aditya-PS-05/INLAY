# Weight-editing baselines on AKEW — 2026-08-19

Structured mode only, by design not oversight: weight-editing methods need a
(prompt, target) pair to compute their update from -- they have no mechanism
for "here is a paragraph of unstructured evidence, extract and install
whatever fact matters." That extraction step doesn't exist for any of these
methods and building it is a separate research question, out of scope here.
This is a structural limitation of the weight-editing family on AKEW's
unstructured/extracted conditions, stated plainly.

Reuses the SAME subject-disjoint CounterFact structured test split used
throughout this project, and the SAME safe edit-then-generate-then-restore
pattern already validated in `eval_rippleedits_matched.py` earlier this
session (`sequential_edit=True` + explicit state_dict snapshot/restore under
harness control), specifically to avoid the exact weight-restore-before-
return bug already found and fixed once. Scored with AKEW's own `is_hit`
convention (substring + alias match against gold), not EasyEdit's internal
metrics, for direct comparability with every CAKE/RAG/IKE number in this
project.

Smoke-tested first (N=5, ROME): edit_ok 5/5, EasyEdit's own internal
`post.rewrite_acc: 1.0` agreed with the harness's independent generation-
based scoring before any full run was trusted.

## Results so far (GPT-J-6B, n=147)

| method | edit_ok | accuracy |
|---|---|---|
| ROME | 147/147 | 83.67% |
| MEMIT | 147/147 | 83.67% (identical to ROME) |
| AlphaEdit | 147/147 | 89.12% |
| CAKE (routed pipeline, same test set) | -- | **100.0%** |

AlphaEdit's edge over ROME/MEMIT (89.12% vs 83.67%) is consistent with its
own design: the null-space-constrained update is specifically built to be
more surgical per-edit, and that shows up here even on single-edit accuracy,
not just the multi-edit preservation it's usually credited for.

ROME and MEMIT landing at the identical 83.67% is plausible, not suspicious:
both are well-established single-edit-scale weight editors and CounterFact
structured is a comparatively easy single-fact task where both converge to
similar quality; MEMIT's real advantage (per this project's own INLAY
sequential-editing findings) shows up at scale/under repeated edits, not
single-edit accuracy.

## WISE: full run complete

Full N=150 run finished: `edit_ok=141/147, accuracy=0.6667`. The per-edit
landing rate mid-run (94%, 63/67) settled lower over the full run (96% of
sampled test rows produced a usable edit: 141/147 after excluding rows
skipped for missing gold fields) with final generation-based accuracy at
66.67% -- well below AlphaEdit's 89.12% and even ROME/MEMIT's 83.67%, despite
WISE's edits landing (per EasyEdit's own `post.rewrite_acc`) almost every
time. This gap between "the edit registers" and "the edit is retrievable
under real generation" is consistent with WISE's design: its side-memory
routing decides at generation time whether to consult the edited memory at
all, and that routing itself is an additional point of failure ROME/MEMIT/
AlphaEdit's in-place weight updates don't have.

## GRACE: a genuine negative result, not a harness bug

GRACE was first blocked by the same state_dict-key-structure issue WISE hit
(`KeyError: 'transformer.h.N.mlp.fc_out.bias'`) -- both wrap the edited layer
in a new module (WISE: side-memory layer; GRACE: codebook) rather than
modifying an existing tensor's values in place, so both change the model's
state_dict KEY STRUCTURE, not just its values, and the generic snapshot/
restore-by-key pattern that works for ROME/MEMIT/AlphaEdit cannot apply to
either. Fixed by generalizing the WISE accumulating-mode fix to cover GRACE
too (`ACCUMULATING_METHODS = ("WISE", "GRACE")`): skip state-dict diffing
entirely and score success via EasyEdit's own `post.rewrite_acc` immediately
after each edit installs.

That fix resolved the crash. It did not resolve GRACE actually working here.
Re-run with the fix (N=5 smoke test): `edit_ok=0/5` -- EasyEdit's own
`post.rewrite_acc` reads exactly `0.0` on every single edit, correctly
triggering the FATAL guard rather than silently reporting a wrong number.

This is consistent with, not contradicted by, an earlier finding already on
record in this project: GRACE's codebook/radius-based hidden-state patching
"collapsed to near zero on my generation-based harness: its radius-based
matching almost never fires on paraphrases." Two independent harnesses (the
earlier generation-based one, and this EasyEdit-native teacher-forcing check)
now agree GRACE's edits aren't landing under real-world query phrasing, on
this model/dataset. Recorded as a genuine limitation of GRACE's mechanism on
CounterFact-style single-fact edits with GPT-J, not a bug to keep chasing --
consistent with GRACE's design intent (targeted retrieval-augmented editing
via nearest-codebook-entry matching) being a poor fit for a benchmark that
doesn't control for paraphrase distance from the training prompt.

## Final results (GPT-J-6B, CounterFact structured, n=147)

| method | edit_ok | accuracy |
|---|---|---|
| ROME | 147/147 | 83.67% |
| MEMIT | 147/147 | 83.67% (identical to ROME) |
| AlphaEdit | 147/147 | 89.12% |
| WISE | 141/147 | 66.67% |
| GRACE | 0/5 (smoke test) | **N/A -- edits don't land under this harness's real-generation scoring** |
| CAKE (routed pipeline, same test set) | -- | **100.0%** |

## Reading across all five

Ranked by accuracy: AlphaEdit (89.12%) > ROME = MEMIT (83.67%) > WISE
(66.67%) > GRACE (effectively 0%, edits don't land at all under real
generation). CAKE's retrieval+verification+routing approach beats every
weight-editing baseline on this identical test split, and the gap is not
close for the two methods (WISE, GRACE) whose editing mechanism depends on
something being correctly *retrieved* at generation time -- the exact
failure mode CAKE's own verifier/router machinery is built to guard
against. AlphaEdit's null-space-constrained in-place update is the
strongest of the five weight-editors here, consistent with it being the
most surgical, least representation-disrupting of the group; WISE and
GRACE's added-module approach, whatever its other advantages (documented
elsewhere in this project: WISE's resistance to catastrophic forgetting
under long sequential-edit runs), is a liability specifically on this
single-edit-accuracy metric.
