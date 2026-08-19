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
| CAKE (routed pipeline, same test set) | -- | **100.0%** |

ROME and MEMIT landing at the identical 83.67% is plausible, not suspicious:
both are well-established single-edit-scale weight editors and CounterFact
structured is a comparatively easy single-fact task where both converge to
similar quality; MEMIT's real advantage (per this project's own INLAY
sequential-editing findings) shows up at scale/under repeated edits, not
single-edit accuracy.

AlphaEdit, WISE, GRACE in progress; each appended here as results land.
