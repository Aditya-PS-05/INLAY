# Mistral gap-fill: GRACE and AlphaEdit, CounterFact + zsRE, N=2000

**2026-08-22/23.** Closes the two remaining open items flagged in
`full_results_audit_2026-08-19_addendum.md` Part C/D: *"GRACE and AlphaEdit on
Mistral remain unrun, the only open items left for [Mistral]."* All four runs below
(GRACE CF, GRACE zsRE, AlphaEdit CF, AlphaEdit zsRE) completed successfully at N=2000,
matching this project's established convention for Mistral (ROME/MEMIT/WISE were already
run at N=2000 per the addendum's Part C).

## Result

| Method | Benchmark | N | ES | PS | NS | score(hm) | write cost | Source log |
|---|---|---|---|---|---|---|---|---|
| GRACE | CounterFact | 2000 | 0.0 | 0.0059 | 1.0 | **0.0** | 6.54s | ohio `kw_grace_cf_mistral2k.log` |
| GRACE | zsRE | 2000 | 0.1214 | 0.0032 | 1.0 | **0.0092** | 9.11s | ohio `kw_grace_zsre_mistral2k.log` |
| AlphaEdit | CounterFact | 2000 | 0.4156 | 0.261 | 0.9095 | **0.4089** | 10.29s | g6e4xlarge `kw_alphaedit_cf_mistral2k.log` |
| AlphaEdit | zsRE | 2000 | 0.616 | 0.5928 | 0.9913 | **0.6946** | 11.72s | g6e4xlarge `kw_alphaedit_zsre_mistral2k.log` |

**Mistral-7B-v0.3 CounterFact now reads (all N=2000 except INLAY/base/RAG per the
project's own established N=5000/N=2000 convention):** INLAY 0.8995 (N=5000) >
WISE 0.5733 > AlphaEdit **0.4089** > MEMIT 0.4047 > ROME 0.3398 > GRACE **0.0**.

**Mistral-7B-v0.3 zsRE now reads:** INLAY 0.9999 (N=2000) > WISE 0.9570 > MEMIT 0.7178
> ROME 0.7018 > AlphaEdit **0.6946** > GRACE **0.0092**.

(Note ROME's zsRE hm of 0.7018 sits between MEMIT and AlphaEdit's new numbers rather than
below them, unlike on CounterFact where ROME trails everything but GRACE — consistent
with ROME's per-model unevenness already noted elsewhere in this project, e.g. its
weaker, untuned-hyperparameter CounterFact showing on Mistral.)

This closes the Mistral gap-fill chain entirely: CounterFact and zsRE, all six comparable
methods (ROME, MEMIT, WISE, GRACE, AlphaEdit, INLAY) now have a number on Mistral-7B-v0.3.

## Corroboration against this project's own GRACE/AlphaEdit pattern on other models

Per this project's stated convention — treat GRACE's near-zero result as a real finding
once corroborated, not a bug — both new numbers fall inside the range already established
on GPT-J and Qwen, rather than standing alone as an unverified anomaly:

| Method | Benchmark | GPT-J-6B | Qwen2.5-7B | Mistral-7B-v0.3 (new) |
|---|---|---|---|---|
| GRACE | CounterFact hm | 0.0 | 0.0 | **0.0** |
| GRACE | zsRE hm | 0.0035 | 0.0555 | **0.0092** |
| AlphaEdit | CounterFact hm | 0.4595 | (dualmetric only, not hm-comparable) | **0.4089** |
| AlphaEdit | zsRE hm | 0.6698 | 0.6824 | **0.6946** |

GRACE lands at exactly 0.0 on CounterFact for the third model family in a row (GPT-J,
Qwen, now Mistral), and its zsRE number (0.0092) sits between the GPT-J (0.0035) and
Qwen (0.0555) values rather than as an outlier. AlphaEdit's zsRE number (0.6946) is
within 0.03 of both GPT-J (0.6698) and Qwen (0.6824). Both new results are consistent
with — not contradicted by — this project's existing pattern; no follow-up run was
needed to corroborate an anomaly because there is no anomaly here.

## Protocol

Followed the same convention as the existing Mistral MEMIT/WISE gap-fill runs
(`mistral_gapfill.sh`, referenced in the addendum): `eval_cf_we.py` / `eval_zsre_we.py`
against EasyEdit's own native `data/counterfact.json` (21,919 records) /
`data/zsre.json` (19,086 records) — **not** `akew_eval_weightedit.py`, which targets
AKEW's separate, much smaller 975-record `CounterFact.json` used for Task 1's out-of-scope
condition. This substitution from the task brief's suggested script is deliberate and
explicit: `eval_cf_we.py`/`eval_zsre_we.py` is what produced every other N=2000
EasyEdit-native ES/PS/NS number cited across this project's audits (ROME, MEMIT, WISE,
GRACE, AlphaEdit on GPT-J/Qwen), so using it for the Mistral gap-fill keeps every number
in this table on the same metric family and is directly comparable to them. `mom2_n_samples`
forced to 3000 (project convention; the yaml default of 100000 would take ~9 days per the
addendum's own "config bomb" note).

Both editors use `sequential_edit=False, keep_original_weight=True` inside
`eval_cf_we.py`/`eval_zsre_we.py` — EasyEdit's own per-edit isolated protocol, where
`editor.edit()` restores weights internally after scoring each edit (verified this is
safe here specifically because the SAME harness already produced every other N=2000
number in this table; the sequential_edit=False bug this project found and fixed
elsewhere was in a *different*, hand-rolled harness — `eval_rippleedits_matched.py` and
`akew_eval_weightedit.py` — that generated AFTER `edit()` returned rather than inside
EasyEdit's own scoring call).

## Bugs found and fixed to make these runs possible

Two real bugs in the vendored EasyEdit `AlphaEdit_main.py`, both a missing `"mistral"`
branch in an architecture-name string match — Mistral was never added to either check when
Llama/GPT-J/Qwen support was written:

1. **Null-space matrix `P` shape selection** (line ~64): the branch that decides `P`'s
   shape checked for `"llama"`/`"gpt-j-6b"`/`"qwen"` but not `"mistral"`, so `P` would
   silently fall through to no branch and crash later with a shape mismatch. Fixed by
   adding `"mistral" in hparams.model_name.lower()` to the condition.
2. **`cache_c` shape selection** (line ~81), a **second, separate occurrence of the exact
   same architecture-name check** a few lines below the first, controlling a different
   accumulator (`cache_c`, not `P`). The first fix alone was not sufficient: after
   patching only the `P` branch, AlphaEdit ran past `P` computation but crashed
   `~1 minute` into CounterFact edit #1 with `NameError: name 'cache_c' is not defined`
   — the second branch had fallen through silently with no `else`, leaving `cache_c`
   never assigned. Found by reading the actual traceback (this project's own stated
   discipline: read the error, don't guess) rather than assuming the first fix covered
   it. Fixed identically: added `"mistral" in hparams.model_name.lower()` to the second
   check.

Both fixes applied on both g6e4xlarge and ohio-g6e2xlarge (`AlphaEdit_main.py.bak_mistral_patch`
holds the pre-patch original on each host). Also added `hparams/GRACE/mistral-7b.yaml`
(adapted from the `llama-7b.yaml`/`gpt-j-6b.yaml` templates, `inner_params:
model.layers[27].mlp.down_proj.weight` — layer 27 of Mistral's 32, matching the
proportional depth EasyEdit's own Llama-2-7B GRACE config uses) and
`hparams/AlphaEdit/mistral-7b.yaml` (adapted from the `qwen2.5-7b.yaml` template, with
`mom2_n_samples: 3000` set directly in the yaml rather than patched at runtime, and a
dedicated `P_loc` path so its 4.1GB null-space cache does not collide with the existing
`null_space_project.pt` other models use). These new yaml files are **not** included in
this write-up as artifacts but exist on both GPU hosts under `~/cake/EasyEdit/hparams/`;
they should be copied into the repo's own hparams tree if a future session needs to
reproduce this without re-deriving them.

## Timeline / infrastructure notes (fail-loud discipline)

Both editors were initially launched as a single combined CF+zsRE job with a 2-hour
timeout, following the layout of the historical `mistral_gapfill.sh` script. Both were
under-provisioned and had to be cancelled and resubmitted once measured per-edit rates
were available (GRACE ~6.5-9s/edit, AlphaEdit ~10-12s/edit; at N=2000 CF+zsRE combined
that is 7-9 GPU-hours, not the ~2 hours the historical script's timeout implied for a
smaller/faster prior configuration). Once resubmitted as four separate jobs with
timeouts sized from directly measured throughput, all four completed cleanly with
`exit_code=0`. No number in the table above was accepted from a run that hit a timeout
or crashed; the two `AlphaEdit` runs specifically required fixing the `cache_c`
`NameError` above before producing a valid result on the first non-crashing attempt.

## Reproducing

```
cd ~/cake/EasyEdit && source /opt/pytorch/bin/activate
export HF_HOME=<scratch>/hf_cache   # keep model + null-space caches off the small root disk
python3 eval_cf_we.py   GRACE     ./hparams/GRACE/mistral-7b.yaml     mistral-7b 2000
python3 eval_zsre_we.py GRACE     ./hparams/GRACE/mistral-7b.yaml     mistral-7b 2000
python3 eval_cf_we.py   AlphaEdit ./hparams/AlphaEdit/mistral-7b.yaml mistral-7b 2000
python3 eval_zsre_we.py AlphaEdit ./hparams/AlphaEdit/mistral-7b.yaml mistral-7b 2000
```

## Source logs

- `~/kw_grace_cf_mistral2k.log`, `~/kw_grace_zsre_mistral2k.log` (ohio-g6e2xlarge)
- `~/kw_alphaedit_cf_mistral2k.log`, `~/kw_alphaedit_zsre_mistral2k.log` (g6e4xlarge)
- `~/cake/EasyEdit/easyeditor/models/alphaedit/AlphaEdit_main.py` (patched on both hosts;
  `.bak_mistral_patch` holds the pre-patch original)
- `~/cake/EasyEdit/hparams/GRACE/mistral-7b.yaml`, `~/cake/EasyEdit/hparams/AlphaEdit/mistral-7b.yaml`
  (new, both hosts)
