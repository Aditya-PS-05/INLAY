# Audit addendum — 2026-08-19 (matched RippleEdits completed; harness bug postmortem)

Extends `full_results_audit_2026-08-17.md`. Every number below is from a logged
run with `<<<JSON>>>` markers; hosts and log files cited inline.

## A. CRITICAL: all pre-Aug-18 weight-editor matched-RippleEdits results are INVALID

The `rm_ROME_gptj.log`, `rm_WISE_gptj.log`, `rm_AlphaEdit_gptj.log` results
(and the earlier crashed Qwen attempts) reported numbers identical to base
(aggregate 0.074, criteria digit-for-digit). Root cause, confirmed by probe:

1. EasyEdit's `BaseEditor.edit(sequential_edit=False)` applies the edit,
   evaluates internally, then RESTORES original weights before returning —
   so any harness that generates after `edit()` returns scores the unedited
   base model. No exception is raised.
2. A bare `except Exception: pass` around the edit call additionally swallowed
   WISE's `KeyError: 'loc_prompt'` on every example.

Patch (applied to `src/eval_rippleedits_matched.py` on virginia + ohio,
original backed up as `.bak_prefix`):
- `sequential_edit=True` + explicit per-edit state-dict restore (ROME/AlphaEdit)
- `loc_prompts` supplied for WISE; WISE runs in native accumulating mode,
  flagged `wise_mode: sequential_accumulating` in output JSON
- failure accounting (`edit_ok`/`edit_fail`/`first_edit_error`) in output
- hard exit(3) if every edit failed; exit(4) if a "successful" edit changed
  zero weights

The guard proved itself twice: it refused to emit numbers when AlphaEdit-Qwen
loaded a stale GPT-J null-space cache (16384 vs 18944 shape mismatch), and
again surfaced the `mom2_n_samples: 100000` config bomb in the Qwen yaml
(project convention is 3000; at ~8s/sample streaming, 100k × 5 layers ≈ 9 days).
Fixes: model-specific `P_loc` filenames in AlphaEdit yamls, upstream
`torch.save(P, hparams.P_loc)` patch in AlphaEdit_main.py, `mom2_n_samples: 3000`,
and the 5-layer Qwen covariance cache (t512_3000, 5×1.4GB) copied from
g6e4xlarge `/mnt/scratch/stats/Qwen2.5-7B/` to virginia
`data/stats/._hugging_cache_Qwen2.5-7B/wikipedia_stats/`.

## B. Matched RippleEdits — COMPLETE VALID TABLE (rm2 runs, Aug 18–19)

Protocol: identical wikidata-verified manifest (n=100 edits) for every method;
generation-based scoring; edits verified applied (edit_ok=100, edit_fail=0 in
every run). GPT-J runs on ohio (`rm2_*_gptj.log`), Qwen on virginia
(`rm2_*_qwen.log`). base/RAG/INLAY rows carried from the valid Aug-17 runs
(same manifest, gradient-free paths unaffected by the harness bug).

| aggregate       | GPT-J-6B | Qwen2.5-7B |
|-----------------|----------|------------|
| in_context (RAG)| 0.3964   | 0.4381     |
| WISE†           | 0.3425   | 0.1421     |
| INLAY (cake)    | 0.2253   | 0.2871     |
| AlphaEdit       | 0.1409   | 0.1683     |
| ROME            | 0.1314   | 0.1453     |
| base            | 0.0740   | 0.1522     |

† WISE in sequential_accumulating mode (side-memory not reset between edits —
its native lifelong-editing regime; all other methods are single-edit isolated
with per-edit weight restore).

Detail rows (propagation / preservation):
- ROME gptj: 0.1655/0.0633 · WISE gptj: 0.4774/0.0725 · AlphaEdit gptj: 0.1704/0.0819
- ROME qwen: 0.1646/0.1067 · WISE qwen: 0.0505/0.3254 · AlphaEdit qwen: 0.2060/0.0928

Headline findings:
1. INLAY is the ONLY editor above base on both models' aggregates.
2. On Qwen, INLAY (0.287) beats every weight/adapter editor; ROME and WISE land
   at or below the unedited base (0.152).
3. WISE is high-variance across models: strongest editor on GPT-J (0.343,
   driven by SubjAlias 0.894 and Comp-II 0.857), near-zero propagation on Qwen
   (0.05; Comp-II 0.0).
4. RAG leads everywhere — the paper's named INLAY limitation stands vs RAG,
   NOT vs the weight-editing family (previous assumption inverted).

## C. Other results landed since the Aug-17 audit

- **AlphaEdit CF GPT-J N=2000 (dualmetric), ohio `dm_alphaedit_gptj2k.log`**
  (closes the audit's "one real gap"): eff_prob 0.9955 / para_prob 0.9665 /
  eff_tok 0.9955 / para_tok 0.699 / neighborhood_keep 0.982 / 12.08s per edit.
  Metric family: dualmetric — NOT comparable to the EasyEdit-native hm column.
- **AlphaEdit zsRE GPT-J retry2, g6e4xlarge `zsre_alphaedit_gptj2k_retry2.log`**:
  ES 0.6347 / PS 0.5274 / NS 0.997 / hm 0.6705 — confirms the earlier ohio
  run (0.6698). Double-confirmed.
- **MEMIT zsRE GPT-J rerun, g6e4xlarge `zsre_memit_gptj2k.log`**: hm 0.6502 —
  confirms earlier ohio 0.6497. Double-confirmed.
- **Sequential editing, GPT-J (ohio)**: ROME collapses as on GPT-2-XL
  (retention/locality 0 by n=25, `seq_rome_gptj.log`); MEMIT holds retention
  and locality 1.0 through the recorded checkpoints (`seq_memit_gptj.log`).
  NOTE for the blog/paper: soften "weight editors compound damage" to name
  ROME specifically; MEMIT (built for mass editing) does not collapse here.
- **Mistral gap-fill (g6e4xlarge, in progress)**: MEMIT CF Mistral N=2000:
  ES 0.4218 / PS 0.2509 / NS 0.9466 / hm 0.4047 (`cf_memit_mistral2k.log`).
  Mistral CF now reads INLAY 0.900 > MEMIT 0.405 > ROME 0.340.
- **MEMIT zsRE Mistral N=2000 (g6e4xlarge, `zsre_memit_mistral2k.log`, completed
  2026-08-19 03:03 UTC)**: ES 0.6434 / PS 0.6178 / NS 0.9935 / hm 0.7178 /
  19.6s per edit. Mistral zsRE now reads INLAY 1.00 > MEMIT 0.718 > ROME 0.702.
  Remaining in chain: WISE CF (done, see below), WISE zsRE (running).
- **WISE CounterFact Mistral N=2000 (g6e4xlarge, `cf_wise_mistral2k.log`, completed
  2026-08-19 08:43 UTC)**: ES 0.9962 / PS 0.3097 / NS 1.00 / hm 0.5733 / 10.2s
  per edit (native accumulating mode, same caveat as entry 14/34 elsewhere).
  Mistral CF now reads INLAY 0.900 > WISE 0.573 > MEMIT 0.405 > ROME 0.340.

## D2. vNext audit: gate-bypass and product-key capacity bugs (fixed, validated)

Two confirmed bugs from a full code audit, both fixed on branch `vnext-gatefix`
(baseline preserved at git tag `cake-current`, commit `efaff81`):

**Bug A (gate bypass, CRITICAL for validity):** `answer_playback()` in
`gpt2_memory_semkey.py`, the function every real generation path runs through
(RippleEdits, demos), checked only the absolute score and silently skipped the
margin and relation-residual gates that `gated_logits()` already supported.
This means the relation gate's measured over-firing reduction (0.9667 -> 0.6667
at `rel_gate=0.2`, logged in `relgate2.log` on ohio) was validated only on a
separate teacher-forced harness (`eval_relgate_sweep.py`) and was **never
actually active** in the matched-RippleEdits run that produced the published
headline numbers. Fix: added `route()`, one shared REJECT/DIRECT decision used
identically by both `gated_logits()` and `answer_playback()`. Regression tests
in `tests/` verify same-subject/different-relation queries are now correctly
rejected and both code paths agree on every decision.

**Bug B (product-key capacity):** `ProductKeyMemory.write()` advanced both
allocator indices together, so every write landed on the diagonal, capping real
capacity at `n_sub` slots (4096, matching every eval config) against the
claimed `n_sub**2` (16.7M). Verified this did NOT corrupt any published number:
the single-edit-isolated protocol clears memory before every write, and the
400-edit sequential test never approached the diagonal ceiling. Fixed to a
row-major allocator over the full grid regardless.

**Validation rerun (rel_gate=0.2, the exact operating point behind the
published "0.97 -> 0.67" claim), CAKE only, matched manifest:**

| aggregate | published (bug-affected) | corrected (fix genuinely active) | Δ |
|---|---|---|---|
| GPT-J | 0.2253 | 0.2283 | +0.0030 |
| Qwen | 0.2871 | 0.2891 | +0.0020 |

Both within noise. **No correction to the published headline numbers is
needed.** GPT-J detail at rel_gate=0.2: Relation_Specificity 0.0867,
Forgetfulness 0.0370, preservation_avg 0.0619 -- still weak even with the fix
genuinely active, meaning the relation gate's isolated over-firing metric
(0.97->0.67) does not translate into a meaningfully stronger matched-RippleEdits
preservation score at this operating point. The honest reading: the bug was
real, but the underlying compositional/preservation weakness this section of
the blog already names as an open limitation is not solved by activating the
gate at this threshold. A more aggressive rel_gate (0.4-0.5) cuts over-firing
further per the original sweep (down to 0.37/0.10) at real propagation cost
(0.77/0.47) -- untested against the full matched protocol; a candidate for a
future ablation, not run here.

## D. Remaining open gaps after this addendum

- Mistral: WISE + MEMIT zsRE (running); GRACE/AlphaEdit on Mistral never run.
- Sequential at GPT-J scale for INLAY/WISE/AlphaEdit (harness support needed).
- Apples-to-apples MEMIT/AlphaEdit Qwen CF in EasyEdit-native metric.
- WISE matched-ripple in a per-edit-isolated protocol (current run is
  accumulating mode; a fresh-editor-per-edit variant would isolate the mode
  effect, at ~100× editor-init cost).
