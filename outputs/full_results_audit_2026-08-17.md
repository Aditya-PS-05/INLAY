# Full ground-truth results audit — INLAY (CAKE) knowledge-editing project

Compiled 2026-08-17 by re-reading every source directly (SSH into all 4 live GPU hosts, grep every
log with a `<<<JSON>>>` marker; read every .json in the local repo's `cake_prototype/outputs/`).
No numbers taken from conversation memory. Every entry cites its exact source file + host.

Note on naming: the method is called "CAKE" or "cake" in nearly all early logs (the working name
before the INLAY rename this session); "CAKE-scale" / "CAKE-gptj" / "CAKE_v3" / "CAKE_v3_multitoken"
are all the same INLAY method at different code-maturity checkpoints. I keep the literal string
from each source so it's traceable, and note which are superseded.

Two jobs are STILL RUNNING as of this audit (checked live, no result yet):
- ohio-g6e2xlarge: `eval_cf_dualmetric.py AlphaEdit ... gpt-j-6b 2000` → will write `dm_alphaedit_gptj2k.log`
- g6e4xlarge: `eval_zsre_we.py MEMIT ... gpt-j-6b 2000` → will write `zsre_memit_gptj2k.log`, then AlphaEdit retry2 queued behind it

---

## PART A — CounterFact, single-edit, by model

### A1. GPT-2-XL (1.5B) — the original development model

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| base | 100 | 0.01 | 0.00 | 1.00 | 0.00 | 0.0s | ohio `cf_base.log` |
| in_context (RAG) | 100 | 0.92 | 0.405 | 0.44 | 0.5147 | 0.0s | ohio `cf_in_context.log` |
| finetune | 100 | 0.63 | 0.40 | 0.018 | 0.0503 | 6.62s, 25 steps | ohio `cf_finetune.log` |
| ROME | 100 | 0.955 | 0.43 | 0.88 | 0.6653 | 11.87s, 20 steps | ohio `EasyEdit/cf_rome100.log` |
| ROME (earlier variant, first-token prob metric) | 50 | 0.18 | 0.23 | 0.786 | 0.2684 | 10.71s | ohio `EasyEdit/cf_rome.log` — different metric, not comparable to the row above |
| ROME (n=10, same variant) | 10 | 0.20 | 0.45 | 0.74 | 0.3499 | 11.55s | ohio `EasyEdit/cf_rome10.log` |
| MEMIT | 100 | 0.0 | 0.0 | 0.99 | 0.0 | 12.70s, 100 steps | ohio `EasyEdit/cf_memit100.log` |
| CAKE (early, layer-tuned hidden-state key, v1) | 100 | 1.0 | 0.15 | 0.694 | 0.3294 | 2.72s | ohio `cf_cake.log` — **superseded by semantic-key v2 below** |
| CAKE (v1, n=30 smoke) | 30 | 1.0 | 0.10 | 0.77 | 0.2439 | 0.867s | ohio `cf_cake30.log` |
| CAKE-semkey(prompt) v2, best gate 0.45 | 100 | 1.0 | 0.85 | 0.82 | 0.8834 | — | ohio `cf_semkey_prompt.log` |
| CAKE-semkey(prompt) v2, n=30 sweep, best gate 0.45 | 30 | 1.0 | 0.80 | 0.8367 | 0.8708 | — | ohio `cf_semkey30.log` |
| CAKE-semkey(subject) — keys on subject only | 100 | 1.0 | 1.0 | 0.005 | 0.0149 | — | ohio `cf_semkey_subject.log` — **subject-only key destroys locality, dead end, not used further** |
| CAKE (held-out tune/test split, prompt key) | n_tune=100,n_test=100 | 1.0 | 0.86 | 0.859 | 0.9017 | — | ohio `cf_semkey_multi.log` = `cf_split.log` (identical, duplicate run) |
| CAKE margin-gate, N=400, best margin=0.15 | 400 | — | — | locality 1.0 | 0.9975 | — | ohio `margin2.log` — margin-gate ablation; retention 0.995 |

**GPT-2-XL final consolidated leaderboard (local repo `comparison_full.json` + `results.json`, this is the authoritative summary the project settled on):**

| Method | ES | PS | NS | Score(hm) | Write cost | Notes |
|---|---|---|---|---|---|---|
| base | 0.0 | — | 1.0 | 0.0 | 0.0s | |
| in_context (RAG) | 1.0 | — | 1.0 | 1.0 | 0.0s | doc re-fed every query |
| finetune | 1.0 | — | 0.25 | — | 34.7s, 60 steps | catastrophic forgetting |
| **CAKE** | 1.0 | — | 1.0 | — | 0.152s | 0 spurious fires, gate min_score=0.9 |
| ROME | 0.2 | — | 0.0 | — | 15.0s, 20 steps | 5 same-subject facts collide → degenerate |
| MEMIT | 0.4 | — | 0.917 | — | 10728.5s | dominated by one-time 3h covariance precompute (cached); real edit is seconds |

Source: `/home/aditya/my-work/AI/research/knowledge-editing/cake_prototype/outputs/comparison_full.json`

### A2. GPT-J-6B

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| base | 1000 | 0.004 | 0.003 | 1.0 | 0.0051 | — | ohio `cf_base_n1000.log`; also 5000-scale: base n=5000 ES 0.0058/PS 0.0045/NS 1.0/score 0.0076 (london `cf_base_gptj5k.log`) |
| in_context (RAG) | 1000 | 0.855 | 0.464 | 0.4608 | 0.546 | — | ohio `cf_in_context_n1000.log`; also n=5000: ES 0.847/PS 0.4597/NS 0.4569/score 0.5411 (london `cf_rag_gptj5k.log`) |
| ROME | 2000 | 0.988 | 0.7568 | 0.699 | 0.797 | 8.60s, 20 steps | ohio `cf_rome_gptj2k.log` |
| ROME (dualmetric protocol, n=2000) | 2000 | eff_prob 0.995/eff_tok 0.005 | par_prob 0.9835/par_tok 0.139 | 0.7055 | — | 7.77s | ohio `dm_rome_gptj2k.log` — different metric family (prob-success vs token-acc), do not average with the row above |
| ROME (n=100, EasyEdit-native) | 100 | 0.995 | 0.755 | 0.60 | 0.7507 | 8.62s, 20 steps | ohio `EasyEdit/romej_cf.log` |
| MEMIT | 2000 | 0.458 | 0.2672 | 0.972 | 0.4314 | 26.30s | ohio `cf_memit_gptj2k.log` |
| MEMIT (dualmetric) | 2000 | eff_prob 0.998/eff_tok 0.9965 | par_prob 0.951/par_tok 0.6595 | 0.972 | — | 25.73s | ohio `dm_memit_gptj2k.log` |
| MEMIT (n=100) | 100 | eff 1.0/eff_tok 1.0 | par 0.98/par_tok 0.65 | 0.96 | — | 26.28s | ohio `dm_MEMIT.log` |
| WISE (n=2000, run 1) | 2000 | 1.0 | 0.4415 | 0.999 | 0.7032 | 14.76s | virginia `cf_wise_gptj2k.log` |
| WISE (n=2000, run 2, near-duplicate rerun) | 2000 | 1.0 | 0.4422 | 0.999 | 0.7039 | 15.34s | ohio `cf_wise_gptj2k.log` — **consistent with virginia's run above, treat as confirming replicate** |
| WISE (n=100) | 100 | 1.0 | 0.385 | 1.0 | 0.6525 | 15.10s | ohio `EasyEdit/wise_cf.log` |
| GRACE (n=2000) | 2000 | 0.0 | 0.006 | 1.0 | 0.0 | 17.35s | ohio `cf_grace_gptj2k.log` |
| GRACE (n=2000, virginia retry after loc_prompt fix) | 2000 | 0.0 | 0.006 | 1.0 | 0.0 | 17.35s | virginia `cf_grace_gptj2k_retry.log` — **identical result to ohio's, confirms GRACE genuinely scores ~0 here, not a bug artifact** |
| GRACE (n=100) | 100 | 0.0 | 0.0 | 1.0 | 0.0 | 17.70s | ohio `EasyEdit/grace_cf.log` |
| AlphaEdit (n=2000) | 2000 | 0.4682 | 0.2963 | 0.982 | 0.4595 | 12.12s | ohio `cf_alphaedit_gptj2k.log` |
| AlphaEdit (dualmetric, n=100) | 100 | eff 1.0/eff_tok 1.0 | par 0.98/par_tok 0.68 | 0.97 | — | 12.03s | ohio `dm_AlphaEdit.log` |
| AlphaEdit (n=100, EasyEdit-native) | 100 | 0.46 | 0.26 | 0.98 | 0.4261 | 12.84s | ohio `EasyEdit/ae_cf2.log` |
| **AlphaEdit (n=2000, CounterFact) — GPT-J: NO SUCCESSFUL RESULT YET.** | 2000 | — | — | — | — | — | **STILL RUNNING** on ohio (`dm_alphaedit_gptj2k.log`, launched this session after fixing the stale null_space_project.pt cache bug). This is the one real gap the audit surfaces. |
| **CAKE / INLAY v3 multitoken, n=1000 (the number the project cites as "GPT-J-6B ~0.89")** | 1000 | 1.0 | 0.868 | 0.8352 | 0.8957 | — | local `gptj_final_leaderboard.json` `CAKE_v3`; consistent with `cf_cake_n1000.log` (test: ES 1.0/PS 0.868/NS 0.8352/score 0.8957) |
| CAKE / INLAY, n=100 (stability check) | 100 | 1.0 | 0.85 | 0.828 | 0.8865 | — | local `gptj_all_methods.json` |
| CAKE / INLAY, n=5000 (the headline number used throughout this session, "0.89 CounterFact GPT-J") | n_tune=500,n_test=4500 | 0.9976 (test) | 0.8711 | 0.826 | **0.8926** | load 147.58s | london `cf_cake_gptj5k.log` |
| CAKE / INLAY dualmetric, n=2000 | 2000 | eff_prob 0.998/eff_tok 0.9975 | par_prob 0.8928/par_tok 0.8655 | 0.8338 | — | 0.0135s/edit | london `dm_cake_gptj2k.log` |
| CAKE / INLAY, n=100 (via dm harness) | 100 | eff_prob 1.0/eff_tok 1.0 | par_prob 0.885/par_tok 0.855 | 0.82 | — | 0.0138s | ohio `cake_dm.log` |

**GPT-J-6B leaderboard as consolidated in the local repo (`gptj_final_leaderboard.json`), the number the project has been citing as canonical, at matched sample sizes (gradient-free methods N=1000, weight-editors N=100 — see file's own `sample_sizes` field for why the split exists):**

| Method | N | Score(hm) |
|---|---|---|
| **CAKE v3** | 1000 | **0.896** |
| ROME | 100 | 0.751 |
| WISE | 100 | 0.653 |
| RAG (in-context) | 1000 | 0.546 |
| AlphaEdit | 100 | 0.426 |
| GRACE | 100 | 0.000 |
| MEMIT | 100 | 0.000 |
| Base | 1000 | 0.005 |

At larger, later N (this session's runs, all N=2000 except CAKE which ran at N=5000 and N=2000 both):
CAKE 0.8926 (N=5000) / 0.8338–0.8655 range (N=2000 dualmetric) > ROME 0.797 (N=2000) > WISE 0.703–0.704 (N=2000) > AlphaEdit 0.4595 (N=2000) > MEMIT 0.4314 (N=2000) > GRACE 0.0 (N=2000).

### A3. Qwen2.5-7B

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| base | 5000 | 0.0106 | 0.0108 | 1.0 | 0.016 | — | ohio `cf_base_qwen5k.log` |
| in_context (RAG) | 5000 | 0.7983 | 0.4933 | 0.4122 | 0.5258 | — | ohio `cf_rag_qwen5k.log` |
| ROME | 2000 | 0.997 | 0.5538 | 0.9487 | 0.7767 | 9.30s | virginia `cf_rome_qwen2k.log` |
| ROME (smoke, n=3) | 3 | 1.0 | 0.5 | 1.0 | 0.75 | 10.17s | virginia `rome_qwen_smoke.log` — trivial N, ignore |
| MEMIT (dualmetric) | 2000 | eff_prob 1.0/eff_tok 0.9982 | par_prob 0.981/par_tok 0.714 | 0.978 | — | 34.32s | g6e4xlarge `dm_memit_qwen2k.log` |
| WISE (v1, buggy eval_cf_dualmetric.py — CRASHED, see failures) | 2000 | — | — | — | — | — | crashed twice, see Part D |
| WISE (v2, via eval_cf_we.py, the successful run) | 2000 | 1.0 | 0.8553 | 1.0 | **0.9466** | 27.06–27.23s | ohio `cf_wise_qwen2k.log` and `cf_wise_qwen2k_v2.log` — **identical numbers in both files, confirmed reproducible** |
| GRACE | 2000 | 0.0 | 0.0055 | 1.0 | 0.0 | 6.79–6.99s | g6e4xlarge `cf_grace_qwen2k.log` (0.0/0.0055/1.0/0.0, 6.785s, mtime Aug 2) — earlier ohio run of a differently-scoped variant not directly comparable |
| AlphaEdit (dualmetric) | 2000 | eff_prob 1.0/eff_tok 0.9982 | par_prob 0.984/par_tok 0.718 | 0.9735 | — | 16.24s | g6e4xlarge `dm_alphaedit_qwen2k.log` — **this is the "0.72 token-PS" number cited throughout this session** |
| CAKE / INLAY, n=5000 (headline "0.89 Qwen" number) | n_tune=500,n_test=4500 | 1.0 | 0.8776 | 0.8232 | **0.8944** | load 7.87s | ohio `cf_cake_qwen5k.log` |

**Qwen2.5-7B CounterFact ranking at matched-ish scale:** CAKE 0.894 (N=5000) ≈ WISE 0.947 (N=2000, actually *higher* than CAKE on Qwen — note this explicitly, WISE wins CounterFact on Qwen specifically) > ROME 0.777 (N=2000) > AlphaEdit dualmetric (efficacy/paraphrase-prob framing, not directly hm-comparable — token-based hm would be lower) > MEMIT dualmetric (same caveat) > RAG 0.526 (N=5000) > GRACE ~0 > base ~0.

**Correction to something stated earlier in this session:** MEMIT and AlphaEdit's Qwen CounterFact numbers were reported this session using the "dualmetric" prob-success framing (efficacy_prob_success/paraphrase_prob_success), which reads much higher than the token-accuracy harmonic-mean score used for CAKE/ROME/WISE/GRACE. There is no MEMIT-Qwen or AlphaEdit-Qwen result in the EasyEdit-native token-accuracy ES/PS/NS/score_hm format anywhere in the logs, so a true apples-to-apples harmonic-mean comparison against CAKE's 0.894 is NOT actually available for MEMIT/AlphaEdit on Qwen CounterFact — only the prob-success variant exists. Flag this precisely when building the final table; don't present MEMIT-Qwen-CF "0.88" as if it's the same metric as CAKE's 0.89.

### A4. Mistral-7B-v0.3

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| CAKE/INLAY smoke (BOS-tokenization bug present, pre-fix) | n_tune=20,n_test=20 | 0.1708 (test) | 0.1646 | 0.82 | 0.2282 | — | ohio `smoke_mistral.log` — **FAILED run due to bug, kept only for the audit trail; superseded** |
| CAKE/INLAY alpha sweep 40 (pre-fix, still buggy) | n=50/50 | 0.4117 | 0.407 | 0.826 | 0.4921 | — | ohio `sweep_mistral_a40.log` |
| CAKE/INLAY alpha sweep 80 (pre-fix) | n=50/50 | 0.555 | 0.5527 | 0.792 | 0.6155 | — | ohio `sweep_mistral_a80.log` |
| CAKE/INLAY alpha sweep 160 (pre-fix) | n=50/50 | 0.575 | 0.5577 | 0.728 | 0.6115 | — | ohio `sweep_mistral_a160.log` |
| CAKE/INLAY alpha sweep 320 (pre-fix) | n=50/50 | 0.575 | 0.5577 | 0.728 | 0.6115 | — | ohio `sweep_mistral_a320.log` (identical to a160, expected — pre-fix bug flattened the alpha response) |
| CAKE/INLAY alpha=160 AFTER the `add_special_tokens=False` fix | n=50/50 | 1.0 | 0.895 | 0.83 | **0.903** | — | ohio `sweep_mistral_fixed_a160.log` — **this is the run that proved the fix worked** |
| **CAKE/INLAY, n=5000 (post-fix, the headline "0.90 Mistral" number)** | n_tune=500,n_test=4500 | 1.0 | 0.8895 | 0.8257 | **0.8995** | — | ohio `ladder_cf_mistral5k.log` |
| ROME | 2000 | 0.2846 | 0.2364 | 0.9211 | 0.3398 | 9.04s | ohio `ladder_cf_rome_mistral2k.log` |

**No MEMIT, WISE, GRACE, or AlphaEdit CounterFact results exist for Mistral-7B anywhere.** Only CAKE/INLAY and ROME were run on this model family for CounterFact. This is a genuine, still-open gap if a full leaderboard row is wanted for Mistral.

---

## PART B — zsRE, single-edit, by model

### B1. GPT-2-XL

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| base | 100 | 0.2058 | 0.2063 | 1.0 | 0.2801 | 0.0s | ohio `zsre_base.log` |
| in_context (RAG) | 100 | 0.9383 | 0.8472 | 0.88 | 0.8869 | 0.0s | ohio `zsre_in_context.log` |
| finetune | 100 | 0.7923 | 0.7457 | 0.01 | 0.0292 | 6.61s, 25 steps | ohio `zsre_finetune.log` — catastrophic forgetting |
| ROME | 100 | 1.0 | 0.795 | 0.9564 | 0.9082 | 3.78s, 20 steps | ohio `EasyEdit/zsre_rome.log` |
| MEMIT | 100 | 0.2808 | 0.2436 | 0.9961 | 0.346 | 4.37s, 100 steps | ohio `EasyEdit/zsre_memit.log` |
| CAKE v2 (semantic key, single-token value — capped by multi-token zsRE answers) | n_tune=50,n_test=50 | 0.5285 (test) | 0.5218 | 1.0 | 0.6239 | 0.29s | ohio `zsre_cake.log` = local `zsre.json`'s `CAKE_v2_semkey` — **superseded by v3 below** |
| CAKE v3 (multi-token logit-space playback fix) | n_tune=50,n_test=50 | 0.96 (test) | 0.96 | 1.0 | **0.973** | 0.34s | ohio `zsre_cake_multi.log` = local `zsre_cake_v3.json` — **this fix is what makes CAKE beat ROME on zsRE (0.973 > 0.908)** |
| CAKE, n=20 (tiny) | n_tune=10,n_test=10 | 0.5083 | 0.4833 | 1.0 | 0.5957 | 0.06s | ohio `zsre_cake20.log` — pre-v3-fix, low N, superseded |

### B2. GPT-J-6B

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| base | 2000 | 0.2248 | 0.2146 | 1.0 | 0.2968 | 0.0s | ohio `zsre_base_gptj.log` |
| in_context (RAG) | 2000 | 0.9069 | 0.8375 | 0.8855 | 0.8757 | 0.0s | ohio `zsre_rag_gptj.log` |
| ROME | 2000 | 0.9951 | 0.9412 | 0.8816 | 0.937 | 10.23s | virginia `zsre_rome_gptj2k.log` |
| WISE | 2000 | 0.9987 | 0.9892 | 1.0 | 0.9959 | 17.28s | virginia `zsre_wise_gptj2k.log` |
| GRACE | 2000 | 0.0126 | 0.0013 | 1.0 | 0.0035 | 18.79–18.92s | g6e4xlarge `zsre_grace_gptj2k.log` **and** ohio `zsre_grace_gptj2k.log` — **two independent runs, near-identical (0.0126/0.0013 vs 0.0126/0.0013), fully consistent** |
| MEMIT | 2000 | 0.6146 | 0.5037 | 0.9951 | 0.6497 | 26.29s | ohio `zsre_memit_gptj2k.log` — **this is the one currently STILL RUNNING a second time on g6e4xlarge (`eval_zsre_we.py MEMIT`); the ohio result above already exists and is likely the one to cite unless the new run differs** |
| AlphaEdit | 2000 | 0.6345 | 0.5265 | 0.9966 | 0.6698 | 12.21s | ohio `zsre_alphaedit_gptj2k.log` — **note: an AlphaEdit zsRE retry2 is ALSO currently queued/running on g6e4xlarge behind MEMIT; this existing ohio number may get superseded or confirmed once that finishes** |
| CAKE / INLAY | n_tune=500,n_test=1500 | 1.0 | 1.0 | 1.0 | **1.0** | 14.20s | ohio `zsre_cake_gptj.log` — this is the "GPT-J zsRE ~1.00" number cited throughout the session |
| CAKE, n=20 smoke | n_tune=10,n_test=10 | 1.0 | 1.0 | 1.0 | 1.0 | 0.098s | ohio `zsre_smoke.log` — trivial N |

### B3. Qwen2.5-7B

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| base | 2000 | 0.3089 | 0.3045 | 1.0 | 0.3989 | 0.0s | virginia `zsre_base_qwen.log` |
| in_context (RAG) | 2000 | 0.9318 | 0.8554 | 0.83 | 0.8703 | 0.0s | virginia `zsre_rag_qwen.log` |
| ROME | 2000 | 0.9903 | 0.953 | 0.9828 | 0.9751 | 10.58s | virginia `zsre_rome_qwen2k.log` |
| WISE | 2000 | 0.9991 | 0.9971 | 0.9998 | 0.9987 | 31.12s | g6e4xlarge `zsre_wise_qwen2k.log` |
| GRACE | 2000 | 0.0504 | 0.0301 | 1.0 | 0.0555 | 8.99s | g6e4xlarge `zsre_grace_qwen2k.log` |
| MEMIT | 2000 | 0.6113 | 0.57 | 0.9925 | 0.6822 | 34.80s | g6e4xlarge `zsre_memit_qwen2k.log` |
| AlphaEdit | 2000 | 0.6133 | 0.5689 | 0.9922 | 0.6824 | 15.32s | g6e4xlarge `zsre_alphaedit_qwen2k.log` |
| CAKE / INLAY | n_tune=500,n_test=1500 | 0.9942 (headline) / 0.9999 (later confirming run) | 0.9949 / 0.9999 | 1.0 | 0.9963–0.9999 | 13.48–13.69s | virginia `zsre_cake_qwen.log` (0.9963) and reconfirmed in the Mistral ladder script's Qwen-adjacent smoke-test path — **treat 0.996–1.00 as the range, both runs agree closely** |
| CAKE, n=20 smoke | n_tune=10,n_test=10 | 1.0 | 1.0 | 1.0 | 1.0 | 0.12s | virginia `vsmoke2.log` — trivial N |

### B4. Mistral-7B-v0.3

| Method | N | ES | PS | NS | Score(hm) | Write cost | Source |
|---|---|---|---|---|---|---|---|
| CAKE / INLAY | n_tune=500,n_test=1500 | 0.9999 | 0.9999 | 1.0 | **0.9999** | 13.69s | ohio `ladder_zsre_mistral2k.log` |
| ROME | 2000 | 0.6306 | 0.5972 | 0.9856 | 0.7018 | 12.50s | ohio `ladder_zsre_rome_mistral2k.log` |

No MEMIT/WISE/GRACE/AlphaEdit zsRE results exist for Mistral. Same gap as CounterFact-Mistral.

---

## PART C — RippleEdits (compositional/multi-hop portability)

### C1. Non-matched-manifest run, GPT-J-6B, n=100 (earlier, less rigorous protocol — base/RAG/CAKE share an identical query set; ROME's set differs due to a subject-extraction heuristic filtering examples, so ROME's number here is NOT a fair comparison)

| Method | Logical_Gen | Comp_I | Comp_II | Subj_Alias | Rel_Spec | Forget | Propagation | Preservation | Aggregate |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.0333 | 0.0763 | 0.0 | 0.0 | 0.2217 | 0.1091 | 0.0274 | 0.1654 | 0.0734 |
| in_context (RAG) | 0.1037 | 0.3997 | 0.5308 | 0.758 | 0.3261 | 0.2364 | 0.4481 | 0.2812 | **0.3925** |
| CAKE | 0.0333 | 0.1363 | 0.241 | 0.8161 | 0.061 | 0.0364 | 0.3067 | 0.0487 | 0.2207 |
| ROME (caveat: different/filtered sample, indicative only) | 0.0765 | 0.0701 | 0.0 | 0.0 | 0.2348 | 0.0862 | 0.0367 | 0.1605 | 0.0779 |

Source: local `rippleedits.json`, ohio `ripple_base.log`/`ripple_cake.log`/`ripple_in_context.log`, ohio `EasyEdit/rome_ripple.log`.
**Explicit finding from the file itself:** CAKE is worst of all four on preservation (0.049) — it over-fires and destroys unrelated same-subject facts. This is the sourced basis for "the honest boundary" framing used throughout the project.

### C2. Matched-manifest run (rigorous — identical wikidata-verified subjects across all methods), n_edits=100, both GPT-J-6B and Qwen2.5-7B

| Model | Method | Propagation | Preservation | Aggregate |
|---|---|---|---|---|
| GPT-J-6B | base | 0.0258 | 0.1704 | 0.074 |
| GPT-J-6B | in_context (RAG) | 0.4566 | 0.2761 | **0.3964** |
| GPT-J-6B | cake | 0.3126 | 0.0507 | 0.2253 |
| Qwen2.5-7B | base | 0.0425 | 0.3716 | 0.1522 |
| Qwen2.5-7B | in_context (RAG) | 0.4533 | 0.4076 | **0.4381** |
| Qwen2.5-7B | cake | 0.3458 | 0.1699 | 0.2871 |

Source: virginia `rm_base_gptj.log`, `rm_rag_gptj.log`, `rm_cake_gptj.log`, `rm_base_qwen.log`, `rm_rag_qwen.log`, `rm_cake_qwen.log`.
**No ROME, MEMIT, WISE, GRACE, or AlphaEdit exist in the matched-manifest RippleEdits protocol, on any model.** Only base/RAG/CAKE were run through the rigorous matched version. This is a real, open gap if RippleEdits-vs-weight-editors is wanted.

Two near-duplicate CAKE GPT-J matched-RippleEdits runs exist and roughly agree: `rm_cake_gptj.log` (aggregate 0.2253) vs `rm_cake_smoke.log` (aggregate 0.0124, n=100 but clearly an early/broken low-signal run — **do not use**, superseded by `rm_cake_v2.log` which reports 0.2256, matching `rm_cake_gptj.log` closely) vs `rm_cake_v2.log` (0.2256). Treat 0.2253–0.2256 as the confirmed range.

---

## PART D — Sequential (multi-edit) editing, GPT-2-XL only

| N edits | CAKE retention | CAKE locality | CAKE score | CAKE cum_write_s | base retention/locality/score | RAG retention/locality/score | ROME retention/locality/score/cum_write_s |
|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 1.0 | 1.0 | 0.005 | 0/1.0/0 | 1.0/1.0/1.0 | 0.0/0.85/0.0/12.3s |
| 5 | 1.0 | 0.95 | 0.9744 | 0.024 | 0/1.0/0 | 1.0/1.0/1.0 | 0.0/0.60/0.0/54.2s |
| 10 | 1.0 | 0.90 | 0.9474 | 0.047 | 0/1.0/0 | 0.9/1.0/0.9474 | 0.0/0.30/0.0/100.5s |
| 25 | 1.0 | 0.90 | 0.9474 | 0.116 | 0/1.0/0 | 0.88/1.0/0.9362 | 0.0/0.05/0.0/222.5s |
| 50 | 1.0 | 0.85 | 0.9189 | 0.229 | 0/1.0/0 | 0.88/1.0/0.9362 | 0.0/0.00/0.0/343.0s |
| 100 | 1.0 | 0.70 | 0.8235 | 0.459 | 0.01/1.0/0.0198 | 0.92/1.0/0.9583 | 0.0/0.00/0.0/758.4s |
| 200 | 1.0 | 0.65 | 0.7879 | 0.919 | 0.005/1.0/0.01 | 0.935/1.0/0.9664 | not run past N=100 |
| 400 | 0.995 | 0.55 | 0.7084 | 1.842 | 0.005/1.0/0.01 | 0.94/1.0/0.9691 | not run past N=100 |

Source: local `sequential.json`, cross-checked against ohio `seq_cake.log`, `seq_base.log`, `seq_in_context.log`, `EasyEdit/seq_rome.log`.
**No sequential-editing runs exist for GPT-J, Qwen, or Mistral, and no sequential MEMIT/WISE/GRACE/AlphaEdit exist at all.** This entire benchmark is GPT-2-XL + {base, CAKE, RAG, ROME(≤100 only)} only. A genuine gap if the paper wants sequential editing at the larger model scales.
Finetune sequential was explicitly not run (noted in the file: "GPU was occupied by the user's WISE sweep").

---

## PART E — Ablations / probes (secondary, not headline numbers, listed for completeness)

- **Key-geometry probe** (`EasyEdit/probe_geom.log`, local `probe_geom` inside `comp_probes.json` not separately checked): raw hidden-state key h_L gives separation −0.035 (paraphrase-vs-neighbor, WORSE than random) vs MiniLM(prompt) +0.288 vs MiniLM(subject) +0.358 — this is the sourced justification for switching to a semantic key.
- **Alpha sweep, GPT-J** (`gptj_alpha.log`): alpha=10 → ES 0.65/PS 0.5; alpha≥20 saturates at ES 1.0/PS 0.825.
- **Alpha sweep, Qwen** (`probe_qwen.log`): alpha=10 → ES 0.125/PS 0.1625 (weak); alpha≥40 saturates at ES 1.0/PS 0.8375.
- **Margin-gate sweep** (`margin_curve.json`/`margin2.log`, GPT-2-XL, N=400): margin=0 gives locality 0.5667 (score 0.7221); margin=0.15 gives locality 1.0 (score 0.9975) — margin gate is a genuine, measured locality fix at scale, though (per the code-audit earlier this session) it is a no-op in the single-edit-per-example protocol where only one slot is ever occupied; it only matters in multi-slot regimes like this sequential/multi-fact test.
- **Relation-gate sweep** (`relgate.log`, `relgate2.log`, `cf_relsweep.log`, `cf_rel02.log`, `cf_relgate.log`): rel_gate=0 → over-fire 0.9667, propagation 1.0; rel_gate=0.2 → propagation stays 1.0 but overfire drops toward 0.67; rel_gate=0.5 → overfire 0.1 but propagation collapses to 0.4667. This is the sourced basis for the "0.97→0.67 over-fire, cost is CF-PS 0.87→0.66" claim used in outreach materials — confirmed present in the logs (`cf_rel02.log` shows CF-PS 0.63 at rel_gate=0.2, n_test=50, close to the "0.66" figure cited; minor discrepancy likely due to different N/seed between the outreach-doc citation and this specific log — flag for the parent to reconcile if exact figure matters).
- **24-fact multi-subject document test** (`multi_subject.json`/`geom_comp.log`/`multi_soft.log`): cake efficacy 1.0/generalization 0.25/locality 1.0/write_s 0.079–0.537 vs finetune efficacy 1.0/generalization 0.833/locality 0.1667/write_s 6.47–123s (catastrophic forgetting) vs in_context efficacy 1.0/generalization 0.542/locality 1.0.
- **Portability probe** (`port_cake.log`): cake n=100, ES 1.0, portability 1.0, n_gen_probes=1000 — a narrower, easier portability metric than RippleEdits; do not conflate with Part C's numbers.

---

## PART F — Attempted but FAILED (no valid number, listed so nothing looks silently missing)

| Attempt | Host | Log | Failure |
|---|---|---|---|
| WISE-Qwen via `eval_cf_dualmetric.py` (1st try) | g6e4xlarge/ohio | (overwritten, no longer on disk) | `KeyError: 'loc_prompt'` — WISE needs `loc_prompt` in the request dict, script didn't provide it |
| WISE-Qwen via `eval_cf_dualmetric.py` (2nd try, after loc_prompt patch) | ohio | `cf_wise_qwen2k.log` intermediate crash (superseded by final success) | `ValueError: You must specify exactly one of input_ids or inputs_embeds` — a WISE-internal masking bug in this script path unrelated to loc_prompt; resolved by switching to `eval_cf_we.py` instead (see A3 WISE v2, the successful run) |
| AlphaEdit-Qwen, original attempt (weeks-old cache) | g6e4xlarge | (overwritten) | Disk-full corruption of `null_space_project.pt` during a 3.86GB blob write when the 25GB root volume was 100% full; fixed by moving HF_HOME to the 549GB `/mnt/scratch` NVMe |
| AlphaEdit-GPT-J zsRE, first relaunch attempt | g6e4xlarge | `zsre_alphaedit_gptj2k.log` (original, pre-Aug-16, this one succeeded — see B2) / then a same-day RE-relaunch crashed | `OSError: Repo id must be in the form ... './hugging_cache/gpt-j-6B'` — missing symlink from the HF cache snapshot dir to the `hugging_cache/` convention EasyEdit expects; fixed by creating the symlink |
| GRACE-GPT-J zsRE, same relaunch batch | g6e4xlarge | same missing-symlink error | same root cause, same fix |
| AlphaEdit-GPT-J zsRE retry (2nd attempt) | g6e4xlarge | crashed with CUDA OOM | A buggy `pgrep` wait-condition (regex alternation not supported by this pgrep) let AlphaEdit launch while GRACE (23.4GB resident) was still running on the same 44GB GPU; fixed by using two separate `pgrep` checks; **retry2 is the currently-running job, no result yet** |
| Llama-3-8B, any run | ohio | `smoke_llama.log` | `403 Client Error: gated repo, not in authorized list` — the HF token's account never got Llama-3 access approved; abandoned in favor of Mistral-7B as the third model family |

---

## Summary: what a clean final comparison table should use (my recommendation to the parent)

For the headline "all methods, all models, CounterFact + zsRE" table, the highest-N / most-recent / bug-free numbers are:

**CounterFact (N=2000 for weight-editors, N=5000 for CAKE/base/RAG — the project's established convention):**
- GPT-J: CAKE 0.8926 (or 0.8957 at N=1000, essentially flat) · ROME 0.797 · WISE 0.703 · AlphaEdit 0.4595 · MEMIT 0.4314 · GRACE 0.0 · RAG ~0.546 (N=1000) · base ~0.005–0.008
- Qwen: CAKE 0.8944 · WISE 0.9466 (**wins on Qwen specifically**) · ROME 0.7767 · GRACE ~0.0 · RAG 0.5258 · base 0.016 · (MEMIT/AlphaEdit only exist in the non-comparable dualmetric prob-success framing — flag as such, don't force into the hm column)
- Mistral: CAKE 0.8995 · ROME 0.3398 (weaker — untuned hyperparameters, as already noted in this project's outreach docs) · no MEMIT/WISE/GRACE/AlphaEdit exist

**zsRE (N=2000):**
- GPT-J: CAKE 1.0 · WISE 0.9959 · ROME 0.937 · AlphaEdit 0.6698 · MEMIT 0.6497 · GRACE 0.0035 · RAG 0.8757 · base 0.2968
- Qwen: CAKE 0.9963–0.9999 · WISE 0.9987 · ROME 0.9751 · AlphaEdit 0.6824 · MEMIT 0.6822 · GRACE 0.0555 · RAG 0.8703 · base 0.3989
- Mistral: CAKE 0.9999 · ROME 0.7018 · no MEMIT/WISE/GRACE/AlphaEdit exist

**Write cost (per-edit, GPU-normalized, roughly consistent across the L40S hosts):** CAKE/INLAY ~5–15ms · GRACE ~7–19s · ROME ~8–10s · AlphaEdit ~12–16s · WISE ~15–31s · MEMIT ~25–35s.

**Real open gaps, not yet filled by any run:** AlphaEdit CounterFact GPT-J at N=2000 (running now), MEMIT/WISE/GRACE/AlphaEdit on Mistral (any benchmark), MEMIT/WISE/GRACE/AlphaEdit in the matched-manifest RippleEdits protocol (any model), sequential/lifelong editing at any model besides GPT-2-XL for any method, and a true harmonic-mean-comparable MEMIT/AlphaEdit-on-Qwen CounterFact number (only the prob-success variant exists).

**Total distinct results found and tabulated: 91** (counting each method×model×benchmark×N combination once; excludes the two still-running jobs and the 7 failed/crashed attempts listed in Part F, which are tracked separately for completeness).
