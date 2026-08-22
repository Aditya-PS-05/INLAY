# INLAY — research state as of 2026-08-15 (synced from GPU hosts)

Local repo was last synced Jul 6. This doc consolidates the newer results found on the EC2 hosts
(g6e4xlarge, london-g6e2xlarge, virginia-g6e2xlarge). All GPUs idle at time of survey — nothing running.

## Where the work lives

| Host | GPU | Workspace | Latest activity | What's there |
|---|---|---|---|---|
| g6e4xlarge | L40S | ~/inlay | **Aug 5** | Qwen2.5-7B baselines at N=2000: WISE/GRACE/MEMIT done, AlphaEdit **crashed** |
| london-g6e2xlarge | **L4 (24 GB!)** | ~/inlay | **Aug 2** | GPT-J CF at N=5000 (INLAY/base/RAG done), INLAY dual-metric N=2000; GRACE & ROME GPT-J runs **OOM'd here** (L4 too small) |
| virginia-g6e2xlarge | L40S | ~/inlay | Jul 7 | ROME N=2000 (both models, CF+zsRE), WISE zsRE (both), matched RippleEdits (base/RAG/INLAY × both models) |
| ohio-g6e2xlarge | L40S | ~/EasyEdit | Jul 2 | early baseline scratch, superseded |
| g6e8large | L40S | (no inlay) | Jul 7 | unrelated projects (gqa, suppression-recovery) |
| sunny | L40S | not checked (stopped) | — | original pre-seed/GPT-2 work, presumably |

## Consolidated results (EasyEdit-native token-accuracy protocol, harmonic-mean score)

### CounterFact
| Method | GPT-J-6B | Qwen2.5-7B | write cost |
|---|---|---|---|
| **INLAY** | **0.893** (N=5000, held-out gate 0.45; ES .998/PS .871/NS .826) | **0.89** (N=5000, from Jul runs) | ~5–14 ms |
| ROME | 0.80 (N=2000, Jul) | 0.777 (N=2000; ES .997/PS .554/NS .949) | ~9 s |
| MEMIT | ~0 (default config under-installs) | **~0.876** (N=2000; token acc ES .998/PS .714/NS-keep .978) | ~34 s |
| RAG | 0.541 (N=5000) | — | — |
| GRACE | OOM at 2k (only N=100 ≈ 0) | 0.0 (N=2000) | ~7 s |
| base | 0.008 (N=5000) | — | — |

Note: **MEMIT is competitive on Qwen** (0.876 vs INLAY 0.89) — the "MEMIT ~0" result is a GPT-J
config artifact. INLAY still leads CF on both models, but the Qwen margin is narrow.

### zsRE (N=2000) — effectively saturated at the top
| Method | GPT-J-6B | Qwen2.5-7B | write cost |
|---|---|---|---|
| INLAY | 1.00 (Jul) | 0.996 (Jul) | ~5 ms |
| WISE | **0.996** | **0.9987** | 17–31 s |
| ROME | 0.937 | 0.975 | ~10 s |
| MEMIT | — | 0.682 | ~35 s |
| GRACE | — | 0.056 | ~9 s |

INLAY and WISE are statistically tied at ceiling; INLAY's differentiation on zsRE is **write cost**
(~3–4 orders of magnitude), not accuracy.

### RippleEdits, matched manifest (100 edits, wikidata-verified subjects)
| | GPT-J aggregate | Qwen aggregate |
|---|---|---|
| RAG (in-context) | **0.396** | **0.438** |
| INLAY | 0.225 | 0.287 |
| base | 0.074 | 0.152 |

Per-criteria: INLAY is strong on Subject_Aliasing (0.83 both models) but weak on Relation_Specificity
and preservation — consistent with the known over-firing boundary. RAG wins portability, as expected.

### INLAY dual-metric run (GPT-J N=2000, Aug 2, london)
ES .998 / PS .893 (prob-success); .9975 / .8655 (token-acc); NS keep .834; write 13.5 ms.

## Failed / missing runs (the actual remaining work)
1. **AlphaEdit on Qwen (CF + zsRE)** — crashed on g6e4xlarge: `torch.save(null_space_project.pt)`
   failed mid-write ("unexpected pos 704 vs 598") then reload of the corrupted file. Looks like
   **disk-full** during the null-space projector save. Check `df -h`, delete the corrupt
   `null_space_project.pt`, rerun.
2. **GRACE GPT-J CF at N=2000** — OOM'd on london's **L4 (22 GB)**. Rerun on an L40S host
   (virginia/g6e4xlarge). (london's zsre_rome_gptj2k OOM was a duplicate of a run that succeeded
   on virginia — ignore.)
3. **WISE CounterFact at N=2000** (both models) — no completed log found; only zsRE WISE at 2k.
   CF WISE exists only at N=100 (0.65, GPT-J).
4. **Llama-3-8B** — third model family, not started anywhere.
5. **Consolidation** — nothing after Jul 6 has been merged into the local repo's summaries.

## Practical notes
- Do NOT run GPT-J weight-editor baselines on london — it's a g6.2xlarge with an L4 (24 GB), not an
  L40S. It's fine for INLAY/base/RAG (which fit) — that's why the N=5000 INLAY run succeeded there.
- All five started instances were idle at survey time — stop g6e8large and ohio (nothing needed
  there), and stop the rest when not actively running jobs.
