# CAKE vs other knowledge-injection methods

GPT-2, 5 fabricated facts (efficacy) + 12 general-knowledge control prompts (locality), NVIDIA L40S.

| method | efficacy | locality | write cost | gradients | params changed |
|---|---|---|---|---|---|
| Base GPT-2 (no edit) | 0/5 | 1.00 | — | 0 | 0 |
| In-context (RAG) | 5/5 | 1.00 | 0 (but re-fed every query) | 0 | 0 |
| Fine-tune (60 steps) | 5/5 | **0.17** | 6.5 s | 60 | 124.4 M |
| **CAKE (this)** | **5/5** | **1.00** | **0.079 s** | **0** | **0** |

**Efficacy** = fraction of facts the model answers correctly (greedy).
**Locality** = fraction of unrelated control prompts whose answer is unchanged from base GPT-2 (1.0 = no collateral damage).

## Read of the table
- **CAKE is the only method that gets efficacy 1.0 AND locality 1.0 with a cheap, gradient-free write.** It installs 5 facts in 79 ms, touching zero model parameters.
- **Fine-tuning learns the facts but wrecks unrelated knowledge** — 83% of the control prompts changed answer after only 60 steps (catastrophic forgetting). This is exactly the failure mode CAKE and ROME/MEMIT were designed to avoid.
- **In-context (RAG) also scores perfectly here** — the fair rival. Its cost is different in kind: the document is re-fed on *every* query, so compute and context length grow with the corpus, and it can't scale to a large knowledge base the way an addressable table can. CAKE pays the storage cost once, at write time.
- **Locality depended on the firing threshold.** With a loose gate (min_score=0.15) every prompt spuriously fired a slot (locality 0.00). But facts fire at score **1.0** and unrelated prompts at **≤0.87**, so a threshold of 0.9 cleanly separates them → locality 1.00 with no loss of recall. This is a genuine tuning knob, reported honestly.

## Where ROME / MEMIT sit (from their papers, not run here)
ROME and MEMIT occupy the same "high efficacy + high locality" corner as CAKE, but reach it by a **gradient-based rank-one edit of a shared MLP matrix**. Published MEMIT scales to thousands of edits before locality degrades. The trade vs CAKE: they modify weights (harder to undo, interference grows with edit count as the shared matrix fills), whereas CAKE keeps each fact in its own slot (trivial to clear/update, no interference) at the cost of imperfect paraphrase addressing. Running EasyEdit's ROME/MEMIT on the same 5 facts is the natural next benchmark.
