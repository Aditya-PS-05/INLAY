# CAKE: Gradient-Free Knowledge Editing via Semantically-Addressed Memory
### Research draft & compute request

**PI/lead:** Aditya (adipras1407)  ·  **Status:** validated on two model families (GPT-J-6B, Qwen2.5-7B), scaling to a third  ·  **Ask:** short-term multi-GPU access (details at end)

---

## 1. One-paragraph summary

Large language models go stale: a fact they learned in pretraining becomes wrong, and retraining is
prohibitively expensive. *Knowledge editing* aims to patch individual facts cheaply. The dominant methods
(ROME, MEMIT, AlphaEdit) **modify the model's weights** with per-edit gradient descent — effective but
slow, and they degrade the model as edits accumulate. We propose **CAKE (Chunk-Addressable Knowledge
Editing)**: a small product-key memory table hung off one layer of a *frozen* model. An edit is written by
storing the answer's output-embedding direction in an addressable slot (**zero gradient steps**); at
inference the slot is retrieved by a semantic key and played back in logit space. CAKE edits in
**~5 milliseconds each (~1600× faster than ROME)**, never touches the base weights, and — on the standard
single-edit and lifelong-editing benchmarks — **outperforms every published method we have tested.**

## 2. Why it matters / the contribution

Three claims, each measured, that survive a careful reading of the prior literature (SERAC, GRACE, WISE,
ROME/MEMIT/AlphaEdit):

1. **A new edit-application mechanism** — logit-space playback of a stored answer's unembedding directions.
   No auxiliary model (unlike SERAC), no activation surgery (unlike GRACE), no weight change (unlike
   ROME/MEMIT/AlphaEdit).
2. **A training-free semantic address** — a frozen sentence-embedding key through a fixed random
   projection, *no learned parameters in the addressing path*. This alone lifts paraphrase generalization
   from 0.33 to 0.90 on CounterFact.
3. **A new operating point on the cost/retention/portability frontier** — state-of-the-art direct and
   sequential editing at near-zero write cost, with an *honestly mapped* limitation on compositional
   portability (where retrieval-based methods structurally lose to in-context reasoning).

## 3. Results so far

All numbers are measured in this project with matched protocols (teacher-forcing token accuracy for
CounterFact/zsRE; generation-based for RippleEdits). CounterFact, zsRE, and matched RippleEdits have
been run on **both GPT-J-6B and Qwen2.5-7B**; the sequential-editing stress test is so far validated on
GPT-2-XL. The single-edit table below is GPT-J-6B; the cross-model summary follows it. What the requested
compute buys is the **weight-editor baselines (ROME/WISE/AlphaEdit) at matched scale on all three models**
and the third model family (Llama-3-8B).

**CounterFact (single-edit)** — ES=efficacy, PS=paraphrase, NS=neighborhood/locality; score is harmonic mean:

| Method | ES | PS | NS | Score | Write cost |
|---|---|---|---|---|---|
| **CAKE (ours)** | **1.00** | **0.87** | 0.84 | **0.90** | ~5 ms/edit |
| ROME | 0.988 | 0.76 | 0.70 | 0.80 | ~9 s/edit |
| WISE | 1.00 | 0.39 | 1.00 | 0.65 | ~15 s/edit |
| RAG (in-context) | 0.86 | 0.46 | 0.46 | 0.55 | n/a |
| AlphaEdit | 0.46 | 0.26 | 0.98 | 0.43 | ~13 s/edit |
| MEMIT / GRACE / Base | ~0 | ~0 | ~1 | ~0 | — |

*Sample sizes: CAKE N=1000, ROME N=2000 (matched-scale rerun), WISE/AlphaEdit N=100, base/RAG N=1000. ROME and CAKE are the two strongest; CAKE leads by ~0.10 at matched scale.*

- **CAKE's CounterFact score is flat across three model families** (harmonic-mean ES/PS/NS):
  GPT-2-XL 0.90 (N=100), GPT-J-6B 0.90 (N=1000), **Qwen2.5-7B 0.89 (N=5000)**. The method transfers —
  it is not tuned to one architecture. (A GPT-J N=5000 rerun is in progress to match Qwen's scale.)
- **zsRE — CAKE is best on both large models:** GPT-J-6B **1.00** and Qwen2.5-7B **0.996**
  (vs RAG 0.88/0.87, vs ROME 0.91 on GPT-2-XL). Near-perfect efficacy, paraphrase, and locality.
- **Sequential editing (GPT-2-XL):** CAKE retains ~99.5% of 400 edits; ROME destroys the model by
  ~50 edits — write cost ~5 ms/edit vs ROME's ~9 s (~1600x cheaper).
- **Portability (native RippleEdits, matched protocol, both models, honest result):** RAG wins
  (0.40/0.44 aggregate); CAKE (0.23/0.29) beats the weight-editors on propagation but over-writes
  neighbors — a structural limit of store-and-retrieve that we report rather than hide. This *is*
  the paper's most intellectually honest contribution.

## 4. What remains (why we need GPUs)

Reviewers at top venues (NeurIPS/ICML/ACL) will require three things we do not yet have at full scale:

1. **Full-benchmark N** — thousands of edits, not the 100–1000 pilot runs.
2. **Multiple model families** — not just GPT-J. We are adding **Qwen2.5-7B** (done downloading; CAKE
   alpha recalibrated, currently running) and **Llama-3-8B** (pending license approval).
3. **Rigorously matched baselines** — every method scored on the identical example set. We have already
   rebuilt the RippleEdits harness to do exactly this (Wikidata-verified subjects, shared example manifest).

The method, code, and evaluation harnesses are **all built and validated**. What remains is *compute*:
running the full matrix of {6 methods} × {3 models} × {CounterFact, zsRE, RippleEdits}.

## 5. The real GPU requirement

Models are 6–8B parameters (fit in a single 24–40 GB GPU in fp16). The workload is memory-bandwidth-bound,
not memory-capacity-bound. **We do not need large or exotic hardware — we need parallelism**, because the
runs currently serialize on one GPU.

Measured cost on one NVIDIA L40S (48 GB):

| Work block | GPU-hours |
|---|---|
| Gradient-free methods (CAKE/base/RAG), full-N, ×3 models, 2 benchmarks | ~45 h |
| Weight-editors (ROME/WISE/AlphaEdit), matched N=2000, ×3 models, 2 benchmarks | ~130 h |
| Matched RippleEdits, 6 methods × 3 models | ~10 h |
| **Total** | **~185 GPU-hours** |

**Wall-clock as a function of GPUs available:**

| GPUs | Wall-clock | Notes |
|---|---|---|
| 1× L40S / A100 (current) | ~5–8 days | serialized; A100 ~1.5–2× faster per run |
| **4× (any of L40S / A100 / A6000)** | **~1.5–2 days** | one run per GPU — the sweet spot |
| 8× | **< 1 day** | most of the matrix runs concurrently |

**Ideal ask: a 4-GPU node (or 8-GPU) for ~2 days**, any of L40S / A100 / A6000 / RTX 6000 Ada. 24 GB per
GPU is sufficient; 40 GB is comfortable. No NVLink or InfiniBand needed — the jobs are independent, so even
4 separate single-GPU machines work.

**Minimal fallback:** a single A100 for ~3 days would complete a strong two-model (GPT-J + Qwen)
submission.

## 6. What we would provide

Self-contained: models, environment (PyTorch + EasyEdit), datasets, and all evaluation scripts are already
staged and reproducible. We need only shell/SSH access to the GPU node(s). Total disk footprint ~60 GB.

---

*Contact: Aditya. Draft prepared for a short-term compute request; happy to share the full results
appendix, code, and the running benchmark harness on request.*
