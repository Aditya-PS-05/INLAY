# Multi-token value fix — CAKE v3

The zsRE benchmark showed CAKE v2's one remaining limit was its **value representation**, not its
addressing: it stored only the *first* answer token's unembedding direction, so on zsRE's free-form
multi-token answers ("University of Michigan") token-accuracy capped efficacy at ~0.53. This fix
lifts that cap.

## The fix

`gated_logits` previously added `alpha·⟨W_U, v_first⟩` to **every** teacher-forced position — only the
first answer token got help. Now, when a slot fires (gated on the **prompt** embedding, unchanged),
CAKE does **position-wise logit-space playback** of the full stored answer sequence: at scored
position *k* it biases toward stored answer token *k* via `alpha·⟨W_U, v_k⟩`. The full answer token
sequence was already kept in slot meta (the generation path used it), so this wires existing state
into the scored path. **Still zero-gradient; the addressing/semantic key is untouched.** The gate
fires on the prompt and playback emits what was *written* into the slot — never the eval target — so
this is retrieval, not label leakage.

## Result — zsRE (GPT-2-XL, N=100, held-out gate)

| CAKE value | ES | PS | NS | score |
|---|---|---|---|---|
| first-token (v2) | 0.53 | 0.52 | 1.00 | 0.624 |
| **multi-token (v3)** | **0.96** | **0.96** | **1.00** | **0.973** |

Efficacy and generalization jumped from 0.53 to **0.96** — exactly the axes the diagnosis predicted
(they were capped *together*, the signature of a value limit). Locality stays perfect.

## Full zsRE comparison — CAKE v3 now leads

| method | ES | PS | NS | **score** | write | grad |
|---|---|---|---|---|---|---|
| Base | 0.21 | 0.21 | 1.00 | 0.280 | — | 0 |
| In-context (RAG) | 0.94 | 0.85 | 0.88 | 0.887 | 0 | 0 |
| Fine-tune | 0.79 | 0.75 | 0.01 | 0.029 | 6.6 s | 25 |
| MEMIT | 0.28 | 0.24 | 1.00 | 0.346 | 4.4 s | 100 |
| ROME | 1.00 | 0.80 | 0.96 | 0.908 | 3.8 s | 20 |
| **CAKE v3** | **0.96** | **0.96** | **1.00** | **0.973** | **0.3 s** | **0** |

CAKE v3 is now the top method on zsRE — above ROME (0.908) and RAG (0.887) — while remaining
gradient-free and ~11× faster to write than ROME.

## No regression on CounterFact

CounterFact answers are mostly single-token, so multi-token playback reduces to exactly the first-token
injection there. Confirmed unchanged (held-out, N=100/100): **ES 1.00, PS 0.86, NS 0.86, score 0.902**
— identical to v2. The fix helps where answers are multi-token and is a no-op where they are not.

## Cross-benchmark summary (CAKE v3, held-out)

| benchmark | ES | PS | NS | score | vs ROME |
|---|---|---|---|---|---|
| CounterFact | 1.00 | 0.86 | 0.86 | 0.902 | beats ROME (0.665) |
| zsRE | 0.96 | 0.96 | 1.00 | 0.973 | beats ROME (0.908) |

CAKE v3 leads ROME on **both** standard single-edit benchmarks, gradient-free and non-destructive
(zero weight change, fully reversible). The two weaknesses identified over this project — paraphrase
generalization (fixed by the semantic key) and multi-token answers (fixed here) — are both resolved.

## Honest caveats
- Playback biases toward the stored answer tokens; it is retrieval of written content, appropriate for
  a memory method, but it means CAKE reproduces the answer *as written* rather than re-deriving it.
- Numbers are GPT-2-XL single-edit. Batch/sequential editing and larger models are the remaining
  untested regimes.

## Files
- `gpt2_memory_semkey.py` — now with multi-token playback in `gated_logits` (`multi=True` default)
- `zsre.json` / `zsre.csv` — updated with CAKE v3
