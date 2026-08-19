# CAKE at the 6B tier — GPT-J-6B CounterFact

The GPT-2-XL results left one central question for the "state of the art" claim: does CAKE's win hold
on a standard, larger model, or was it an artifact of GPT-2-XL? This ports CAKE to **GPT-J-6B** — the
same 6B model the knowledge-editing literature (and this project's WISE sweep) argues over — and re-runs
CounterFact against the same baselines.

## Porting cost: one line

The semantic-key redesign made CAKE model-agnostic. The addressing key is a MiniLM sentence embedding,
not a hidden state, so the only model surfaces CAKE touches are `config.hidden_size` and
`lm_head.weight` — shared by GPT-2, GPT-J, and Llama. The port required:
- swapping `GPT2LMHeadModel`/`GPT2TokenizerFast` for `AutoModelForCausalLM`/`AutoTokenizer` (load in fp16),
- **rescaling alpha 10 → 20** to match GPT-J's unembedding magnitude (its `W_U` rows have mean norm 1.31;
  a quick alpha probe showed ES jumps 0.65 → 1.00 at alpha ≥ 20, then plateaus).

**No architectural change, no layer retuning, no gradient.** That is itself a result: weight-editors need
per-model layer selection and (for MEMIT) per-model covariance statistics; CAKE needs a scalar rescale.

## Results — CounterFact single-edit, GPT-J-6B (N=100)

| method | ES | PS | NS | **score** | write | grad |
|---|---|---|---|---|---|---|
| Base | 0.02 | 0.00 | 1.00 | 0.000 | — | 0 |
| In-context (RAG) | 0.84 | 0.42 | 0.45 | 0.515 | 0 | 0 |
| ROME | 0.995 | 0.755 | 0.60 | 0.751 | 8.6 s | 20 |
| **CAKE v3** | **1.00** | **0.85** | **0.828** | **0.886** | ~0 | **0** |

CAKE gate 0.45 (selected on a held-out tune split); ROME layer 5, 20 grad steps, no covariance.

**Among methods that actually edit, CAKE leads every axis and the combined score (0.886 vs ROME 0.751)
on GPT-J-6B** — the same ordering as on GPT-2-XL. (Base shows NS=1.0 trivially because it makes no edit;
its ES=0.02 makes it a non-editor floor.)

## The win scales

| model | CAKE ES | PS | NS | CAKE score | ROME score |
|---|---|---|---|---|---|
| GPT-2-XL (1.5B) | 1.00 | 0.86 | 0.859 | **0.902** | 0.665 |
| GPT-J-6B (6B) | 1.00 | 0.85 | 0.828 | **0.886** | 0.751 |

CAKE's score is essentially flat across a 4× model-size jump (0.902 → 0.886). ROME actually improves at
6B (0.665 → 0.751) — GPT-J's cleaner MLP structure suits ROME — but still trails CAKE by ~0.13. The
GPT-2-XL result was not an artifact; **the method transfers.**

## What this does and does not settle for "state of the art"

**Settles:** CAKE's advantage is not model-specific. On two standard models, single-edit CounterFact,
against ROME/RAG/base with an honest held-out protocol, CAKE wins — gradient-free, non-destructive,
one scalar to retune.

**Still open** (the remaining SOTA gates, unchanged by this run):
- **Competitor set.** ROME is beaten, but the current *sequential/lifelong* editors — WISE, GRACE,
  AlphaEdit — are CAKE's real conceptual rivals and are still unrun. EasyEdit ships GPT-J-6B hparams for
  WISE and GRACE, so this is the natural next step.
- **Portability / ripple.** CAKE plays back stored answer tokens; it retrieves rather than re-derives.
  CounterFact ES/PS/NS don't test whether an edit propagates to *related* facts (edit "Paris→Germany",
  ask about the Eiffel Tower's country). This is where a retrieval method is expected to be weakest and
  it is not yet measured.
- **Full benchmark size.** N=100 sample vs the published 10k.

## Files
- `eval_cf_gptj.py` (CAKE, held-out gate, fp16), `eval_cf_edit_gptj.py` (ROME native metrics),
  `eval_cf_baseline_gptj.py` (base/RAG), `probe_alpha_gptj.py` (alpha rescale probe)
- `gptj_counterfact.json` / `gptj_counterfact.csv`
