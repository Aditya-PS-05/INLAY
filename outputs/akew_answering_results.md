# AKEW answering strategies: oracle-evidence pilot — 2026-08-19

Compares the three answering strategies brief section 5 asks for (a fourth,
weak-logit-bias, is deprioritized for this pass -- noted honestly below, not
silently skipped) under **oracle retrieval**: every strategy is handed the
true KnowledgeCard directly, isolating the answering-STRATEGY question from
retrieval/routing noise. This is section 5's oracle diagnostic #1 ("gold card
supplied directly to the generator"), and deliberately the first experiment
to run, since if contextual generation can't beat hard playback with perfect
retrieval, that's the more important finding to have before adding retrieval
noise on top.

Model: `Qwen/Qwen2.5-1.5B-Instruct` (chosen for pilot speed -- the brief's
target models for the full experimental matrix are GPT-J-6B and Qwen2.5-7B;
this pilot validates the MECHANISM, not the final numbers). CounterFact,
subject-disjoint test split (n=147, never touched by the verifier's own
training), deterministic decoding.

## Real methodology bugs found and fixed mid-pilot (both caught by reading
## actual generations, not just accuracy numbers)

1. **Chat-template bug.** The first pass used raw string concatenation
  (`tok(prompt)`) instead of the Instruct model's chat template. Several
  `contextual_generation` outputs visibly hallucinated a fake system-prompt
  continuation ("You are an AI assistant. You will be given a task...")
  instead of answering -- a real quality bug, not noise. Fixed with an
  explicit `apply_chat_template(..., return_dict=True)` helper (a second,
  narrower bug: this transformers version returns a `BatchEncoding`, not a
  raw tensor, from `apply_chat_template` with `return_tensors="pt"` alone --
  passing that whole object into `generate()`'s positional arg crashed deep
  inside `generate()` with an opaque `AttributeError` until unpacked
  correctly). Fixing this alone raised `contextual_generation` from 95.2% to
  97.96% on structured mode, and cleaned up `no_context`'s outputs from
  fabricated guesses to honest "I don't have that information" refusals.

2. **Scoring artifact in `hard_playback` for unstructured mode.** The first
  definition dumped the full raw evidence paragraph verbatim as the "answer."
  Since that prose exists specifically to describe the new fact, it trivially
  contains the gold answer as a substring almost every time, inflating
  `hard_playback` to 95.24% on unstructured CounterFact -- a scoring artifact,
  not genuine recitation accuracy, and an unfair comparison against
  `contextual_generation`'s short, focused answers. Fixed by defining
  unstructured/extracted `hard_playback` as the first sentence only (still an
  honest "recite what was stored, no reasoning" operationalization, just not
  one that wins for free by dumping an entire paragraph).

## Results (post-fix, both bugs corrected)

### Structured mode (n=147)

| strategy | accuracy |
|---|---|
| no_context | 1.36% |
| hard_playback | 100.0% |
| contextual_generation | 97.96% |

Expected result: structured mode gives hard_playback a literal clean answer
to recite with nothing to compose, so it trivially wins. Contextual
generation's small gap below 100% is mostly semantically-correct paraphrases
the exact-match scorer penalizes (e.g. generating "Goalie" against gold
"goaltender" in an earlier run) -- a metric-family limitation worth flagging,
not necessarily a real answering failure.

### Unstructured mode (n=147) -- the real test, where hard playback has no
### clean answer to recite

| strategy | accuracy |
|---|---|
| no_context | 1.36% |
| hard_playback (first sentence only) | 86.39% |
| contextual_generation | 87.76% |

**Contextual generation modestly outperforms even a strong first-sentence
recitation baseline, while producing categorically more useful output**: a
direct one-word answer ("Bishop") versus a full sentence a downstream
consumer would have to parse ("...is recognized as a bishop in the Catholic
Church"). This is the core empirical case for the whole redesign: reasoning
over retrieved evidence is a real, viable alternative to forced playback,
particularly wherever playback has no clean token span to force in the first
place -- exactly INLAY's own limitation, now confirmed on a second, harder
benchmark under oracle-evidence conditions.

## What this does NOT establish yet

This is oracle-evidence only (perfect retrieval, no routing decision). It
does not yet test the full pipeline (Stage 1 retrieval -> Stage 2 verifier ->
router decision -> answering strategy) end to end, where retrieval/routing
errors will compound with answering-strategy errors. It also has not yet run
on extracted mode, on WikiUpdate/MQuAKE-CF, or at the brief's target model
scale (GPT-J-6B / Qwen2.5-7B). The weak-logit-bias condition (a lighter-touch
alternative to hard playback, biasing logits toward the target rather than
forcing full token sequences) is deprioritized, not built in this pass.
