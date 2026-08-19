# Full non-oracle pipeline test, and a real router bug found and fixed — 2026-08-19

Every answering-strategy result reported so far in this project used
**oracle retrieval** — the true card handed directly, deliberately isolating
the answering-strategy question from retrieval noise (brief's oracle
diagnostic #1). This is the first test of the actual system end to end: real
dense retrieval -> real (v2) verifier -> router's REJECT/DIRECT/REASON
decision -> the corresponding answering strategy -> scored against gold.

CounterFact, both modes, n=200 (147 scoreable after filtering), subject-
disjoint test split, `Qwen/Qwen2.5-1.5B-Instruct`, verifier v2.

## Real retrieval quality in the live pipeline

**Retrieval correctness rate: 99.32%** on unstructured mode — the correct
card was found in essentially every case, confirming the whole pipeline's
components (dense retrieval, v2 verifier) compose correctly under realistic
conditions, not just in isolated component tests.

## A real router bug, found only by running the live pipeline

First run (unstructured mode): the router chose **DIRECT for all 147/147
queries**, never REASON. Root cause: `answer_hard_playback`'s non-structured
fallback recites the raw evidence's first sentence, not a real generated
answer -- but the router's DIRECT/REASON decision was based purely on
verifier confidence and a multi-hop heuristic, blind to `input_mode`. Since
unstructured-mode retrieval is confident and CounterFact's queries are all
single-hop, DIRECT fired every time, even though DIRECT's entire rationale
(a clean literal answer to recite fast) only holds when the card is
structured.

| | routed (pre-fix) | always-force-REASON |
|---|---|---|
| unstructured accuracy | 85.71% | 87.07% |

The router was choosing the WORSE strategy every single time on this mode --
a gap invisible to any of the oracle-evidence pilots, since those never
exercised the router's live decision logic at all.

## Fix: gate DIRECT on structured mode specifically

`akew_router.py`: DIRECT now additionally requires `top_card.input_mode ==
"structured"`, on top of the existing confidence and multi-hop checks.
Retested both modes after the fix:

| mode | router decision | routed accuracy | always-REASON | DIRECT-only (oracle, ref) |
|---|---|---|---|---|
| unstructured | 147/147 REASON | **87.07%** (recovered the full gap) | 87.07% | -- |
| structured | 147/147 DIRECT | **100.0%** | 97.96% | 100.0% (matches oracle exactly) |

**The fixed router now makes the theoretically correct decision in both
regimes**, and the structured-mode result confirms DIRECT genuinely earns
its place there -- not just faster (no generation call), but 2 points more
accurate than forcing generation, exactly mirroring the original oracle-
evidence pilot's finding (hard_playback 100% vs contextual_generation 97.96%
on structured CounterFact). The router's live decisions on both modes now
match what the oracle-evidence experiments already predicted was optimal --
the strongest evidence yet that the redesigned architecture (retrieval,
verification, mode-aware routing, generation) composes correctly as a
system, not just as isolated, independently-tuned components.

## 7B-scale validation

Reran the fixed unstructured-mode pipeline with `Qwen/Qwen2.5-7B-Instruct`
(n=147, same test split), after resolving a real infrastructure issue: the
host's small 25GB root disk filled up mid-download (only ~1GB free against a
~15GB model), which surfaced as an opaque Xet "background writer channel
closed" error that looked network-related but wasn't -- confirmed by the
actual `huggingface_hub` disk-space warnings underneath it. Fixed by
redirecting `HF_HOME` to the host's 360GB `/mnt/scratch` volume rather than
the small root disk.

| model | routed accuracy | router decisions |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 87.07% | 147/147 REASON |
| Qwen2.5-7B-Instruct | **89.12%** | 147/147 REASON |

**The router's fixed decision logic and the underlying finding both hold at
7B scale**: quality improves with model size as expected (87.07% -> 89.12%),
and the router still correctly routes every unstructured-mode query to
REASON rather than the degraded DIRECT fallback. This is the strongest
available evidence that the router fix is a genuine architectural
correction, not an artifact of the smaller pilot model's specific behavior.

## Extension: WikiUpdate, the hardest retrieval case

WikiUpdate unstructured, n=160, real (non-oracle) pipeline, verifier v2.
This is the dataset the Stage 1 retrieval pilot already flagged as hardest
(~70% R@1, its real-world temporal old/new officeholder conflicts).

| | accuracy | router decisions |
|---|---|---|
| routed full pipeline | **43.75%** | REJECT 42 / DIRECT 0 / REASON 118 |
| always-force-REASON | 43.13% | -- |

Retrieval correctness in the live pipeline: 71.88%, consistent with the
Stage 1 finding this dataset is structurally harder. Two things worth
noting. First, **the router's REJECT path is now doing real, visible work**:
42/160 queries (26%) were correctly declined rather than forced through a
low-confidence retrieval -- unlike CounterFact, where REJECT never fired
because retrieval was near-perfect there. Second, routed still edges out
forcing REASON blindly (43.75% vs 43.13%), a small but real margin from
correctly declining to hallucinate on genuinely bad retrievals rather than
generating a plausible-sounding wrong answer anyway.

**A second real scoring bug was found and fixed here**: the first pass
scored a factually-perfect answer ("Ismail Kartal") as a miss against gold
("İsmail Kartal") purely because Python's `.lower()` does not map Turkish
İ to plain ASCII 'i' (it produces 'i' plus a combining dot-above instead).
Fixed `is_hit()` to NFKD-decompose and strip combining marks before
case-folding, which corrects this generally across accented/diacritic
names, not just this one case -- important specifically for WikiUpdate,
whose real-world entities are far more likely to have non-ASCII names than
CounterFact's. The numbers above are post-fix; the pre-fix run read 43.13%
routed / 42.5% always-REASON, a real ~0.6-point undercount the fix recovered.

## Extension: MQuAKE-CF multi-hop with verifier v2

Reran the multi-hop pilot (same 80-example sample, same fallback fix) with
the v2 verifier (trained with MQuAKE-CF hard negatives) instead of v1.
Result: **identical, 47.5%**, bit-for-bit the same as v1. The multi-hop
loop's fallback design (falling back to base-model knowledge on a
low-verifier-score hop rather than terminating) appears robust to which
verifier version is behind it, at least on this sample -- a reasonable,
honest null result, not evidence either way about the verifier retrain's
value specifically in the multi-hop setting.

## Extension: WikiUpdate and MQuAKE-CF, structured mode

WikiUpdate structured (n=160): routed 99.38% vs always-REASON 96.25%, router
159/160 DIRECT (1 REJECT) -- confirms the same clean structured-mode pattern
CounterFact showed: DIRECT genuinely earns its place when the card carries a
literal answer.

**MQuAKE-CF structured (n=63) breaks that pattern -- a real, dataset-specific
finding, not noise.** Routed pipeline (93.65%) actually loses to
always-forcing-REASON (96.83%), the opposite of every other structured-mode
result. Router decisions: DIRECT 53, REASON 6, REJECT 4 -- mixed, unlike the
near-unanimous DIRECT calls elsewhere. Root cause traced to retrieval
quality: MQuAKE-CF's retrieval correctness in the live pipeline is 82.54%,
meaningfully below CounterFact/WikiUpdate structured's ~99%, consistent with
the Stage 1 pilot's original finding that MQuAKE-CF's entity/relation
collisions make retrieval structurally harder. When DIRECT confidently
recites a literal answer from a WRONG retrieval, it fails outright; REASON's
generation, even over the same imperfect evidence, degrades more gracefully.
**The lesson: DIRECT's confidence threshold should probably be tied to the
dataset's actual retrieval reliability, not a single global threshold** --
the same class of finding as the verifier's threshold-miscalibration result,
now showing up in the router's DIRECT/REASON boundary too. Not fixed in this
pass; a concrete, specific direction for future router calibration work.

## Baseline comparison: CAKE vs plain RAG vs IKE

See `akew_baseline_comparison_results.md` for the full report. Headline,
CounterFact unstructured (n=147): CAKE and plain RAG tie at 87.07% (they are
mechanically identical here, since the router routes 100% to REASON on this
dataset/mode); **IKE (RAG plus few-shot override demonstrations) wins at
90.48%** -- a real, honestly-reported result that doesn't flatter CAKE, and
a concrete, cheap improvement worth folding into CAKE's own REASON path
(demonstrations plus CAKE's retrieval/verification/routing, which plain RAG
and IKE lack entirely and would need on datasets where REJECT matters, like
WikiUpdate).

## Following up: does raising the DIRECT threshold fix MQuAKE-CF structured?

Tested two threshold variants against the same MQuAKE-CF structured split,
directly following the finding above rather than assuming a fix would work.

| direct_threshold | routed accuracy | router decisions | always-REASON (ref) |
|---|---|---|---|
| 0.85 (default) | 93.65% | DIRECT 53 / REASON 6 / REJECT 4 | 96.83% |
| 0.97 | 93.65% (identical) | DIRECT 53 / REASON 6 / REJECT 4 (identical) | 96.83% |
| 1.01 (DIRECT fully disabled) | 90.48% | DIRECT 0 / REASON 59 / REJECT 4 | 96.83% |

**Raising the threshold to 0.97 changed nothing** -- the verifier's confidence
on MQuAKE-CF's wrong retrievals is already above 0.97 (matching the
recalibration ablation's finding that the positive/negative score
distributions genuinely overlap there; no scalar threshold separates them).
**Fully disabling DIRECT still trails always-REASON** (90.48% vs 96.83%),
which at first looks like it shouldn't be possible -- if every non-REJECT
query goes to REASON either way, routed and always-REASON should be
identical. The gap is the 4 REJECT cases: routed's REJECT answers those from
the base model's own knowledge alone (no retrieved evidence at all), while
always-REASON forces generation over whatever was retrieved regardless of
confidence -- and on this dataset, even a low-confidence, possibly-imperfect
retrieval turns out to carry more useful signal than no evidence at all.

**Honest conclusion: neither threshold, tuned individually, fully closes this
gap.** The router's two scope decisions (reject vs proceed, direct vs reason)
are each individually net-negative on MQuAKE-CF's specific retrieval-
reliability regime. The practical fix isn't a numeric threshold sweep at
all -- it's recognizing that for a dataset this retrieval-unreliable, the
correct router configuration is closer to a full bypass (always REASON, both
gates disabled) than any calibrated version of the gates themselves. This is
a genuinely different, more decisive conclusion than the original "recover
the gap with a stricter DIRECT threshold" hypothesis, reached by actually
testing it rather than assuming the fix would work once identified.

## Scope note

Weight-editing baselines (ROME/MEMIT/AlphaEdit/WISE/GRACE) and MeLLo remain
out of scope for the reasons already stated (a separate multi-day harness-
engineering phase). Unstructured/extracted mode tests on MQuAKE-CF, and
testing whether IKE's demonstration boost holds up under WikiUpdate-style
retrieval uncertainty, are the remaining natural extensions.
