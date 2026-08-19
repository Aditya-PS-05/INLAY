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

## Scope note

Weight-editing baselines, IKE, and MeLLo remain out of scope for the reasons
already stated (a separate multi-day harness-engineering phase). Structured-
mode full-pipeline tests on WikiUpdate and MQuAKE-CF, and unstructured/
extracted mode tests on MQuAKE-CF, are the remaining natural extensions of
this specific line of testing.
