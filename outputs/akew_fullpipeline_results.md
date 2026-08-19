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

## Scope note

This test used CounterFact only; WikiUpdate and MQuAKE-CF full-pipeline
tests (with the multi-hop fallback wired into live routing, not just the
oracle-decomposition pilot) are the natural next extension, not yet run.
