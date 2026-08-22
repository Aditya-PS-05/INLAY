# AKEW acquisition and schema validation — 2026-08-19

Repo: https://github.com/bobxwu/AKEW, pinned at commit
`6bcd8e9a28cf16a530739c14425e82e9bede2cec` (2025-02-12), cloned to
`../AKEW/repo` (sibling of `inlay_prototype/`).

## Record counts (validated against the actual files, not assumed from the paper)

| dataset | brief's expected count | actual count | match |
|---|---|---|---|
| CounterFact.json | 975 edits | 975 | exact |
| MQuAKE-CF.json | 436 edits, ~354 questions | 354 question-groups, 436 individual edits (277 groups w/ 1 edit, 72 w/ 2, 5 w/ 3) | exact |
| WikiUpdate.json | ~1,067 updates | 1,056 | **off by 11** — the paper's figure and the repo's shipped data differ; recorded here rather than silently normalized to the paper's number |

## Schema (validated on every record, not sampled)

`requested_rewrite` fields present with zero missing/empty values across all
records in all three files:

- Structured: `prompt`, `subject`, `relation_id`, `target_new`, `target_true`, `fact_new`
- Unstructured: `fact_new_uns` (raw prose evidence)
- Extracted: `unsfact_triplets_GPT` (list of `{subject, prompt, target}` triples)
- WikiUpdate additionally carries `time_new`/`time_true` (validity intervals, present on 100% of records) and `subject_id`/`answer_GPT`.
- MQuAKE-CF additionally carries `questions` (3 rephrasings per group, present on 100% of groups) and `single_hops`/`new_single_hops` (per-hop decomposition with per-hop cloze/answer/aliases).

No missing top-level `case_id` / `requested_rewrite`, no missing/empty
`fact_new_uns` or `unsfact_triplets_GPT` anywhere in any of the three files.

## Data layer built (`src/akew_data.py`, `tests/test_akew_leakage.py`)

`load_akew(dataset_name, input_mode)` normalizes all three datasets under all
three AKEW input conditions into a `KnowledgeCard` (visible to the editor) and
a separate `GoldRecord` (evaluator-only, never passed to ingestion/retrieval).

Leakage control is structural, not a post-hoc check: `_rr_get()` raises
`LeakageError` if code tries to read any field in `GOLD_FIELDS`
(`target_new`, `target_true`, `answer_new`, `answer_new_alias`,
`answer_true`, `answer_true_alias`, `answer_GPT`, `fact_new`, `target_old`)
outside the one structured-mode branch explicitly allowed to. This caught a
real bug in my own first draft (structured mode's legitimate read of
`fact_new` went through the guarded reader and was correctly blocked) before
any experiment could have run on it.

`tests/test_akew_leakage.py` validates, over every card in every
dataset×mode combination (not a sample): count regression against the table
above, zero leakage-guard violations, and correct field presence per mode
(structured cards carry `canonical_fact_text` and nothing else; unstructured/
extracted cards carry `raw_evidence_text` and never `canonical_fact_text`).

All 9 combinations pass. Path resolution is anchored to the module's own file
location (`os.path.dirname(os.path.abspath(__file__))`), not the caller's cwd,
after an initial version broke depending on whether it was invoked from
`inlay_prototype/` or `inlay_prototype/src/`.

## Stage 1 retrieval pilot (dense, no ANN, full data, no sampling)

`src/akew_retrieval.py` (exact dense retrieval, raw normalized MiniLM
embeddings, no JL projection) + `src/akew_retrieval_pilot.py`, run on
g6e4xlarge over the complete dataset (not a sample) for all 9 dataset x mode
combinations, using each card's own AKEW `question` field as the query
(read only by the evaluator script, never during index construction):

| dataset | mode | Recall@1 | Recall@5 | MRR | no-candidate rate |
|---|---|---|---|---|---|
| CounterFact | structured | 0.999 | 1.000 | 0.9995 | 0.0% |
| CounterFact | unstructured | 0.995 | 1.000 | 0.9974 | 0.0% |
| CounterFact | extracted | 0.992 | 0.998 | 0.9947 | 0.2% |
| MQuAKE-CF | structured | 0.773 | 0.917 | 0.8287 | 8.3% |
| MQuAKE-CF | unstructured | 0.750 | 0.913 | 0.8115 | 8.7% |
| MQuAKE-CF | extracted | 0.727 | 0.906 | 0.7956 | 9.4% |
| WikiUpdate | structured | 0.996 | 1.000 | 0.9979 | 0.0% |
| WikiUpdate | unstructured | 0.703 | 0.842 | 0.7587 | 15.8% |
| WikiUpdate | extracted | 0.703 | 0.854 | 0.7632 | 14.6% |

**Retrieval quality is not uniform across datasets or input conditions**,
which matters for interpreting anything downstream: CounterFact is close to
saturated everywhere (its facts are largely unrelated to each other, an easy
retrieval problem by construction). MQuAKE-CF is meaningfully harder even at
the single-hop level, likely because it was built for multi-hop chains and
so entities/relations recur across records more than CounterFact's do.
WikiUpdate's structured condition is near-perfect, but unstructured/extracted
collapse to ~70% Recall@1 with a 15%+ no-candidate rate -- a real, large gap,
not router noise. The likely cause: WikiUpdate's own schema stores explicit
`time_true`/`time_new` validity intervals specifically because it contains
temporally conflicting facts (an old office-holder and a new one for the same
role), and unstructured prose about the new holder can closely resemble prose
about the old one, a genuinely hard disambiguation case for pure semantic
similarity that a router will not be able to fully cover for if the correct
card isn't even retrieved into the candidate set to begin with.

Practical implication for section 4's scope verifier: its hardest job is not
uniform across the benchmark. On CounterFact it barely matters (retrieval is
already near-perfect). On WikiUpdate's unstructured conditions, no verifier
can recover from a 15% retrieval miss rate -- that failure has to be attacked
at the retrieval/embedding stage (or accepted and reported honestly as a
retrieval-limited ceiling), not the verifier stage.

## What this milestone does NOT cover yet

This is section 2+3 of the vNext brief only: data acquired, validated, and
normalized into knowledge cards with leakage controls. Not yet built: the
two-stage scope router (section 4), the REASON-mode contextual generation and
iterative multi-hop path (section 5), the experimental matrix (section 6), or
any of the metrics/ablations/validation-controls sections (7-9). Those are
the next phases.
