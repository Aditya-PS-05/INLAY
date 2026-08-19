# AKEW acquisition and schema validation — 2026-08-19

Repo: https://github.com/bobxwu/AKEW, pinned at commit
`6bcd8e9a28cf16a530739c14425e82e9bede2cec` (2025-02-12), cloned to
`../AKEW/repo` (sibling of `cake_prototype/`).

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
`cake_prototype/` or `cake_prototype/src/`.

## What this milestone does NOT cover yet

This is section 2+3 of the vNext brief only: data acquired, validated, and
normalized into knowledge cards with leakage controls. Not yet built: the
two-stage scope router (section 4), the REASON-mode contextual generation and
iterative multi-hop path (section 5), the experimental matrix (section 6), or
any of the metrics/ablations/validation-controls sections (7-9). Those are
the next phases.
