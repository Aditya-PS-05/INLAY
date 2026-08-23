# The out-of-scope condition: routing headroom appears once negatives exist

**2026-08-22.** Constructs and measures the missing evaluation condition named in
`akew_routing_headroom_results.md`'s own "the experiment this demands" section: ask
evaluation questions against an index that does **not** contain the corresponding edit,
so abstention (REJECT) becomes the objectively correct action for some queries.

## Method

For each of the same 9 dataset x mode cells used in the original headroom measurement
(CounterFact / WikiUpdate / MQuAKE-CF, each in structured / unstructured / extracted
input mode), a fraction of queries (`oos_frac = 0.5`, assigned per-query via an
independent seeded RNG so the split does not correlate with sampling order) has its
**own edit card removed from the retrieval index** before the per-action executor runs.
Concretely: `DenseCardIndex.query()` over-fetches `TOPK + 10` candidates, the query's
own `edit_id` is filtered out in Python, and the result is truncated back to `TOPK = 5`
— equivalent to that card never having been indexed, without paying for a leave-one-out
re-encode per query.

For a removed-edit ("negative") query, no edit genuinely applies to the retrieved
context, so the objectively correct behaviour is "answer from parametric knowledge" —
i.e. REJECT — and correctness is scored against the **pre-edit** answer
(`target_true`/`aliases_true`), not the post-edit one. Untouched ("positive") queries
are scored exactly as in the original measurement, against `target_new`/`aliases_new`,
with the edit fully present in the index. REJECT/REASON/DIRECT are each actually
executed and scored per query, exactly as in `akew_outcome_labels.py` — this script
(`akew_outcome_labels_oos.py`) only changes which queries get a real vs. an absent edit
and which gold record they are scored against; the executor, the verifier, the retrieval
index, and `akew_headroom.py`'s oracle-vs-static analysis are byte-identical to the
original run and required zero modification.

Model: Qwen/Qwen2.5-1.5B-Instruct (matching the original headroom run). Splits: `train`
for CounterFact/WikiUpdate, `test` for MQuAKE-CF (held out, matching the project's OOD
convention elsewhere in this codebase). Subject-disjoint split, seed 0.

**Smoke test first, per this project's own convention.** n=100, CounterFact/structured,
oos_frac=0.5 was run and inspected before committing to the full 1,689-query scale — see
`smoke_oos_cf_structured.log` on g6e4xlarge. It produced headroom **+0.1500** (vs the
original's exactly +0.0000), with 4/100 queries where REJECT was the sole winning
action, confirming the harness modification behaves as designed before the full run.

## Result: full scale, n=1,689 (identical sample sizes to the original measurement)

| cell | n | static | oracle | headroom | REJECT-only |
|---|---|---|---|---|---|
| CounterFact/structured | 250 | 0.5520 | 0.6360 | **+0.0840** | 4 |
| CounterFact/unstructured | 250 | 0.5440 | 0.5600 | **+0.0160** | 4 |
| CounterFact/extracted | 250 | 0.4560 | 0.4720 | **+0.0160** | 4 |
| WikiUpdate/structured | 250 | 0.4840 | 0.5160 | **+0.0320** | 7 |
| WikiUpdate/unstructured | 250 | 0.2200 | 0.2520 | **+0.0320** | 8 |
| WikiUpdate/extracted | 250 | 0.2320 | 0.2480 | **+0.0160** | 4 |
| MQuAKE-CF/structured | 63 | 0.4127 | 0.5397 | **+0.1270** | 7 |
| MQuAKE-CF/unstructured | 63 | 0.4286 | 0.5556 | **+0.1270** | 8 |
| MQuAKE-CF/extracted | 63 | 0.3651 | 0.4603 | **+0.0952** | 6 |
| **POOLED** | **1689** | **0.4133** | **0.4553** | **+0.0420** | **52** |

Compare against the original zero-headroom measurement, same 9 cells, same n:

| cell | static | oracle | headroom (original) |
|---|---|---|---|
| POOLED | 0.7874 | 0.7874 | **+0.0000** |

Headroom moved from **+0.0000 to +4.20 points pooled**. Every single cell now shows
positive headroom (range +0.0160 to +0.1270), where every cell previously showed exactly
zero. Queries where REJECT is the sole winning action rose from **0/1689 (0.00%)** to
**52/1689 (3.08%)**.

## Mechanism check: the effect lives entirely in the negative population

Splitting each cell's queries by whether they are a positive (edit present, scored
against `target_new`) or negative (edit absent, scored against `target_true`) query
isolates where the headroom comes from:

| population | n | static | oracle | headroom | REJECT-only | REJECT-only rate |
|---|---|---|---|---|---|---|
| Positive (edit present) | 810 | 0.7753 | 0.7778 | +0.0025 | 2 | 0.25% |
| Negative (edit absent) | 879 | 0.0796 | 0.1581 | **+0.0785** | 50 | 5.69% |

The positive population reproduces the original near-zero-headroom result almost exactly
(+0.0025 vs. the original's +0.0000 — the tiny residual is WikiUpdate/{extracted,
unstructured}, where a same-subject-different-fact card sometimes gets retrieved even
with the query's own card present, giving REJECT/REASON a genuine independent chance to
differ; this is a pre-existing, sample-size-driven wrinkle in the original protocol, not
an artifact introduced here). All of the new headroom is concentrated in the negative
population, exactly as the mechanism predicts: when there is no relevant edit, REASON
answers with the (wrong, misretrieved) card's context and DIRECT recites it verbatim,
while REJECT correctly falls back to the model's own unedited knowledge. Per-cell
negative-only detail (static / oracle / headroom / REJECT-only-count):

| cell | static (neg) | oracle (neg) | headroom (neg) | REJECT-only (neg) |
|---|---|---|---|---|
| CounterFact/structured | 0.1250 | 0.2891 | +0.1641 | 4 |
| CounterFact/unstructured | 0.2188 | 0.2500 | +0.0312 | 4 |
| CounterFact/extracted | 0.1406 | 0.1719 | +0.0312 | 4 |
| WikiUpdate/structured | 0.0078 | 0.0703 | +0.0625 | 7 |
| WikiUpdate/unstructured | 0.0078 | 0.0625 | +0.0547 | 7 |
| WikiUpdate/extracted | 0.0078 | 0.0312 | +0.0234 | 3 |
| MQuAKE-CF/structured | 0.0000 | 0.2162 | +0.2162 | 7 |
| MQuAKE-CF/unstructured | 0.1081 | 0.3243 | +0.2162 | 8 |
| MQuAKE-CF/extracted | 0.0270 | 0.1892 | +0.1622 | 6 |

MQuAKE-CF shows the largest per-cell effect (+0.216 on structured and unstructured) —
consistent with its role elsewhere in this project as the dataset where routing
gains/losses have been largest (`akew_fullpipeline_results.md`'s net-negative gating
finding, `akew_reliability.py`'s OOD-generalization target).

## What this does and does not establish

**What survives, unrevised:** every finding in `akew_routing_headroom_results.md`
about the *original* AKEW-as-shipped condition stands. AKEW's own eval questions all
target edited facts; on that population REJECT genuinely is never uniquely correct, and
a one-line static policy genuinely is oracle-optimal there. This experiment does not
retract that — it constructs a *different* population (half of it synthetic negatives)
to show what changes once out-of-scope queries exist at all.

**What this establishes:** the zero-headroom result was a property of the query
population, not of routing or of this method's mechanism. Once negatives exist — even
by the crude method of deleting a card from the index, not by curating genuinely novel
out-of-scope questions — an oracle router pulls measurably ahead of the static policy,
and it does so specifically by using REJECT on exactly the queries where REJECT is
correct. This is the first evidence in this project that a scope-classifier's abstention
path can win points on an evaluation constructed to give it the chance to.

**What this does not establish:** this is a synthetic negative, not a curated one. A
"negative" here is any query whose card was artificially removed from an otherwise
intact index — it says nothing about how a *real* deployment's out-of-scope query
distribution would look (queries about facts that were never edited at all, adversarial
paraphrases of edited facts, or genuinely unrelated questions), only that the mechanism
the paper's limitations section predicted (headroom requires negatives) reproduces when
negatives of any kind are present. `oos_frac = 0.5` was chosen for a roughly balanced
positive/negative split (realized split: 810 positive / 879 negative across all 9
cells, close to but not exactly 50/50 because the per-query RNG draw is independent of
cell size); the headroom magnitude at other negative fractions was not swept and would
be expected to scale roughly with the negative fraction, not measured here.

**Honesty on candidate starvation and skipped queries.** The over-fetch buffer (`TOPK +
10` for negative queries) was checked directly rather than assumed sufficient:
`n_starved` (negative queries where filtering the excluded card left fewer than 5
remaining candidates) and `n_no_target_true` (negative queries skipped because
`target_true` was empty, so there is nothing to score REJECT against) are **both 0
across all 9 cells** — every query in the full run was scored on its full intended
protocol, with no silent degradation.

## Reproducing

```
# smoke test (run first, inspect the numbers, then scale up)
AKEW_SPLIT=train python3 src/akew_outcome_labels_oos.py CounterFact structured 100 0.5
python3 src/akew_headroom.py "outputs/outcome_labels_oos_CounterFact_structured_train_f0.50.json"

# full scale (9 cells, ~1,689 queries total)
for m in structured unstructured extracted; do
  AKEW_SPLIT=train python3 src/akew_outcome_labels_oos.py CounterFact $m 250 0.5
  AKEW_SPLIT=train python3 src/akew_outcome_labels_oos.py WikiUpdate  $m 250 0.5
  AKEW_SPLIT=test  python3 src/akew_outcome_labels_oos.py MQuAKE-CF   $m 250 0.5
done
python3 src/akew_headroom.py "outputs/outcome_labels_oos_*_f0.50.json"
```

## Source logs and files (g6e4xlarge, `~/kw/cake_prototype/`)

- Smoke test: `~/kw/smoke_oos_cf_structured.log` -> `outputs/outcome_labels_oos_CounterFact_structured_train_f0.50.json`
- Full run (9 cells): `~/kw/oos_{cf,wiki,mquake}_{structured,unstructured,extracted}.log`
  -> `outputs/outcome_labels_oos_{CounterFact,WikiUpdate}_*_train_f0.50.json` and
  `outputs/outcome_labels_oos_MQuAKE-CF_*_test_f0.50.json`
- Original comparison table: `outputs/outcome_labels_{CounterFact,WikiUpdate}_*_train.json`,
  `outputs/outcome_labels_MQuAKE-CF_*_test.json` (unmodified, already present before this
  session; source for the original +0.0000 pooled headroom cited above).
- New source file: `src/akew_outcome_labels_oos.py` (adds the index-exclusion + gold-swap
  logic on top of the unmodified `akew_outcome_labels.py`; `src/akew_headroom.py` required
  zero changes to consume its output).
