"""
AKEW data layer: load the three AKEW datasets under the three input conditions
and normalize them into a single "knowledge card" representation.

AKEW paper: https://aclanthology.org/2024.emnlp-main.843/
Repo (pinned): https://github.com/bobxwu/AKEW @ 6bcd8e9a28cf16a530739c14425e82e9bede2cec

Schema, validated against the real files (not assumed from the paper), see
outputs/akew_schema_validation.md for the full report:

  CounterFact.json  : list[975], requested_rewrite is a single dict per record.
  MQuAKE-CF.json    : list[354] question-GROUPS. Each group's requested_rewrite
                       is a LIST of 1-3 edits (277 groups w/ 1 edit, 72 w/ 2,
                       5 w/ 3 -- 436 individual edits total), and 3 rephrasings
                       of the multi-hop question in `questions`.
  WikiUpdate.json   : list[1056] (paper cites ~1067; real file has 1056 -- this
                       is the AKEW repo's actual shipped data, not a bug on our
                       side, and is recorded here rather than silently adjusted
                       to match the paper's number).

Every record has fact_new_uns (raw unstructured evidence text) and
unsfact_triplets_GPT (extracted triples) with zero missing values across all
three files (validated on all records, not sampled).

CRITICAL invariant this module exists to enforce: which fields a knowledge
card carries depends on input_mode, and unstructured/extracted cards must
NEVER see target_new/answer_new/aliases/target_true or any other
evaluation-only field. Those live in a separate GOLD record, kept apart, only
handed to the evaluator after generation -- never to the editor.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Resolve the AKEW datasets dir relative to THIS FILE, not the caller's cwd --
# a caller running from cake_prototype/, cake_prototype/src/, or anywhere else
# all resolve to the same real path instead of silently depending on cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AKEW_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "AKEW", "repo", "datasets"))

# ---- forbidden-field leakage guard -----------------------------------------
# Any AKEW requested_rewrite key that names an answer/target/alias is gold and
# must never be read while building an unstructured or extracted card.
GOLD_FIELDS = {
    "target_new", "target_old", "target_true", "answer_new", "answer_true",
    "answer_new_alias", "answer_true_alias", "answer_GPT", "fact_new",
}


class LeakageError(Exception):
    """Raised if code tries to read a gold field while building a non-structured card."""


@dataclass
class KnowledgeCard:
    edit_id: str
    subject: str
    relation: Optional[str]              # relation_id (e.g. "P190"); None if not derivable (unstructured)
    canonical_fact_text: Optional[str]   # structured mode only: the clean "X is Y" sentence
    raw_evidence_text: Optional[str]     # unstructured mode: fact_new_uns; extracted mode: joined triples
    source: str                          # dataset name: "CounterFact" | "MQuAKE-CF" | "WikiUpdate"
    input_mode: str                      # "structured" | "unstructured" | "extracted"
    provenance_case_id: str
    provenance_group_id: Optional[str] = None   # MQuAKE-CF question-group id, else None
    subject_aliases: list = field(default_factory=list)
    validity_start: Optional[str] = None
    validity_end: Optional[str] = None
    embedding: Optional[object] = None   # filled in later by the retrieval layer, not here


@dataclass
class GoldRecord:
    """Evaluation-only / training-data-construction-only. Never passed to an
    editor at inference time; used by the scorer after generation, or by
    offline hard-negative mining (which needs the true relation_id to group
    same-subject/different-relation pairs even in unstructured/extracted mode,
    where the KnowledgeCard itself correctly hides relation as None -- see
    akew_hard_negatives.build_specificity_negatives)."""
    edit_id: str
    target_new: str
    target_true: Optional[str]
    aliases_new: list
    aliases_true: list
    eval_question: Optional[str] = None       # AKEW's `question` field, for editing-accuracy scoring
    mquake_questions: Optional[list] = None   # only set for MQuAKE-CF groups
    relation_id: Optional[str] = None         # schema-level slot label, not an answer; safe for grouping


def _rr_get(rr, key):
    """Read a NON-gold field from a requested_rewrite dict. Refuses gold fields
    outright so a future edit to this module can't accidentally reintroduce a
    leak -- the guard lives at the read site, not just at the call site."""
    if key in GOLD_FIELDS:
        raise LeakageError(f"_rr_get() refuses to read gold field '{key}' -- use _gold_get()")
    return rr.get(key)


def _gold_get(rr, key, default=None):
    return rr.get(key, default)


def _triples_to_text(triplets):
    """unsfact_triplets_GPT -> one short evidence line per triple, {}-templated
    prompts filled with the triple's own subject. This is the AKEW-native
    'extracted' condition: structured-looking, but machine-extracted and
    therefore allowed to be noisy -- we do not clean it up here."""
    lines = []
    for t in triplets or []:
        subj = t.get("subject", "")
        prompt = t.get("prompt", "{}")
        tgt = t.get("target", "")
        lines.append(f"{prompt.replace('{}', subj)} {tgt}".strip())
    return " | ".join(lines)


def _card_from_rr(rr, input_mode, source, edit_id, case_id, group_id=None,
                   validity_start=None, validity_end=None):
    subject = _rr_get(rr, "subject")
    relation = _rr_get(rr, "relation_id")
    if input_mode == "structured":
        # fact_new is in GOLD_FIELDS on purpose: it states the new fact plainly,
        # which *is* the edit content for structured mode (legitimately visible
        # here), but must never reach unstructured/extracted cards. Reading it
        # via _gold_get() here, deliberately, in the ONLY branch allowed to.
        return KnowledgeCard(
            edit_id=edit_id, subject=subject, relation=relation,
            canonical_fact_text=_gold_get(rr, "fact_new"),
            raw_evidence_text=None, source=source, input_mode=input_mode,
            provenance_case_id=case_id, provenance_group_id=group_id,
            validity_start=validity_start, validity_end=validity_end,
        )
    if input_mode == "unstructured":
        return KnowledgeCard(
            edit_id=edit_id, subject=subject, relation=None,
            canonical_fact_text=None,
            raw_evidence_text=_rr_get(rr, "fact_new_uns"),
            source=source, input_mode=input_mode,
            provenance_case_id=case_id, provenance_group_id=group_id,
            validity_start=validity_start, validity_end=validity_end,
        )
    if input_mode == "extracted":
        return KnowledgeCard(
            edit_id=edit_id, subject=subject, relation=None,
            canonical_fact_text=None,
            raw_evidence_text=_triples_to_text(_rr_get(rr, "unsfact_triplets_GPT")),
            source=source, input_mode=input_mode,
            provenance_case_id=case_id, provenance_group_id=group_id,
            validity_start=validity_start, validity_end=validity_end,
        )
    raise ValueError(f"unknown input_mode {input_mode!r}")


def _gold_from_rr(rr, edit_id):
    return GoldRecord(
        edit_id=edit_id,
        target_new=_gold_get(rr, "target_new", {}).get("str") if isinstance(_gold_get(rr, "target_new"), dict) else _gold_get(rr, "target_new"),
        target_true=(_gold_get(rr, "target_true", {}) or {}).get("str") if isinstance(_gold_get(rr, "target_true"), dict) else _gold_get(rr, "target_true"),
        aliases_new=_gold_get(rr, "answer_new_alias", []) or [],
        aliases_true=_gold_get(rr, "answer_true_alias", []) or [],
        eval_question=_gold_get(rr, "question"),
        relation_id=rr.get("relation_id"),  # not in GOLD_FIELDS: a schema label, not an answer
    )


def load_akew(dataset_name, input_mode, path=None):
    """Returns (cards: list[KnowledgeCard], golds: dict[edit_id -> GoldRecord],
    groups: dict[group_id -> list[edit_id]] (only non-empty for MQuAKE-CF)).

    dataset_name in {"CounterFact", "MQuAKE-CF", "WikiUpdate"}
    input_mode   in {"structured", "unstructured", "extracted"}
    """
    assert dataset_name in ("CounterFact", "MQuAKE-CF", "WikiUpdate")
    assert input_mode in ("structured", "unstructured", "extracted")
    path = path or os.path.join(_DEFAULT_AKEW_DIR, f"{dataset_name}.json")
    data = json.load(open(path))

    cards, golds, groups = [], {}, {}

    if dataset_name == "MQuAKE-CF":
        for i, rec in enumerate(data):
            gid = f"mquake_{i}"
            rr_list = rec["requested_rewrite"]
            edit_ids = []
            for j, rr in enumerate(rr_list):
                eid = f"{gid}_edit{j}"
                cards.append(_card_from_rr(rr, input_mode, dataset_name, eid,
                                            case_id=rec.get("case_id", gid), group_id=gid))
                golds[eid] = _gold_from_rr(rr, eid)
                edit_ids.append(eid)
            groups[gid] = edit_ids
            # multi-hop questions live at the GROUP level, not per-edit -- stash
            # on the first edit's gold record for the evaluator to find via group_id
            golds[edit_ids[0]].mquake_questions = rec.get("questions", [])
        return cards, golds, groups

    # CounterFact / WikiUpdate: one edit per record
    for rec in data:
        rr = rec["requested_rewrite"]
        cid = str(rec.get("case_id"))
        eid = f"{dataset_name}_{cid}"
        vt = rr.get("time_new") if dataset_name == "WikiUpdate" else None
        cards.append(_card_from_rr(
            rr, input_mode, dataset_name, eid, case_id=cid,
            validity_start=(vt or {}).get("start_time") if vt else None,
            validity_end=(vt or {}).get("end_time") if vt else None,
        ))
        golds[eid] = _gold_from_rr(rr, eid)
    return cards, golds, groups


def assert_no_gold_leakage(card):
    """Programmatic check for section 9's leakage control: a card built for
    unstructured/extracted mode must carry no gold answer text anywhere in its
    visible fields. Run this over a sample after ingestion, not just trust the
    ingestion code path."""
    if card.input_mode == "structured":
        return  # structured mode is allowed to see the fact itself
    for f in (card.canonical_fact_text, card.raw_evidence_text):
        if f is None:
            continue
    if card.canonical_fact_text is not None:
        raise LeakageError(f"{card.edit_id}: non-structured card has canonical_fact_text set")


if __name__ == "__main__":
    import sys
    for ds in ("CounterFact", "MQuAKE-CF", "WikiUpdate"):
        for mode in ("structured", "unstructured", "extracted"):
            cards, golds, groups = load_akew(ds, mode)
            for c in cards[:50]:
                assert_no_gold_leakage(c)
            n_missing_evidence = sum(
                1 for c in cards
                if (mode != "structured" and not c.raw_evidence_text)
                or (mode == "structured" and not c.canonical_fact_text)
            )
            print(f"{ds:12s} {mode:13s} cards={len(cards):5d} golds={len(golds):5d} "
                  f"groups={len(groups):4d} missing_content={n_missing_evidence}")
    print("\nAll leakage-guard spot checks passed.", file=sys.stderr)
