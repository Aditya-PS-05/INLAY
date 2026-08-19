"""
Hard-negative construction for the scope-verifier cross-encoder (brief section
4, stage 2). Paraphrase augmentation alone is insufficient -- specificity
negatives (same-subject/different-relation especially) are what actually teach
the verifier to reject over-firing, per EVOKE's finding that this is the
under-covered case in prior editing-scope work.

Each of the 8 categories the brief lists is built here. Categories that need
nearest-neighbor lookups (3, 6, 8) take a pre-built DenseCardIndex; the rest
are pure data operations over the loaded KnowledgeCards/GoldRecords.

Output shape: NegativeExample(query_text, wrong_card, negative_type,
source_edit_id) -- paired against the query's TRUE card by the caller when
assembling (anchor, positive, negative) training triples, so this module
never needs to know what "positive" means for a given query.
"""
import random
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NegativeExample:
    query_text: str
    wrong_card_id: str
    negative_type: str          # one of the 8 category names below
    source_edit_id: str         # the edit_id the query actually belongs to
    candidate_text: str = ""    # the actual text a trainer feeds the verifier as the
                                 # candidate side -- explicit rather than parsed back
                                 # out of synthetic ids like "edit__stale(...)"


CATEGORIES = [
    "same_subject_diff_relation",
    "same_relation_diff_subject",
    "nearest_semantic_neighbor",
    "stale_object_same_slot",
    "same_document_sibling_triple",
    "lexically_similar_irrelevant",
    "prefixed_unrelated_question",
    "ranked_below_correct",
]


def _query_for(card, gold):
    return gold.eval_question if (gold and gold.eval_question) else (card.canonical_fact_text or card.raw_evidence_text or "")


def _card_text(card):
    return card.canonical_fact_text or card.raw_evidence_text or ""


def build_specificity_negatives(cards, golds):
    """Categories 1 and 2: same-subject/different-relation and
    same-relation/different-subject. Pure grouping, no retrieval needed.

    Groups by the TRUE relation_id from golds (a schema-level slot label, not
    an answer -- safe for offline training-data construction), not from
    card.relation, which is correctly None for unstructured/extracted cards
    at inference time and would make every card compare equal to every other
    (None == None), silently producing zero negatives in those modes."""
    by_subject = defaultdict(list)
    by_relation = defaultdict(list)
    rel_of = {}
    for c in cards:
        g = golds.get(c.edit_id)
        rel = g.relation_id if g else None
        rel_of[c.edit_id] = rel
        by_subject[c.subject].append(c)
        if rel:
            by_relation[rel].append(c)

    negs = []
    for c in cards:
        gold = golds.get(c.edit_id)
        q = _query_for(c, gold)
        if not q:
            continue
        my_rel = rel_of[c.edit_id]
        # type 1: same subject, different relation
        siblings = [o for o in by_subject[c.subject] if o.edit_id != c.edit_id and rel_of[o.edit_id] != my_rel]
        for o in siblings[:2]:
            negs.append(NegativeExample(q, o.edit_id, "same_subject_diff_relation", c.edit_id, _card_text(o)))
        # type 2: same relation, different subject
        if my_rel:
            cousins = [o for o in by_relation[my_rel] if o.edit_id != c.edit_id and o.subject != c.subject]
            for o in cousins[:2]:
                negs.append(NegativeExample(q, o.edit_id, "same_relation_diff_subject", c.edit_id, _card_text(o)))
    return negs


def build_neighbor_negatives(cards, golds, index, topk=8, per_query=3):
    """Categories 3, 6, 8: nearest-neighbor-based negatives from a pre-built
    DenseCardIndex over `cards`. Distinguishes 'nearest semantic neighbor
    belonging to another edit' (rank 1-2, clearly wrong) from 'lexically
    similar but irrelevant' (mid-rank, still wrong) from 'ranked below correct'
    (any wrong candidate below the true card's own rank) -- three different
    signals for the verifier even though they share a retrieval mechanism."""
    negs = []
    for c in cards:
        gold = golds.get(c.edit_id)
        q = _query_for(c, gold)
        if not q:
            continue
        results = index.query(q, topk=topk)   # [(card, score), ...] desc by score
        ids = [r[0].edit_id for r in results]
        true_rank = ids.index(c.edit_id) + 1 if c.edit_id in ids else None
        wrong = [(r[0], i) for i, r in enumerate(results) if r[0].edit_id != c.edit_id]
        if not wrong:
            continue
        # type 3: nearest wrong neighbor (rank 1-2 among wrong candidates)
        for card_w, _rank in wrong[:1]:
            negs.append(NegativeExample(q, card_w.edit_id, "nearest_semantic_neighbor", c.edit_id, _card_text(card_w)))
        # type 6: mid-rank wrong neighbor (rank 3+), still similar enough to
        # have made the candidate set at all
        for card_w, _rank in wrong[2:2 + 1]:
            negs.append(NegativeExample(q, card_w.edit_id, "lexically_similar_irrelevant", c.edit_id, _card_text(card_w)))
        # type 8: any wrong candidate ranked BELOW the true card specifically
        # (only meaningful when the true card was actually retrieved)
        if true_rank is not None:
            below = [r[0] for i, r in enumerate(results) if i + 1 > true_rank][:per_query]
            for card_w in below:
                negs.append(NegativeExample(q, card_w.edit_id, "ranked_below_correct", c.edit_id, _card_text(card_w)))
    return negs


def build_stale_object_negatives(cards, golds, dataset_name):
    """Category 4: old/stale object for the same subject-relation slot.
    Only meaningful for WikiUpdate, the one dataset with explicit
    time_true/time_new validity intervals -- the query (about the NEW fact)
    paired against a synthetic card built from target_true (the OLD fact) is
    exactly the 'stale answer' confusion a real deployed memory would hit."""
    if dataset_name != "WikiUpdate":
        return []
    negs = []
    for c in cards:
        gold = golds.get(c.edit_id)
        if not gold or not gold.target_true:
            continue
        q = _query_for(c, gold)
        if not q:
            continue
        # synthetic stale-card id: not a real KnowledgeCard, just a label the
        # verifier training loop treats as "the old answer text", constructed
        # here rather than smuggled through akew_data (keeps the gold-field
        # guard meaningful: this module reads target_true explicitly as an
        # evaluator-side operation on already-separated GoldRecords, not
        # during ingestion).
        stale_id = f"{c.edit_id}__stale"
        card_text = _card_text(c)
        if gold.target_new and str(gold.target_new) in card_text:
            stale_text = card_text.replace(str(gold.target_new), str(gold.target_true))
        else:
            stale_text = f"{c.subject} {gold.target_true}"
        negs.append(NegativeExample(q, stale_id, "stale_object_same_slot", c.edit_id, stale_text))
    return negs


def build_sibling_triple_negatives(cards, golds, raw_triplets_by_edit):
    """Category 5: another triplet from the same raw document/edit. Only
    meaningful in extracted mode, where a single edit can yield multiple
    triples and a verifier must learn that 'came from the same source' is not
    the same as 'answers this specific question'. raw_triplets_by_edit:
    edit_id -> list of triple dicts (subject/prompt/target), same shape as
    unsfact_triplets_GPT, passed in by the caller since akew_data's cards
    already collapse triples into one joined text string."""
    negs = []
    for c in cards:
        triples = raw_triplets_by_edit.get(c.edit_id, [])
        if len(triples) < 2:
            continue
        gold = golds.get(c.edit_id)
        q = _query_for(c, gold)
        if not q:
            continue
        # the query asks about the EDIT's own relation; a sibling triple about
        # a different aspect of the same source is a same-document distractor
        for t in triples[1:3]:
            sib_text = f"{t.get('prompt','{}').replace('{}', t.get('subject',''))} {t.get('target','')}"
            negs.append(NegativeExample(q, f"{c.edit_id}__sibling", "same_document_sibling_triple",
                                        c.edit_id, sib_text))
    return negs


def build_prefixed_unrelated_negatives(cards, golds, seed=0):
    """Category 7: an unrelated question prefixed with text from a DIFFERENT
    edited question, testing whether the verifier is fooled by surface overlap
    with an edited prompt's phrasing rather than its actual content."""
    rng = random.Random(seed)
    pool = [(c, golds.get(c.edit_id)) for c in cards]
    pool = [(c, g) for c, g in pool if g and g.eval_question]
    negs = []
    for c, g in pool:
        other_c, other_g = rng.choice(pool)
        if other_c.edit_id == c.edit_id:
            continue
        prefix = other_g.eval_question.split(" ")[:4]
        fused_q = " ".join(prefix) + " ... " + g.eval_question
        # this fused query should NOT match other_c (it's not really about
        # other_c's fact, just superficially phrased to start like it) --
        # the true label is: matches c's own card, NOT other_c's
        negs.append(NegativeExample(fused_q, other_c.edit_id, "prefixed_unrelated_question",
                                    c.edit_id, _card_text(other_c)))
    return negs


def build_all_hard_negatives(cards, golds, index, dataset_name, raw_triplets_by_edit=None):
    negs = []
    negs += build_specificity_negatives(cards, golds)
    negs += build_neighbor_negatives(cards, golds, index)
    negs += build_stale_object_negatives(cards, golds, dataset_name)
    if raw_triplets_by_edit:
        negs += build_sibling_triple_negatives(cards, golds, raw_triplets_by_edit)
    negs += build_prefixed_unrelated_negatives(cards, golds)
    return negs
