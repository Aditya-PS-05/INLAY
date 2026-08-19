"""
Iterative multi-hop reasoning for MQuAKE-CF (brief section 5):
  1. identify the next missing sub-question
  2. retrieve and verify a fact for that sub-question
  3. answer the sub-question
  4. append the supported intermediate fact
  5. stop after the required answer, no-progress detection, or three hops

Every intermediate answer records its evidence card id.

Scoping choice, stated honestly rather than silently taken: step 1
("identify the next missing sub-question") uses MQuAKE-CF's own
`new_single_hops` decomposition as the sub-question SEQUENCE, read directly
from the raw AKEW json (an evaluator-side operation, same category as
akew_hard_negatives reading target_true for stale-object construction -- not
ingestion, not something a deployed router would see). This is an
oracle-decomposition diagnostic: it isolates whether retrieval + verification
+ answering compose correctly ACROSS hops, GIVEN a correct decomposition,
before adding the much harder, separate problem of a model decomposing an
unseen multi-hop question into sub-questions itself (MeLLo's actual
contribution, and real future work here -- not built in this pass).

No tree search, no beam search -- the brief explicitly says not to build
that until the simple iterative version has been measured.
"""
import os, json
from akew_answering import is_hit, _chat_generate


def load_mquake_raw(path=None):
    _here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(_here, "..", "..", "AKEW", "repo", "datasets", "MQuAKE-CF.json")
    return json.load(open(path))


class HopStep:
    def __init__(self, hop_idx, sub_question, retrieved_card_id, retrieval_score, verifier_score, answer, evidence_used):
        self.hop_idx = hop_idx
        self.sub_question = sub_question
        self.retrieved_card_id = retrieved_card_id
        self.retrieval_score = retrieval_score
        self.verifier_score = verifier_score
        self.answer = answer
        self.evidence_used = evidence_used   # list of prior hop answers folded into this hop's context

    def to_dict(self):
        return {"hop": self.hop_idx, "sub_question": self.sub_question, "card_id": self.retrieved_card_id,
                "retrieval_score": self.retrieval_score, "verifier_score": self.verifier_score,
                "answer": self.answer, "evidence_used": self.evidence_used}


def answer_multihop_group(rec, index, verifier, model, tok, device, max_hops=3,
                          verifier_threshold=0.5):
    """rec: one raw MQuAKE-CF record (from the AKEW json, not a KnowledgeCard --
    needed for new_single_hops, which lives only in the raw data).
    index: a DenseCardIndex built over the full MQuAKE-CF card set for this
    input mode, so retrieval can find any group's edits -- including OTHER
    groups' cards, whose interference is real signal, not something to
    suppress by artificially restricting the candidate pool to this group."""
    import numpy as np
    hops = rec.get("new_single_hops", [])[:max_hops]
    if not hops:
        return None

    steps = []
    accumulated_evidence = []
    for i, hop in enumerate(hops):
        sub_q = hop.get("question") or hop.get("cloze", "")
        if not sub_q:
            break
        candidates = index.query(sub_q, topk=3)
        if not candidates:
            steps.append(HopStep(i, sub_q, None, None, None, None, list(accumulated_evidence)))
            break
        top_card, retrieval_score = candidates[0]
        cand_text = top_card.canonical_fact_text or top_card.raw_evidence_text or ""
        raw_score = verifier.predict([[sub_q, cand_text]], convert_to_numpy=True, show_progress_bar=False)[0]
        v_score = float(1 / (1 + np.exp(-raw_score)))

        if v_score < verifier_threshold:
            # no-progress detection: this hop found nothing usable -- stop
            # here rather than hallucinating forward with no support.
            steps.append(HopStep(i, sub_q, top_card.edit_id, retrieval_score, v_score, None, list(accumulated_evidence)))
            break

        # answer this hop using contextual generation over the retrieved card
        # PLUS whatever prior hops already established (accumulated_evidence),
        # so hop 2 can build on hop 1's answer rather than reasoning in isolation.
        prior_ctx = "\n".join(f"- {e}" for e in accumulated_evidence)
        evidence_block = f"Updated evidence:\n{prior_ctx}\n- {cand_text}" if prior_ctx else f"Updated evidence:\n- {cand_text}"
        user_content = f"{evidence_block}\n\nQuestion: {sub_q}\nAnswer based only on the evidence above, in a few words."
        ans = _chat_generate(model, tok, user_content, device, max_new_tokens=15)

        steps.append(HopStep(i, sub_q, top_card.edit_id, retrieval_score, v_score, ans, list(accumulated_evidence)))
        accumulated_evidence.append(f"{sub_q} -> {ans}")

    final_answer = steps[-1].answer if steps else None
    gold_final = rec.get("new_answer")
    gold_aliases = rec.get("new_answer_alias", [])
    from akew_data import GoldRecord
    gold_obj = GoldRecord(edit_id="", target_new=gold_final, target_true=None,
                          aliases_new=gold_aliases, aliases_true=[])
    hit = bool(final_answer) and is_hit(final_answer, gold_obj)

    return {"case_id": rec.get("case_id"), "n_hops_attempted": len(steps), "n_hops_total": len(hops),
            "steps": [s.to_dict() for s in steps], "final_answer": final_answer,
            "gold_final": gold_final, "hit": hit,
            "stopped_early": len(steps) < len(hops)}
