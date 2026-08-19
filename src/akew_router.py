"""
The actual REJECT / DIRECT / REASON router (brief section 5), built from the
two components already trained and validated in section 4: DenseCardIndex
(Stage 1 retrieval) + the fine-tuned cross-encoder verifier (Stage 2 scope
check). This is the ONE shared routing function -- no downstream answering
path may re-derive a routing decision independently, matching the fix already
applied to INLAY's own gate-bypass bug (gpt2_memory_semkey.route()) and
carrying the same discipline into AKEW's separate router.

Decision rule:
  - no retrieval candidate at all, or verifier score < reject_threshold  -> REJECT
  - verifier score >= direct_threshold AND query looks single-hop/atomic -> DIRECT
  - otherwise (verifier says relevant, but not high-confidence enough or the
    query looks compositional)                                          -> REASON

Thresholds default to the values calibrated in section 4 (val threshold 0.65
for "is this relevant at all"); direct_threshold is set higher, since DIRECT
additionally claims high confidence, not just relevance -- brief section 4:
"score >= direct_threshold: DIRECT, but only for a clearly atomic, single-hop
query."
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteDecision:
    decision: str                  # "REJECT" | "DIRECT" | "REASON"
    card_id: Optional[str]
    retrieval_score: Optional[float]
    verifier_score: Optional[float]
    reason: str


def _looks_multihop(query_text):
    """Heuristic multi-hop / compositional query cue -- brief section 4: 'For
    multi-hop questions, prefer REASON even when a relevant candidate has a
    high score.' A real classifier could replace this; for the pilot, simple
    lexical cues catch the obvious cases without adding a third trained model."""
    q = query_text.lower()
    # crude but cheap: multi-hop questions in AKEW tend to nest a relation
    # inside another relation's answer slot ("the country that X is a citizen
    # of"), which shows up as more than one relation-shaped clause.
    relation_words = ("who", "what", "where", "which", "of the", "that")
    hits = sum(1 for w in relation_words if w in q)
    return hits >= 4 or " of the " in q and " that " in q


class AkewRouter:
    def __init__(self, retrieval_index, verifier, reject_threshold=0.65, direct_threshold=0.85, topk=5):
        self.index = retrieval_index
        self.verifier = verifier
        self.reject_threshold = reject_threshold
        self.direct_threshold = direct_threshold
        self.topk = topk

    def route(self, query_text):
        candidates = self.index.query(query_text, topk=self.topk)
        if not candidates:
            return RouteDecision("REJECT", None, None, None, "no_candidates")

        top_card, retrieval_score = candidates[0]
        cand_text = top_card.canonical_fact_text or top_card.raw_evidence_text or ""
        import numpy as np
        raw_score = self.verifier.predict([[query_text, cand_text]], convert_to_numpy=True, show_progress_bar=False)[0]
        verifier_score = float(1 / (1 + np.exp(-raw_score)))

        if verifier_score < self.reject_threshold:
            return RouteDecision("REJECT", top_card.edit_id, retrieval_score, verifier_score, "below_reject_threshold")

        # FIX (found by the full-pipeline test, akew_fullpipeline_results.md):
        # DIRECT's whole rationale is "a clean literal answer exists to recite
        # fast" -- true in structured mode (the card carries the literal
        # target), but NOT in unstructured/extracted mode, where "DIRECT"
        # degrades to reciting a raw evidence sentence (answer_hard_playback's
        # non-structured fallback), which Section 5's own pilots already
        # showed underperforms genuine contextual generation. The original
        # threshold-only logic was blind to input_mode and chose DIRECT for
        # ALL 147/147 unstructured CounterFact queries, when the full-
        # pipeline test showed forcing REASON instead scored higher (87.07%
        # vs 85.71%). DIRECT is now gated on structured mode specifically.
        if (verifier_score >= self.direct_threshold and not _looks_multihop(query_text)
                and top_card.input_mode == "structured"):
            return RouteDecision("DIRECT", top_card.edit_id, retrieval_score, verifier_score, "high_confidence_atomic_structured")

        return RouteDecision("REASON", top_card.edit_id, retrieval_score, verifier_score,
                              "relevant_but_not_direct_confidence" if verifier_score < self.direct_threshold
                              else ("multihop_cue" if _looks_multihop(query_text) else "non_structured_mode"))
