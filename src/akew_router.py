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
    # Predicted P(top-1 retrieval is correct) from the reliability head, when
    # one is attached; None when running without it (the default), so every
    # pre-existing caller and every prior result is unaffected.
    predicted_reliability: Optional[float] = None


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
    def __init__(self, retrieval_index, verifier, reject_threshold=0.65, direct_threshold=0.85, topk=5,
                 reliability_head=None, bypass_threshold=0.5, reject_floor=None):
        """reliability_head: optional trained ReliabilityHead (akew_reliability.py).

        When attached, the router first asks whether its own gating signal is
        TRUSTWORTHY for this query -- P(top-1 retrieval is correct), predicted
        from the shape of the retrieval+verification result -- and bypasses
        both gates (forcing REASON) when that probability falls below
        bypass_threshold.

        Why this exists: akew_fullpipeline_results.md documents REJECT/DIRECT
        gating being net-negative on MQuAKE-CF in EVERY input mode, while
        helping on CounterFact/WikiUpdate in every mode. Two separate
        experiments (verifier recalibration; a direct_threshold sweep at
        0.85/0.97/1.01) both showed no scalar threshold on the top-1 score can
        fix it, because the verifier is confidently WRONG there -- its score
        distributions genuinely overlap. The head asks a second-order question
        a first-order threshold cannot express: not "is this score high?" but
        "is this score discriminative?"

        reject_floor: optional SECOND, lower reliability threshold enabling a
        three-way policy instead of a binary bypass. Motivated by the one cell
        the binary version lost (akew_reliability_head_results.md, WikiUpdate
        unstructured): there the head's DETECTION was its best anywhere (95.6%
        recall of bad retrievals) but the hard-coded response was wrong,
        because MQuAKE-CF and WikiUpdate want OPPOSITE responses to the same
        signal -- MQuAKE-CF's low-confidence retrievals still carry usable
        signal (reason over them), while WikiUpdate's are actively misleading
        stale/current collisions (decline instead).

        The discriminating observation: WikiUpdate's bad cases score LOWER on
        predicted reliability (mean 0.6596) than MQuAKE-CF's do (0.7492), so
        the MAGNITUDE of predicted unreliability -- not merely whether it
        crossed one line -- carries the missing signal. With reject_floor set:

            p <  reject_floor      -> REJECT   (actively untrustworthy)
            p <  bypass_threshold  -> REASON   (imperfect but usable)
            otherwise              -> normal fixed gating

        None (the default) keeps the binary behaviour, so every number already
        reported with the binary policy is reproduced exactly.

        reliability_head=None likewise reproduces the exact pre-head behaviour
        for every existing caller, down to using the identical single-pair
        verifier call rather than the batched top-k one -- additive, not a
        silent change to any previously reported number.
        """
        self.index = retrieval_index
        self.verifier = verifier
        self.reject_threshold = reject_threshold
        self.direct_threshold = direct_threshold
        self.topk = topk
        self.reliability_head = reliability_head
        self.bypass_threshold = bypass_threshold
        if reject_floor is not None and reject_floor > bypass_threshold:
            raise ValueError(
                f"reject_floor ({reject_floor}) must be <= bypass_threshold "
                f"({bypass_threshold}); a floor above the bypass line would "
                f"make the REASON band empty and silently turn the three-way "
                f"policy back into a binary one with a different threshold.")
        self.reject_floor = reject_floor

    def route(self, query_text):
        candidates = self.index.query(query_text, topk=self.topk)
        if not candidates:
            return RouteDecision("REJECT", None, None, None, "no_candidates")

        top_card, retrieval_score = candidates[0]
        import numpy as np

        predicted_reliability = None
        if self.reliability_head is not None:
            # The head needs verifier scores for ALL top-k candidates, and the
            # top-1 score it needs anyway is the first element -- so this path
            # computes both in one batched call rather than scoring top-1
            # twice. Cost note stated plainly: k cross-encoder pairs per query
            # instead of 1 (k=5 by default).
            from akew_reliability import score_candidates
            all_scores = score_candidates(self.verifier, query_text, candidates)
            verifier_score = all_scores[0]
            pred = self.reliability_head.predict_one(
                query_text, candidates, all_scores, self.direct_threshold)
            predicted_reliability = pred.p_correct

            if self.reject_floor is not None and predicted_reliability < self.reject_floor:
                # Predicted so unreliable that the retrieved evidence is more
                # likely to mislead than to help -- decline rather than reason
                # over it. This is the band the binary policy was missing.
                return RouteDecision("REJECT", top_card.edit_id, retrieval_score, verifier_score,
                                     "below_reliability_reject_floor", predicted_reliability)

            if predicted_reliability < self.bypass_threshold:
                # Both gates are suppressed together, deliberately: the
                # threshold sweep in akew_fullpipeline_results.md showed
                # disabling DIRECT alone still lost to always-REASON (90.48%
                # vs 96.83%), because the REJECT gate was independently
                # net-negative in the same regime. Half a bypass was already
                # tested and was not enough.
                return RouteDecision("REASON", top_card.edit_id, retrieval_score, verifier_score,
                                     "low_predicted_reliability_bypass", predicted_reliability)
        else:
            cand_text = top_card.canonical_fact_text or top_card.raw_evidence_text or ""
            raw_score = self.verifier.predict([[query_text, cand_text]], convert_to_numpy=True, show_progress_bar=False)[0]
            verifier_score = float(1 / (1 + np.exp(-raw_score)))

        if verifier_score < self.reject_threshold:
            return RouteDecision("REJECT", top_card.edit_id, retrieval_score, verifier_score,
                                 "below_reject_threshold", predicted_reliability)

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
            return RouteDecision("DIRECT", top_card.edit_id, retrieval_score, verifier_score,
                                 "high_confidence_atomic_structured", predicted_reliability)

        return RouteDecision("REASON", top_card.edit_id, retrieval_score, verifier_score,
                              "relevant_but_not_direct_confidence" if verifier_score < self.direct_threshold
                              else ("multihop_cue" if _looks_multihop(query_text) else "non_structured_mode"),
                              predicted_reliability)
