"""
IKE (in-context knowledge editing) baseline (brief section 6, method 10).

Distinct from plain RAG: IKE's actual contribution (entry 29 of the interview
lexicon) is demonstration selection -- worked examples showing the model HOW
to override its parametric prior with an injected fact, not just handing it
context and an instruction. "Plain RAG" (method 9) is already covered by this
project's own always_reason_no_routing condition (akew_fullpipeline_eval.py):
retrieve top-1, inject as context, generate, no demonstrations. This module
adds the demonstration layer IKE specifically contributes.

Demonstrations are constructed from OTHER cards in the same index (not the
query's own card), each showing: an injected fact -> a correctly-answered
question using that fact -- teaching the override pattern by example, exactly
IKE's mechanism, rather than IKE's own paper's approach of pre-computed
similarity-selected demonstrations from a large corpus (a full reimplementation
of IKE's original demonstration-retrieval pipeline is out of scope here; this
is a faithful mechanism-level reproduction of what the demonstrations DO, not
a byte-for-byte port of the original codebase).
"""
import random
from akew_answering import _chat_generate


def _card_text(card):
    return card.canonical_fact_text or card.raw_evidence_text or ""


def build_demonstrations(index, exclude_edit_id, n=2, seed=0):
    """Pick n other cards at random (excluding the query's own) to build
    worked demonstration examples from. A real, honest scoping simplification
    from IKE's own similarity-based selection -- see build_demonstrations_
    similarity() below for the real-IKE-mechanism variant, added specifically
    to test whether it fixes the WikiUpdate regression random selection
    caused (outputs/akew_baseline_comparison_results.md)."""
    rng = random.Random(seed)
    pool = [c for c in index.cards if c.edit_id != exclude_edit_id]
    picks = rng.sample(pool, min(n, len(pool)))
    demos = []
    for c in picks:
        text = _card_text(c)
        if not text:
            continue
        demos.append(f"New fact: {text}\nGiven this new fact, answer questions using it, "
                     f"not what you previously believed.")
    return demos


def build_demonstrations_similarity(index, query_text, exclude_edit_id, n=2, topk=8):
    """IKE's actual mechanism: demonstrations selected by similarity to the
    query, not at random. Retrieves the topk nearest cards to the query
    (excluding the query's own true card) and takes the n closest as
    demonstrations. Hypothesis being tested: random demonstrations on
    WikiUpdate's collision-heavy entity space introduced confusing,
    unrelated context (the negative finding in akew_baseline_comparison_
    results.md); similarity-selected demonstrations might be MORE
    confusable in a collision-heavy dataset (a demo about a near-duplicate
    entity competing with the real one), not less -- genuinely uncertain
    which direction this goes, which is exactly why it needs to be tested
    empirically rather than assumed."""
    candidates = index.query(query_text, topk=topk + 1)
    picks = [c for c, _score in candidates if c.edit_id != exclude_edit_id][:n]
    demos = []
    for c in picks:
        text = _card_text(c)
        if not text:
            continue
        demos.append(f"New fact: {text}\nGiven this new fact, answer questions using it, "
                     f"not what you previously believed.")
    return demos


def answer_ike(model, tok, query, card, index, device, max_new_tokens=20, n_demos=2, seed=0):
    demos = build_demonstrations(index, card.edit_id, n=n_demos, seed=seed)
    demo_block = "\n\n".join(demos)
    target_fact = _card_text(card)
    user_content = (
        f"{demo_block}\n\n"
        f"New fact: {target_fact}\n"
        f"Given this new fact, answer using it, not what you previously believed.\n"
        f"Question: {query}\nAnswer in a few words."
    )
    return _chat_generate(model, tok, user_content, device, max_new_tokens)
