# AKEW iterative multi-hop pilot — 2026-08-19

Oracle-decomposition iterative loop (`akew_multihop.py`, brief section 5:
identify next sub-question -> retrieve+verify -> answer -> append -> stop
after 3 hops or no-progress) versus a naive single-shot baseline (no
decomposition at all: retrieve once against the full composed question,
generate once). MQuAKE-CF, structured mode, n=80, deterministic decoding,
`Qwen/Qwen2.5-1.5B-Instruct`.

| strategy | accuracy |
|---|---|
| iterative multihop | 5.0% |
| naive single-shot | 22.5% |

**The iterative loop performs 4.5x worse than doing nothing clever at all.**
This is a real, diagnosed finding, not noise -- every sampled failure showed
`multihop_final_answer: null`, meaning the loop was breaking before producing
any answer at all, not just answering wrong.

## Root cause, confirmed by inspecting the raw data directly

MQuAKE-CF's multi-hop chains mix **edited** facts (the ones actually in
`requested_rewrite`, and therefore the only facts present in the card index)
with **unedited, ordinary world facts** that the chain still has to pass
through. Case 1, for example: 1 edit (Ellie Kemper's citizenship -> Croatia),
but a 2-hop chain: hop 1 is the edited fact (in the index), hop 2 asks for
Croatia's head of state -- an ordinary fact never touched by any edit, and
therefore **not present in the card index at all**.

The current loop has no fallback for this: when retrieval finds nothing
genuinely relevant, the verifier correctly scores it low, and the loop
**breaks with no answer** rather than falling back to the base model's own
parametric knowledge for that hop. Given most MQuAKE-CF groups have 1 edit
across a 2-3 hop chain (277/354 groups have exactly 1 edit), this
edited-vs-unedited-hop gap is the dominant case, not an edge case -- which is
exactly why the loop's answer-production rate collapsed to near zero.

The naive single-shot baseline accidentally handles this better: it retrieves
one relevant card (usually the actually-edited fact, when retrieval finds it)
and asks the model to answer the FULL composed question using that one piece
of context plus whatever the model already knows -- which is structurally
closer to what this benchmark actually requires (edited fact as evidence,
everything else as ordinary parametric knowledge) than a loop that treats
every hop as something that must be retrieved and verified against the edit
index specifically.

## What this changes about the next step

Not "tune the threshold" or "add more training data" -- the verifier itself
is not the bottleneck here (its recall on true positives was already ~99.9%
in the zero-shot MQuAKE-CF evaluation). The fix is architectural: **a hop
that finds no relevant edited card should fall back to asking the base model
to answer that hop from its own knowledge**, not treat retrieval-miss as
terminal failure. That turns the loop from "retrieve-verify-answer at every
hop, unconditionally" into something closer to the router's own REJECT/
DIRECT/REASON logic applied per-hop: REASON when a relevant edit is found,
REJECT-but-continue (answer from base knowledge, keep going) when it isn't,
and only truly stop on repeated no-progress. Not built in this pass --
recorded as the concrete next architectural change, not a vague TODO.

## Scope note

This pilot used oracle sub-question decomposition (MQuAKE-CF's own
`new_single_hops`), not self-decomposition by the model -- see
`akew_multihop.py`'s docstring for why that's an honest, deliberate scoping
choice, isolating the retrieve-verify-answer composition question from the
separate, harder decomposition problem MeLLo's actual method addresses.
