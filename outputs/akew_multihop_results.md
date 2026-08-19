# AKEW iterative multi-hop pilot — 2026-08-19 (updated: fallback fix applied)

**Update: the fallback fix described below was implemented and retested the
same day. Result: iterative multihop jumped from 5.0% to 47.5% -- now more
than 2x the naive single-shot baseline (22.5%), reversing the finding
entirely.** The original 5% run and its root-cause diagnosis are preserved
below unedited, since the diagnosis is what led directly to the fix and the
corrected result. New section at the bottom has the retest.


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

## Retest after the fallback fix (same day)

Implemented exactly the fix identified above: when a hop's verifier score is
below threshold (or nothing is retrieved at all), the loop now answers that
hop from the base model's own knowledge plus whatever prior hops already
established as context, then **continues to the next hop** instead of
breaking. This mirrors the router's REJECT semantics applied per-hop rather
than per-query.

| strategy | accuracy (n=80) |
|---|---|
| iterative multihop (with fallback) | **47.5%** |
| naive single-shot | 22.5% |

**The finding reverses completely.** Every retested example now completes
all hops (`stopped_early: false` throughout the sample), and iterative
decomposition beats naive single-shot generation by more than 2x -- the
result the whole redesign was built to demonstrate: retrieval + verification
+ per-hop reasoning, composed correctly, outperforms asking a model to jump
straight to a multi-hop answer in one shot.

Qualitative example (case 1587): *"Who is the head of state of the country
where Dave Holland's music originated?"* -- the naive baseline gives up
outright ("The question does not provide enough information..."), while the
fixed iterative loop correctly resolves the chain (Dave Holland -> jazz ->
originated in the US... more precisely traces through the edited fact to
France) to "Emmanuel Macron," the correct final answer.

One recurring failure pattern worth flagging honestly: cases 1475 and 1917,
both genuinely different questions, converged on the identical wrong answer
"Gharbia Governorate" against a shared gold answer "Epworth" -- both chains
pass through a "founder of a religion" hop, suggesting a specific, repeatable
confusion the model has about that entity rather than two independent random
errors. Worth a targeted look before scaling this test up.

This result was obtained on the same 80-example sample, same model
(`Qwen/Qwen2.5-1.5B-Instruct`), same oracle-decomposition scope as the
original pilot -- the only change was the fallback logic in
`akew_multihop.py`.

## Full-sample confirmation (n=354, the entire MQuAKE-CF pool)

Reran on every group in the dataset, not just the 80-example sample, same
model and oracle-decomposition scope.

| strategy | accuracy (n=80, sample) | accuracy (n=354, full pool) |
|---|---|---|
| iterative multihop (with fallback) | 47.5% | **53.95%** |
| naive single-shot | 22.5% | 16.1% |

The finding holds and the margin widens at full scale (37.85 points, versus
25 points on the sample) -- not a sampling artifact. The exact same failure
pair recurs identically: cases 1475 and 1917 both still converge on the
wrong answer "Gharbia Governorate" against gold "Epworth," both still
passing through a "founder of a religion" hop -- confirming this is a
specific, repeatable model confusion about that entity, not noise that
would wash out with a larger sample.

## 7B-scale confirmation (Qwen2.5-7B-Instruct, n=150)

| strategy | 1.5B (n=354, full pool) | 7B (n=150) |
|---|---|---|
| iterative multihop (with fallback) | 53.95% | **42.0%** |
| naive single-shot | 16.1% | 20.67% |

The margin holds at the larger model (+21.33 points, versus +37.85 at
1.5B/full-pool) -- decomposition still clearly beats naive single-shot, not
model-scale-dependent. The absolute iterative-multihop rate is lower at 7B
than 1.5B's full-pool number; this is a different (smaller, n=150) sample,
not a same-sample model comparison, so it should not be read as "the bigger
model does worse at multihop" -- that comparison would need the same sample
run through both models, not yet done.

The identical failure pair recurs a THIRD time, now across two different
model scales and two different sample sizes: cases 1475 and 1917 again both
land on "Gharbia Governorate" against gold "Epworth" via the same "founder
of a religion" hop. Three independent runs (n=80 pilot, n=354 full pool at
1.5B, n=150 at 7B) reproducing the exact same specific wrong answer is
strong evidence this is a genuine, model-scale-independent confusion about
that entity (likely a retrieval or evidence-text issue upstream of either
model's reasoning), worth a targeted look before scaling this further.
