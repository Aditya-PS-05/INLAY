# Baseline comparison: INLAY vs plain RAG vs IKE — 2026-08-19

Real (non-oracle) pipeline throughout: dense retrieval, real generation, no
oracle shortcuts. CounterFact unstructured, n=147, `Qwen/Qwen2.5-1.5B-Instruct`.

- **INLAY routed pipeline**: retrieval + v2 verifier + router (REJECT/DIRECT/
  REASON) + the corresponding answering strategy.
- **Plain RAG**: retrieve top-1, inject as context, generate. No gating, no
  demonstrations. Brief section 6, method 9.
- **IKE**: RAG plus 2 demonstration examples (other cards from the index,
  each showing "here is a new fact, use it instead of what you previously
  believed") before the actual query. Brief section 6, method 10. Scoping
  note: demonstrations are selected at random from the pool here, not via
  IKE's own similarity-based selection procedure -- a faithful reproduction
  of IKE's mechanism (few-shot override demonstrations), not a byte-for-byte
  port of its original demonstration-retrieval pipeline.

## Results

| method | accuracy |
|---|---|
| INLAY routed pipeline | 87.07% |
| plain RAG | 87.07% |
| **IKE (with demonstrations)** | **90.48%** |

INLAY and RAG are identical here because the router correctly routes 100% of
these queries to REASON (per `akew_fullpipeline_results.md`'s router fix),
which is mechanically the same operation as plain RAG on this dataset/mode --
expected, not a bug.

## Honest reading: IKE wins here, and that's worth taking seriously

Few-shot demonstrations showing the override pattern genuinely help beyond
single-fact context injection alone -- a real, positive result for IKE, not
smoothed over because it doesn't flatter INLAY. Looking at the sample
outputs, IKE's answers tend to be more confidently stated in full sentences
following the demonstrated pattern ("X found employment in Rome as a..."),
while INLAY/RAG's shorter answers are functionally equivalent when correct
but occasionally more clipped.

One shared failure across all three methods is worth noting: "The Million
Second Quiz was released on what?" (gold: CBS) -- all three answered
"September 2013," the show's air date, not its network. Since INLAY, RAG,
and IKE all failed identically here, this points to a retrieval/evidence
issue (the stored evidence text likely foregrounds the date over the
network) rather than an answering-strategy failure any of the three methods
could fix independently -- a shared upstream limitation, not three
independent bugs.

## What this means for INLAY's positioning

IKE's demonstration mechanism is cheap to add and empirically helps. This is
a concrete, low-cost improvement worth folding into the REASON path: instead
of feeding only the current retrieved evidence, include 1-2 demonstration
examples from the memory showing the override pattern, exactly what IKE
does, combined with INLAY's own retrieval+verification+routing machinery
(which IKE and plain RAG lack entirely -- neither has a REJECT path, so both
would hallucinate confidently on queries INLAY's router correctly declines,
untested here since this comparison ran only on already-confident-retrieval
queries). Not yet tested: whether IKE's demonstration boost holds up when
retrieval is uncertain (WikiUpdate-style), where INLAY's REJECT path has
already shown real, measured value that plain RAG/IKE structurally cannot
replicate.

## Follow-up: folding IKE's demonstrations into INLAY's own REASON path

Directly tested the "what this means for INLAY's positioning" suggestion
above rather than leaving it as a proposal. `answer_contextual()` gained an
optional `demonstrations` parameter (backward-compatible, `None` by default
reproduces every prior result exactly); the REASON path now optionally
includes 2 IKE-style demonstrations on top of INLAY's full retrieval/
verification/routing machinery.

| | accuracy |
|---|---|
| INLAY routed, no demonstrations | 87.07% |
| **INLAY routed + IKE demonstrations** | **89.12%** |
| plain IKE (no retrieval/verification/REJECT) | 90.48% |

A real, honest +2.05 point improvement from adding demonstrations to INLAY's
own pipeline -- confirms the mechanism transfers, worth keeping. It still
trails plain IKE by 1.36 points on this specific dataset/mode, where
retrieval is already near-perfect and REJECT never fires, so INLAY's extra
machinery adds cost without adding value here. The real test of whether
INLAY+IKE beats plain IKE is a dataset where REJECT matters (WikiUpdate,
which plain IKE/RAG cannot benefit from having no REJECT path at all) --
not yet run.

## The demonstration boost does NOT transfer to WikiUpdate -- it reverses

Ran the exact same INLAY+IKE-demonstrations comparison on WikiUpdate
unstructured (n=160), the harder-retrieval dataset. The hoped-for result was
INLAY+IKE beating plain IKE here specifically, since REJECT should matter on
a dataset where retrieval fails ~28% of the time. Instead, a different and
more important finding emerged:

| | accuracy |
|---|---|
| INLAY routed, no demonstrations | 43.75% |
| **INLAY routed + IKE demonstrations** | **40.62%** |

**Demonstrations hurt here, the opposite of CounterFact's +2.05 point gain.**
Confirmed the drop is entirely attributable to the demonstration mechanism
itself, not a code bug: REJECT-routed queries (42/160) produce byte-identical
answers with and without demonstrations by construction (demonstrations only
apply on the REASON path), so the full ~3-point gap is concentrated in the
118 REASON-routed queries.

**Honest reading:** this pilot's demonstration selection is random sampling
from the card pool (`akew_baseline_ike.py`'s own docstring already flags
this as a scoping simplification, not IKE's real similarity-based selection
procedure). On CounterFact, where facts are largely independent of each
other, random demonstrations are harmless noise that still teaches the
override pattern. On WikiUpdate, whose real-world entities collide far more
(the same structural property behind its harder retrieval and its unique
stale-object confusions), a randomly-selected demonstration is more likely
to introduce genuinely distracting or superficially-similar-but-irrelevant
context, actively hurting rather than teaching. **The demonstration
mechanism is not a free win to deploy universally** -- it needs IKE's actual
similarity-based demonstration selection, not this pilot's random-sampling
simplification, before it can be trusted on datasets with WikiUpdate's
collision structure. Recorded as a genuine negative result, not smoothed
into the earlier positive CounterFact finding.

## Does IKE's real similarity-based selection fix the WikiUpdate regression?

The obvious next hypothesis: the random demonstration selection is the
problem, and IKE's real similarity-based selection (its actual published
mechanism, not this project's random-sampling simplification) would fix it.
Implemented `build_demonstrations_similarity()` (nearest cards to the query
by embedding similarity, excluding the true card) and tested it directly
rather than assuming.

| demonstration selection | accuracy |
|---|---|
| none | 43.75% |
| random | 40.62% |
| **similarity-based (IKE's real mechanism)** | **38.75%** |

**Similarity-based selection makes it WORSE, not better.** This decisively
answers the question the earlier scoping note left open: the problem is not
that this project used a cheap random-selection stand-in for IKE's real
mechanism. On a collision-heavy dataset like WikiUpdate, a demonstration
that is MORE similar to the query is MORE likely to be a near-duplicate,
easily-confused fact (the same structural property behind WikiUpdate's
stale-object confusions and harder retrieval), not less -- the opposite of
demonstrations' intended effect. **The fix is not a better demonstration-
selection algorithm** -- selection isn't the axis that matters here (see
correction below).

**Correction (see "Plain IKE on WikiUpdate" further down this page):**
this section's closing claim -- that the demonstration mechanism itself is
"fundamentally at odds with high-entity-collision data" and should be gated
off entirely -- turned out to be too strong. A follow-up segmented
comparison found the real variable wasn't demonstrations vs no
demonstrations, but which INSTRUCTION FRAMING wraps them:
`answer_contextual`'s "answer only from the evidence above" framing (used
in this section's INLAY+IKE test) actively hurts when combined with
demonstrations on this subset (57.63% -> 53.39%), while IKE's own "trust
the new fact over your prior belief" override framing, tested with the
identical demonstrations on the identical subset, helps (57.63% -> 62.71%).
The demonstration content was never the problem; the restrict-to-evidence
instruction paired with it was. See below for the full breakdown.

## Plain IKE on WikiUpdate: closing the one comparison never run

Every prior comparison on this page ran CounterFact unstructured. The
`akew_baseline_compare.py` three-way comparison (INLAY routed vs plain RAG
vs plain IKE, no INLAY machinery folded into IKE) had never been run on
WikiUpdate -- only the INLAY+IKE-demonstrations variant above was. Closing
that gap, same setup (n=160, WikiUpdate unstructured).

| method | accuracy |
|---|---|
| plain RAG | 43.13% |
| **INLAY routed pipeline** | 43.75% |
| **plain IKE (with demonstrations)** | **46.88%** |

**Plain IKE beats INLAY here too**, by 3.13 points -- at first this looks
like it contradicts the finding just above that demonstrations HURT when
folded into INLAY's own REASON path (43.75% -> 40.62%). Both use the
identical random-demonstration mechanism (`build_demonstrations`, n=2,
seed=0), so the difference has to be structural, not mechanism-level:
INLAY+IKE only adds demonstrations to REASON-routed queries, leaving the
42/160 REJECT-routed queries answered identically (`answer_no_context`) in
both conditions. Plain IKE has no REJECT gate at all -- it retrieves and
generates with demonstrations on every query unconditionally, including the
ones INLAY's router declines to even attempt.

Tested the resulting hypothesis directly with a segmented rerun
(`akew_baseline_compare_segmented.py`), splitting accuracy by the router's
own REJECT vs non-REJECT decision on the identical test split.

**The REJECT-subset hypothesis was wrong.** On the 42 REJECT-routed
queries, INLAY's answer-from-parametric-knowledge-alone actually edges out
both RAG and plain IKE's attempt-anyway strategy (4.76% vs 2.38%), not the
other way around -- consistent with the original framing that REJECT is
declining for a reason, and attempting an answer on a genuinely bad
retrieval doesn't reliably beat declining. This subset is small (n=42) and
both numbers are near-floor, so read this as "REJECT is not obviously
worse here," not as a strong positive result for REJECT specifically.

| subset (n) | plain RAG | INLAY routed | plain IKE |
|---|---|---|---|
| REJECT subset (n=42) | 2.38% | **4.76%** | 2.38% |
| non-REJECT subset (n=118) | 57.63% | 57.63% (identical to RAG, expected) | **62.71%** |

**The real driver is the non-REJECT subset, and it exposes a real
methodological difference, not a contradiction.** `answer_ike` (used here)
and `answer_contextual(..., demonstrations=...)` (used in the INLAY+IKE
comparison above) both prepend demonstration examples, but with different
instruction framing:

- `answer_contextual`: *"Answer based only on the evidence above, in a few
  words"* -- a restrictive instruction that tells the model to stick to the
  retrieved evidence specifically.
- `answer_ike`: *"Given this new fact, answer using it, not what you
  previously believed"* -- IKE's actual override framing, paired directly
  with the fact and carrying no "only the evidence" restriction.

Back-calculating the REASON-subset accuracy the earlier INLAY+IKE comparison
implies (its reported 40.62% overall, minus this run's byte-identical
REJECT-subset contribution, since demonstrations only touch the REASON
path): **53.39%** (63/118) for the restrictive INLAY framing with demos,
versus this run's **62.71%** (74/118) for IKE's own override framing with
the identical demonstrations, versus **57.63%** (68/118) with no
demonstrations at all.

**The corrected finding: it isn't "demonstrations hurt on WikiUpdate."**
It's that pairing demonstrations with an "answer only from the evidence"
restriction hurts (57.63% -> 53.39%), while pairing the same demonstrations
with IKE's actual override instruction helps (57.63% -> 62.71%) --
opposite directions from the identical demonstration content. The
restrictive framing appears to compound WikiUpdate's collision problem: it
tells the model to trust ONLY the possibly-wrong retrieved evidence, with
no permission to fall back on the demonstrated override pattern's implicit
"trust the new fact over your prior" license the way IKE's own framing
does. This changes the actionable recommendation from "gate demonstrations
off for collision-heavy datasets" to "use IKE's own override framing, not a
restrict-to-evidence framing, when combining demonstrations with retrieved
context" -- a fixable prompt-design choice, not a fundamental incompatibility
between demonstrations and this class of dataset.
