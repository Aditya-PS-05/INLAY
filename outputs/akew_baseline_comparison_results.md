# Baseline comparison: CAKE vs plain RAG vs IKE — 2026-08-19

Real (non-oracle) pipeline throughout: dense retrieval, real generation, no
oracle shortcuts. CounterFact unstructured, n=147, `Qwen/Qwen2.5-1.5B-Instruct`.

- **CAKE routed pipeline**: retrieval + v2 verifier + router (REJECT/DIRECT/
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
| CAKE routed pipeline | 87.07% |
| plain RAG | 87.07% |
| **IKE (with demonstrations)** | **90.48%** |

CAKE and RAG are identical here because the router correctly routes 100% of
these queries to REASON (per `akew_fullpipeline_results.md`'s router fix),
which is mechanically the same operation as plain RAG on this dataset/mode --
expected, not a bug.

## Honest reading: IKE wins here, and that's worth taking seriously

Few-shot demonstrations showing the override pattern genuinely help beyond
single-fact context injection alone -- a real, positive result for IKE, not
smoothed over because it doesn't flatter CAKE. Looking at the sample
outputs, IKE's answers tend to be more confidently stated in full sentences
following the demonstrated pattern ("X found employment in Rome as a..."),
while CAKE/RAG's shorter answers are functionally equivalent when correct
but occasionally more clipped.

One shared failure across all three methods is worth noting: "The Million
Second Quiz was released on what?" (gold: CBS) -- all three answered
"September 2013," the show's air date, not its network. Since CAKE, RAG,
and IKE all failed identically here, this points to a retrieval/evidence
issue (the stored evidence text likely foregrounds the date over the
network) rather than an answering-strategy failure any of the three methods
could fix independently -- a shared upstream limitation, not three
independent bugs.

## What this means for CAKE's positioning

IKE's demonstration mechanism is cheap to add and empirically helps. This is
a concrete, low-cost improvement worth folding into the REASON path: instead
of feeding only the current retrieved evidence, include 1-2 demonstration
examples from the memory showing the override pattern, exactly what IKE
does, combined with CAKE's own retrieval+verification+routing machinery
(which IKE and plain RAG lack entirely -- neither has a REJECT path, so both
would hallucinate confidently on queries CAKE's router correctly declines,
untested here since this comparison ran only on already-confident-retrieval
queries). Not yet tested: whether IKE's demonstration boost holds up when
retrieval is uncertain (WikiUpdate-style), where CAKE's REJECT path has
already shown real, measured value that plain RAG/IKE structurally cannot
replicate.

## Scope note

This is one dataset/mode comparison (CounterFact unstructured). The weight-
editing methods (ROME/MEMIT/AlphaEdit/WISE/GRACE) and MeLLo remain out of
scope, unchanged from prior scope notes -- a separate multi-day harness-
engineering phase.
