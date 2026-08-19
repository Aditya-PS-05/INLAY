# Retrieval ablations: JL projection and top-k — 2026-08-19

Full data (not sampled), structured mode, all three datasets.

## JL projection vs raw embeddings

`akew_retrieval.py` deliberately ships with no JL projection, flagged in its
own docstring as an ablation to test rather than an assumption inherited from
INLAY. Result: **raw embeddings match or beat JL-projected (128d) embeddings
everywhere.**

| dataset | raw R@1 | JL-projected R@1 | delta |
|---|---|---|---|
| CounterFact | 0.999 | 0.998 | -0.001 |
| WikiUpdate | 0.996 | 0.981 | **-0.015** |
| MQuAKE-CF | 0.773 | 0.771 | -0.002 |

WikiUpdate shows the clearest cost (1.5 points of Recall@1 lost to
projection), the others are within noise. **Confirms the design choice was
correct for AKEW's scale**: JL projection exists to compress storage and
compute for millions of slots (INLAY's own product-key memory, entry 10 of
the interview lexicon); at AKEW's ~1,000-edit scale, there is no storage
problem to solve, so the projection only costs retrieval fidelity with no
compensating benefit. Not a criticism of JL projection in general -- a
confirmation that INLAY's own choice for that regime doesn't transfer to a
regime where the tradeoff it was built for doesn't apply.

## top-k sweep

| dataset | k=1 R@1 | k=5 no-cand rate | k=10 no-cand rate | k=10 MRR |
|---|---|---|---|---|
| CounterFact | 0.999 | 0.0% | 0.0% | 0.9995 |
| WikiUpdate | 0.996 | 0.0% | 0.0% | 0.9979 |
| MQuAKE-CF | 0.773 | 8.26% | **2.98%** | 0.836 |

CounterFact and WikiUpdate are already saturated at k=1 -- retrieving more
candidates does essentially nothing for them. **MQuAKE-CF is the only
dataset where top-k matters**: going from k=5 to k=10 cuts its no-candidate
rate nearly in half (8.26% -> 2.98%) and lifts MRR (0.8287 -> 0.836). This is
a real, cheap, actionable fix specifically for MQuAKE-CF's harder,
entity-dense retrieval problem -- worth wiring into the router as a
per-dataset or adaptive top-k rather than a fixed global k=5, a concrete
follow-up not yet implemented.
