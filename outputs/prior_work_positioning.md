# INLAY — prior-work positioning & novelty map

Purpose: before spending more compute, pin down exactly what INLAY is against the published editing
literature — what to cite, what to differentiate, and which novelty claims survive contact with the
field. Verdict up front: **the core mechanism is genuinely distinct, but INLAY sits inside an existing
family (memory/retrieval-based editing), so novelty must be claimed at the mechanism level, not the
paradigm level.** Overclaiming "first non-parametric editor" against SERAC/GRACE would draw an immediate
reject.

## The design space of knowledge editing (taxonomy for the related-work section)

**1. Locate-then-edit (modify weights).**
- **ROME** — *Locating and Editing Factual Associations in GPT* (2022). Rank-one update to one MLP layer,
  located by causal tracing. Our runs: strong single-edit, collapses under sequential editing (~50 edits).
- **MEMIT** — *Mass-Editing Memory in a Transformer* (2022). Spreads ROME-style updates across several
  layers for batch/mass editing; under-installs single edits (our CF single-edit ES≈0).
- **AlphaEdit** — *Null-Space Constrained Knowledge Editing* (2024). Projects the ROME/MEMIT perturbation
  into the null space of preserved knowledge to protect locality in sequential editing. Our runs:
  locality-first (NS 0.98), under-installs single edits (ES 0.46). The abstract's own framing —
  "perturbation inevitably disrupts preserved knowledge, especially in sequential editing" — is precisely
  the failure mode INLAY avoids by never touching weights.

**2. Meta-learning a weight update.** MEND, KnowledgeEditor (train a hypernetwork to produce the edit).
Not run here; cite as a distinct branch.

**3. Add capacity / patch neurons (parametric, additive).**
- **Transformer-Patcher** (2023) — one neuron per mistake in the last FFN.
- **MELO** (2024, AAAI) — neuron-indexed dynamic LoRA blocks keyed by an inner index.
- **GRACE** — *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors* (2022). A
  codebook of discrete key→value adaptors at one layer; activates only when a runtime activation lands
  within eps of a stored key. **This is INLAY's closest structural relative** (discrete addressable
  memory, non-destructive) and must be differentiated carefully (below).

**4. Memory / retrieval-based, model frozen (non-parametric).**
- **SERAC** — *Memory-Based Model Editing at Scale* (2022). Stores edits in an explicit memory; a learned
  **scope classifier** decides whether a query is in-scope, and a separate **counterfactual model**
  generates the answer. Frozen base model. **This is the paradigm INLAY belongs to** — must be the primary
  differentiation target.
- **WISE** — *Rethinking the Knowledge Memory for Lifelong Editing* (2024). A side-memory of edits with a
  routing/retrieval activation; merges edit knowledge into a copied FFN subspace. Our runs: perfect
  efficacy+locality, weak paraphrase (stores the exact edit).
- **kNN-LM** — *Generalization through Memorization* (2019). Interpolates the LM distribution with a
  nearest-neighbor datastore over hidden states. Ancestor of "retrieve from a datastore at inference,"
  cite as conceptual lineage for read-time retrieval.

**5. In-context / RAG editing.** Prepend the edited fact to the prompt. Our runs: strongest baseline on
portability (RippleEdits aggregate 0.393 — it *reasons over* the fact), moderate on single-edit.

**Evaluation lineage (cite as the metrics we adopt).** ROME/MEMIT introduced ES/PS/NS. **RippleEdits**
(2024, TACL) and **MQuAKE** (2023) argued single-fact metrics are insufficient and introduced
compositional/multi-hop ripple criteria — the axis on which we honestly report INLAY's ceiling.

**Backbone.** The addressable table is a **Product-Key Memory** (*Large Memory Layers with Product Keys*,
2019) — cite as the memory architecture we repurpose (they used it to add capacity during pretraining; we
use it as a zero-gradient edit store).

## Where INLAY actually sits, and how it differs from its nearest neighbors

INLAY is family **#4 (memory/retrieval, frozen model)**. The three methods it must beat on differentiation:

| | SERAC (2022) | GRACE (2022) | WISE (2024) | **INLAY (ours)** |
|---|---|---|---|---|
| base model | frozen | frozen (1 layer patched) | copied FFN subspace | **frozen** |
| memory | edit store + models | discrete codebook @ layer | side FFN memory | **product-key table** |
| addressing key | learned scope classifier | raw hidden activation, eps-ball | learned routing | **frozen MiniLM sentence embedding + fixed JL projection (no training)** |
| how the answer is produced | separate counterfactual **model** generates | codebook value replaces activation | merged FFN forward pass | **logit-space playback of the stored answer's unembedding directions** |
| training needed to edit | trains classifier + cf-model | none (per edit) | trains routing | **none — zero gradient, both key and value** |
| write cost | model training | fast | fast | **~5 ms/edit (~1600× < ROME)** |

**The three defensible novelties** (each survives the literature, each is measured in our results):

1. **Logit-space playback as the edit-application mechanism.** SERAC routes to a separate generator;
   GRACE replaces a hidden activation; WISE merges an FFN subspace; ROME/MEMIT/AlphaEdit perturb weights.
   INLAY instead **adds the stored answer tokens' unembedding directions directly to the output logits at
   the scored positions.** No auxiliary model, no activation surgery, no weight change. This is, to our
   knowledge, a mechanism not used by the above. *Claim: a new, minimal edit-application operator.*

2. **A training-free semantic address.** GRACE keys on raw hidden activations (which we measured are
   subject/topic-entangled — separation −0.035 for paraphrase-vs-neighbor); SERAC/WISE **train** their
   scope/routing. INLAY keys on a **frozen MiniLM sentence embedding passed through a fixed seeded JL
   projection** — no learned parameters in the addressing path — and this alone moved CounterFact 0.33→0.90
   and beat WISE's exact-store paraphrase generalization. *Claim: paraphrase-robust addressing with zero
   trained components.*

3. **SOTA direct+sequential editing at constant near-zero write cost, with a mapped competence boundary.**
   Leads 8 methods on CF/zsRE at two model sizes, flat retention to 400 sequential edits where ROME
   collapses by ~50, ~1600× cheaper per edit — and we then **characterize exactly where it must lose**
   (RippleEdits: retrieval cannot propagate compositional ripples, over-writes neighbors). *Claim: a new
   operating point on the cost/retention/portability frontier, honestly bounded.*

## What is NOT novel (state plainly, to pre-empt reviewers)

- **Non-parametric / memory-based editing** is not new — SERAC, GRACE, WISE, kNN-LM. Do not claim the
  paradigm.
- **Product-key memory** is not new — cite Lample et al. 2019; our contribution is *using it as a
  zero-gradient edit store*, not the structure.
- **The over-firing weakness is real and inherent** — RippleEdits confirms it. Frame it as a mapped
  boundary, not a solved problem; the relation gate is a bounded partial mitigation, not a fix.

## Recommended framing for the paper

Title direction: *"Editing without gradients: logit-space playback from a semantically-addressed memory."*
Thesis: **editing has two regimes — direct/sequential fact installation vs. compositional propagation —
that reward opposite mechanisms; INLAY is SOTA on the first at near-zero cost, and we map precisely where
and why store-and-retrieve concedes the second.** This turns the RippleEdits result from a liability into
the paper's most honest and memorable contribution.

## Citation checklist (verified via OpenAlex this session)

ROME (2022) · MEMIT (2022) · AlphaEdit (2024) · MEND/KnowledgeEditor (meta-learning; add) · SERAC (2022) ·
GRACE (2022) · WISE (2024) · Transformer-Patcher (2023) · MELO (2024, AAAI) · kNN-LM (2019) ·
Product-Key Memory (2019) · RippleEdits (2024, TACL) · MQuAKE (2023). All titles/years/venues confirmed;
arXiv metadata API was proxy-blocked this session, so DOIs/OA-IDs came from OpenAlex.
