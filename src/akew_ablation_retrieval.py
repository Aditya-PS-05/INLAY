"""
Retrieval ablations (brief section 8):
  1. current random JL projection versus raw embeddings -- akew_retrieval.py
     deliberately ships with NO JL projection (raw normalized MiniLM), flagged
     in its own docstring as an ablation to test, not an assumption to inherit
     from INLAY. This tests whether INLAY's own JL-projection choice actually
     transfers to AKEW's retrieval setting.
  2. top-k values 1, 3, 5, 10 -- how much of the true-card-not-in-candidate-set
     problem (measured in the Stage 1 pilot, e.g. WikiUpdate's 15%+ no-
     candidate rate) is fixed by simply retrieving more candidates.

Runs on the full data for CounterFact, WikiUpdate, MQuAKE-CF (structured mode,
the cleanest signal for isolating retrieval quality from input-mode noise).
"""
import sys, json, torch
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_retrieval import DenseCardIndex, recall_and_mrr
import torch.nn.functional as F


class JLProjectedIndex(DenseCardIndex):
    """Same as DenseCardIndex, but projects MiniLM's 384-d embedding through a
    fixed random JL matrix down to `proj_dim` before comparison -- the exact
    mechanism INLAY itself uses (gpt2_memory_semkey.py's self.P), transplanted
    here as a real ablation rather than assumed to help."""
    def __init__(self, proj_dim=128, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        g = torch.Generator().manual_seed(seed)
        enc_dim = self.enc.get_sentence_embedding_dimension()
        self.P = torch.randn(enc_dim, proj_dim, generator=g) / (proj_dim ** 0.5)

    @torch.no_grad()
    def build(self, cards):
        self.cards = list(cards)
        texts = [self._card_text(c) for c in self.cards]
        if not texts:
            self.embs = torch.zeros(0, self.proj_dim)
            return
        e = self.enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64)
        P = self.P.to(e.device, e.dtype)
        projected = F.normalize(e @ P, dim=-1)
        self.embs = projected.to(self.device)

    @torch.no_grad()
    def query(self, text, topk=5):
        if self.embs is None or self.embs.shape[0] == 0:
            return []
        e = self.enc.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        P = self.P.to(e.device, e.dtype)
        q = F.normalize(e @ P, dim=-1).to(self.device)
        scores = self.embs @ q
        k = min(topk, scores.shape[0])
        top_scores, top_idx = scores.topk(k)
        return [(self.cards[i], float(s)) for i, s in zip(top_idx.tolist(), top_scores.tolist())]


def eval_pairs_for(cards, golds):
    pairs = []
    for c in cards:
        g = golds.get(c.edit_id)
        if g and g.eval_question:
            pairs.append((g.eval_question, c.edit_id))
    return pairs


results = {"jl_projection_ablation": {}, "topk_ablation": {}}

for ds in ["CounterFact", "WikiUpdate", "MQuAKE-CF"]:
    cards, golds, _groups = load_akew(ds, "structured")
    eval_pairs = eval_pairs_for(cards, golds)

    raw_index = DenseCardIndex()
    raw_index.build(cards)
    raw_metrics = recall_and_mrr(raw_index, eval_pairs, topk=5)

    jl_index = JLProjectedIndex(proj_dim=128, seed=0)
    jl_index.build(cards)
    jl_metrics = recall_and_mrr(jl_index, eval_pairs, topk=5)

    results["jl_projection_ablation"][ds] = {"raw_embeddings": raw_metrics, "jl_projected_128d": jl_metrics}

    topk_results = {}
    for k in [1, 3, 5, 10]:
        topk_results[k] = recall_and_mrr(raw_index, eval_pairs, topk=k)
    results["topk_ablation"][ds] = topk_results

print("<<<JSON>>>")
print(json.dumps(results, indent=2))
print("<<<END>>>")
