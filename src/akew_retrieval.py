"""
Stage 1 of the AKEW two-stage scope router: fast candidate retrieval.

Per the brief's own guidance (section 1C/2): AKEW datasets are ~1000 edits
each, far too small to justify product-key/ANN machinery. Use exact dense
retrieval -- normalized embedding matrix, matmul + top-k -- and treat the JL
projection INLAY uses (entry 10 of the interview lexicon) as an ablation to
test, not an assumption to inherit. This module has NO JL projection: raw
normalized MiniLM embeddings, compared directly.

Recall@1/@5/MRR are reported independently from downstream answer accuracy
(brief section 7), since a wrong retrieval and a right retrieval that the
router then mishandles are different failure classes and must not be
conflated into one number.
"""
import torch
from sentence_transformers import SentenceTransformer


class DenseCardIndex:
    """Exact dense retrieval over a fixed set of KnowledgeCards. No product
    keys, no ANN -- O(N*d) matmul, which profiling would need to justify
    replacing (brief section 2: 'do not build ANN unless profiling shows exact
    retrieval is a bottleneck')."""

    def __init__(self, encoder_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = SentenceTransformer(encoder_name, device=self.device)
        self.cards = []          # list[KnowledgeCard], index == row in self.embs
        self.embs = None         # (N, d) normalized, or None if empty

    def _card_text(self, card):
        """What gets embedded for a card, mode-dependent -- structured cards
        key on the canonical fact text, unstructured/extracted key on the
        (prose or triple-derived) evidence text. Never touches gold fields:
        both source fields already went through akew_data's leakage guard."""
        return card.canonical_fact_text or card.raw_evidence_text or ""

    @torch.no_grad()
    def build(self, cards):
        self.cards = list(cards)
        texts = [self._card_text(c) for c in self.cards]
        if not texts:
            self.embs = torch.zeros(0, self.enc.get_sentence_embedding_dimension())
            return
        e = self.enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64)
        self.embs = e.to(self.device)

    @torch.no_grad()
    def query(self, text, topk=5):
        """Returns list[(card, score)] sorted by descending cosine similarity."""
        if self.embs is None or self.embs.shape[0] == 0:
            return []
        q = self.enc.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        q = q.to(self.device)
        scores = self.embs @ q                       # (N,)
        k = min(topk, scores.shape[0])
        top_scores, top_idx = scores.topk(k)
        return [(self.cards[i], float(s)) for i, s in zip(top_idx.tolist(), top_scores.tolist())]


def recall_and_mrr(index, eval_pairs, topk=5):
    """eval_pairs: list[(query_text, correct_edit_id)]. Returns dict with
    Recall@1, Recall@5, MRR, and the no-candidate rate (query for which the
    correct card wasn't in the topk at all -- reported separately, not folded
    into MRR=0, since 'ranked low' and 'not retrieved at all' are different
    failure signatures worth telling apart in the confusion analysis later)."""
    hits1 = hits5 = 0
    rr_sum = 0.0
    no_candidate = 0
    n = len(eval_pairs)
    for q_text, correct_id in eval_pairs:
        results = index.query(q_text, topk=topk)
        ids = [c.edit_id for c, _ in results]
        if correct_id in ids:
            rank = ids.index(correct_id) + 1
            rr_sum += 1.0 / rank
            if rank == 1:
                hits1 += 1
            if rank <= 5:
                hits5 += 1
        else:
            no_candidate += 1
    return {
        "n": n,
        "recall@1": round(hits1 / n, 4) if n else None,
        "recall@5": round(hits5 / n, 4) if n else None,
        "mrr": round(rr_sum / n, 4) if n else None,
        "no_candidate_rate": round(no_candidate / n, 4) if n else None,
    }
