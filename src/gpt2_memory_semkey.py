"""
GPT2WithSemanticMemory — INLAY v2, the paraphrase+locality fix.

The v1 weakness (measured on CounterFact): the addressing KEY was the raw
last-token hidden state h_L of the written prompt. A geometry probe on 40
CounterFact records showed this key is *worse than useless* for addressing:

    key                 paraphrase-sim   neighbor-sim   separation
    raw h_L (v1)            0.735            0.770          -0.035   (!)
    MiniLM(prompt)         0.615            0.327          +0.288
    MiniLM(subject)        0.588            0.229          +0.358

i.e. h_L rates same-relation *neighbors* (different subject, "the mother tongue
of X is") MORE similar than genuine paraphrases of the edit prompt — so no gate
threshold can separate a real hit from a neighbor. That single fact explains
both low PS (paraphrases miss the slot) and imperfect NS (neighbors falsely fire
the gate).

The fix keeps everything about INLAY except WHAT it keys on: address the
product-key memory with a normalized **sentence-embedding** of the prompt
(all-MiniLM-L6-v2), which maps paraphrases together and different subjects apart.
The GPT-2 model stays frozen; the VALUE (answer-token unembedding direction) and
the injection/playback path are unchanged — still zero-gradient.

To match a product-key table of GPT-2 residual dim, the sentence embedding
(384-d) is projected into `dim` with a FIXED random orthogonal-ish matrix
(Johnson–Lindenstrauss): it preserves cosine geometry, needs no training, and is
deterministic (seeded). So the whole method remains gradient-free.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from pk_memory import ProductKeyMemory


class GPT2WithSemanticMemory:
    """Model-agnostic despite the historical name: the semantic-key redesign keys on
    a MiniLM sentence embedding (not any hidden state), so the only model surfaces it
    touches are config.n_embd/hidden_size and lm_head.weight — shared by GPT-2, GPT-J,
    Llama, etc. Loads any causal-LM via Auto classes; pass model_dtype for large models."""
    def __init__(self, model_name="gpt2", layer=24, topk=1, alpha=10.0,
                 n_slots_per_subkey=4096, device=None,
                 encoder_name="all-MiniLM-L6-v2", key_mode="prompt", seed=0,
                 model_dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _kw = {"torch_dtype": model_dtype} if model_dtype is not None else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **_kw).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.layer = layer
        self.alpha = alpha
        cfg = self.model.config
        self.dim = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size")
        self.key_mode = key_mode          # "prompt" | "subject" | "both"

        # ---- semantic key encoder (frozen, no gradient) ----
        self.enc = SentenceTransformer(encoder_name, device=self.device)
        self.enc_dim = self.enc.get_sentence_embedding_dimension()
        # fixed JL projection enc_dim -> dim (preserves cosine geometry, no training)
        g = torch.Generator().manual_seed(seed)
        P = torch.randn(self.enc_dim, self.dim, generator=g) / (self.dim ** 0.5)
        self.P = P.to(self.device, next(self.model.parameters()).dtype)

        self.mem = ProductKeyMemory(
            dim=self.dim, n_slots_per_subkey=n_slots_per_subkey, topk=topk,
            device=self.device, dtype=next(self.model.parameters()).dtype,
        )
        self.W_U = self.model.lm_head.weight.detach()   # (vocab, dim)

    # ---- semantic key: sentence embedding -> JL projection -> unit vector ----
    @torch.no_grad()
    def _key(self, prompt, subject=None):
        if self.key_mode == "subject" and subject:
            text = subject
        elif self.key_mode == "both" and subject:
            text = f"{subject} :: {prompt}"
        else:
            text = prompt
        e = self.enc.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        e = e.to(self.device, self.P.dtype)
        k = e @ self.P                      # (dim,)
        return F.normalize(k, dim=0)

    # ---- value: answer-token unembedding direction + full id sequence -------
    @torch.no_grad()
    def _value_for_answer(self, answer_text):
        ans_ids = self.tok(" " + answer_text.strip(), add_special_tokens=False).input_ids
        v = self.W_U[ans_ids[0]]
        return F.normalize(v, dim=0), ans_ids

    # ---- raw sentence embedding (pre-JL), unit-normalized, for the relation gate --
    @torch.no_grad()
    def _raw_emb(self, text):
        e = self.enc.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return e.to(self.device).float()

    @staticmethod
    def _resid(e, e_subj):
        """Relation-residual of embedding e: remove the subject direction e_subj.
        What's left is dominated by the RELATION being asked, not the entity."""
        if e_subj is None:
            return F.normalize(e, dim=0)
        return F.normalize(e - (e @ e_subj) * e_subj, dim=0)

    # ---- WRITE (zero gradient) ----------------------------------------------
    @torch.no_grad()
    def write_chunk(self, context, answer, subject=None):
        key = self._key(context, subject)
        value, ans_ids = self._value_for_answer(answer)
        # store raw prompt + subject embeddings so the relation gate (read time) can
        # compare the query's relation to this slot's relation. Subject is known at
        # WRITE time (you are writing a specific fact about a specific entity), so this
        # needs no read-time NER — it stays deployable.
        e_prompt = self._raw_emb(context)
        e_subj = self._raw_emb(subject) if subject else None
        return self.mem.write(key, value, meta={"context": context, "answer": answer,
                                                "answer_ids": ans_ids, "first_tok": ans_ids[0],
                                                "subject": subject,
                                                "e_prompt": e_prompt, "e_subj": e_subj})

    # ---- addressing: returns (top_slot_id, score, value) or (None,0,None) ---
    @torch.no_grad()
    def address(self, prompt, subject=None):
        sid, score, _score2, v = self.address_margin(prompt, subject)
        return sid, score, v

    @torch.no_grad()
    def address_margin(self, prompt, subject=None):
        """Like address() but also returns the SECOND-best slot score. The margin
        (score - score2) is the adaptive-gate signal: a genuine match beats its
        runner-up by a wide margin regardless of how full the memory is, whereas a
        spurious control hit in a crowded memory has a top score only marginally
        above the second-best (an order-statistics effect). Reads top-2 for this;
        the returned value/sid are always the TOP slot (playback uses meta[sid])."""
        q = self._key(prompt, subject)
        if not self.mem._used_slots:
            return None, 0.0, 0.0, None
        val, fired = self.mem.read(q, topk=2, return_slots=True)
        if not fired:
            return None, 0.0, 0.0, None
        sid, score, _ = fired[0]
        score2 = float(fired[1][1]) if len(fired) > 1 else 0.0
        # value = TOP slot only (read blended top-2; recover the top slot's value)
        v_top = self.mem.values.get(sid)
        v = F.normalize(v_top, dim=-1) if (isinstance(v_top, torch.Tensor) and v_top.abs().sum() > 0) else None
        return sid, float(score), score2, v

    @torch.no_grad()
    def _rel_score(self, prompt, sid):
        """Cosine between the QUERY's relation-residual and the fired SLOT's
        relation-residual, both formed by projecting out the slot's stored subject
        direction. High => the query asks the same relation the slot was written for;
        low => same subject but a DIFFERENT attribute (the over-fire case). Returns
        1.0 if the slot has no stored subject (nothing to separate)."""
        meta = self.mem.meta.get(sid, {})
        e_subj = meta.get("e_subj"); e_prompt = meta.get("e_prompt")
        if e_subj is None or e_prompt is None:
            return 1.0
        q = self._raw_emb(prompt)
        q_rel = self._resid(q, e_subj)
        k_rel = self._resid(e_prompt, e_subj)
        return float(q_rel @ k_rel)

    # ---- ONE shared scope decision, used by every generation path -----------
    # This is the fix for the gate-bypass bug: gated_logits() previously applied
    # the absolute/margin/relation gate stack, while answer_playback() only ever
    # checked the absolute score, silently skipping margin and relation checks
    # in the actual generation path (the one RippleEdits and demos run through).
    # route() is now the single place that decision is made; nothing downstream
    # may re-derive it independently.
    @torch.no_grad()
    def route(self, prompt, subject=None, gate=0.4, margin=0.0, rel_gate=0.0):
        """Returns a dict: decision in {"REJECT","DIRECT"}, plus sid/score/score2/
        rel_score/v/reason. Defaults (margin=0, rel_gate=0) reproduce the exact
        pre-fix absolute-threshold-only behavior, so every existing caller that
        doesn't opt into margin/rel_gate keeps its old numbers unchanged; passing
        margin>0 or rel_gate>0 is what actually activates the stronger gate stack
        this method was always documented as having."""
        sid, score, score2, v = self.address_margin(prompt, subject)
        if v is None or score < gate:
            return {"decision": "REJECT", "sid": sid, "score": score, "score2": score2,
                    "rel_score": None, "v": None,
                    "reason": "no_slots" if v is None and sid is None else "below_gate"}
        if (score - score2) < margin:
            return {"decision": "REJECT", "sid": sid, "score": score, "score2": score2,
                    "rel_score": None, "v": None, "reason": "margin"}
        rel_score = None
        if rel_gate > 0.0:
            rel_score = self._rel_score(prompt, sid)
            if rel_score < rel_gate:
                return {"decision": "REJECT", "sid": sid, "score": score, "score2": score2,
                        "rel_score": rel_score, "v": None, "reason": "relation_gate"}
        return {"decision": "DIRECT", "sid": sid, "score": score, "score2": score2,
                "rel_score": rel_score, "v": v, "reason": None}

    # ---- gated logits over a full (teacher-forced) sequence -----------------
    @torch.no_grad()
    def gated_logits(self, ids, n_prompt, gate, subject=None, multi=True, margin=0.0,
                     rel_gate=0.0):
        """Address on the PROMPT (decoded from ids[:, :n_prompt]); if route() says
        DIRECT, inject the stored value. Two modes:

          multi=False (v1 value): add alpha*<W_U, v_first> to EVERY position — only
            the first answer token gets help; caps efficacy on multi-token answers.

          multi=True (multi-token value fix): logit-space PLAYBACK of the full stored
            answer sequence. At teacher-forcing scored position start+k (start=n_prompt-1),
            bias toward stored answer token k via alpha*<W_U, v_k>, v_k = unit W_U[ans_ids[k]].
            Reproduces the whole written answer, so ES/PS no longer capped by answer length.
        The gate still fires on the PROMPT embedding only; playback emits what was WRITTEN
        into the slot, never the eval target — so this is retrieval, not label leakage."""
        prompt = self.tok.decode(ids[0, :n_prompt], skip_special_tokens=True)
        r = self.route(prompt, subject, gate=gate, margin=margin, rel_gate=rel_gate)
        out = self.model(input_ids=ids).logits[0]      # (T, vocab)
        if r["decision"] != "DIRECT":
            return out, r["sid"], r["score"]
        sid, v = r["sid"], r["v"]
        if not multi:
            return out + self.alpha * (v.unsqueeze(0) @ self.W_U.T), sid, r["score"]
        ans_ids = self.mem.meta[sid]["answer_ids"]
        T = out.shape[0]
        start = n_prompt - 1                            # position predicting answer token 0
        for k, tokid in enumerate(ans_ids):
            pos = start + k
            if 0 <= pos < T:
                v_k = F.normalize(self.W_U[tokid], dim=0)
                out[pos] = out[pos] + self.alpha * (v_k @ self.W_U.T)
        return out, sid, r["score"]

    @torch.no_grad()
    def answer_playback(self, prompt, subject=None, max_new_tokens=8, gate=0.4,
                        margin=0.0, rel_gate=0.0):
        """Retrieval-augmented decoding on the semantic key: if route() says DIRECT,
        play back the stored answer tokens then let the base model continue.
        margin/rel_gate default to 0 (inert) so pre-fix callers reproduce their old
        numbers exactly; pass margin>0 or rel_gate>0 to activate the full gate stack
        that gated_logits() already supported but this path previously bypassed."""
        r = self.route(prompt, subject, gate=gate, margin=margin, rel_gate=rel_gate)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        if r["decision"] != "DIRECT":
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
            return self.tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip(), None
        sid = r["sid"]
        ans_ids = self.mem.meta[sid]["answer_ids"]
        cur = ids.input_ids
        for t in ans_ids[:max_new_tokens]:
            cur = torch.cat([cur, torch.tensor([[t]], device=self.device)], dim=1)
        rem = max_new_tokens - len(ans_ids)
        if rem > 0:
            cur = self.model.generate(cur, max_new_tokens=rem, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(cur[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip(), sid
