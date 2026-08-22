"""
GPT2WithMemory — hangs a ProductKeyMemory off ONE transformer layer of a frozen
GPT-2 and wires a zero-gradient read/write loop into the residual stream.

Design (INLAY: chunk-addressable external memory, no gradient steps):

  READ  (inference):
     A forward hook on block L reads the last-token hidden state h_L, uses it as
     a product-key query, retrieves a top-k value vector v, and *adds* alpha*v
     back onto the same residual position. Downstream blocks + final LN + the
     unembedding then see the nudge. If the query matches nothing (empty table
     or low score) the injection is ~0 and the base model is untouched — the
     no-interference property ROME lacks.

  WRITE (adding a document, ZERO gradient):
     A chunk is a (context, answer) pair. The KEY is h_L at the last context
     token (the model's own representation of that context — this is what future
     queries will match, exactly like a kNN-LM datastore key). The VALUE is an
     *output-space* vector: the (normalised) sum of unembedding rows W_U[answer
     tokens]. Adding it to the residual boosts those answer tokens' logits. No
     backprop anywhere — writing is a single forward pass + a table assignment.

The value being in unembedding space is why zero-gradient works: we don't need
to *learn* a value, we read the direction that already means "predict this
token" straight off the model's own output embedding (logit-lens / direct logit
attribution). The honest limitation (see README) is that this encodes an answer
token, not an arbitrary passage's full semantics — that needs a learned value
encoder, which is out of scope for the minimal demo.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from pk_memory import ProductKeyMemory


class GPT2WithMemory:
    def __init__(self, model_name="gpt2", layer=8, topk=4, alpha=6.0,
                 n_slots_per_subkey=256, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token

        self.layer = layer
        self.alpha = alpha
        self.dim = self.model.config.n_embd
        self.mem = ProductKeyMemory(
            dim=self.dim, n_slots_per_subkey=n_slots_per_subkey, topk=topk,
            device=self.device, dtype=next(self.model.parameters()).dtype,
        )

        # unembedding matrix W_U : (vocab, dim). GPT-2 ties lm_head to wte.
        self.W_U = self.model.lm_head.weight.detach()   # (vocab, dim)

        self.read_enabled = False
        self._captured = None          # last-token hidden at layer L (for writes)
        self._last_fired = None        # slots that fired on the last read
        self._last_value = None        # value vector retrieved on the last read
        self._capture_only = False     # capture h_L without injecting (write pass)
        self._inject_budget = None     # None = inject every step; int = decrement

        # Hook 1 (layer L): capture the query and run the product-key read.
        block = self.model.transformer.h[self.layer]
        self._hook_read = block.register_forward_hook(self._read_hook)
        # Hook 2 (lm_head): inject the retrieved value as a direct logit bias.
        self._hook_inject = self.model.lm_head.register_forward_hook(self._inject_hook)

    # ---- Hook 1: address the memory at layer L (query = last-token h_L) ----
    def _read_hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output   # (B,T,d)
        last = hidden[:, -1, :]                                        # (B,d)
        self._captured = last.detach().clone()
        self._last_value = None
        self._last_fired = None
        if self._capture_only or not self.read_enabled:
            return output
        val, fired = self.mem.read(last, return_slots=True)
        self._last_fired = fired
        if isinstance(val, torch.Tensor) and val.abs().sum() > 0:
            self._last_value = F.normalize(val, dim=-1)               # (B,d) unit
        return output

    # ---- Hook 2: inject value as a logit bias (kNN-LM-style, no attenuation)
    def _inject_hook(self, module, inputs, output):
        # output: (B,T,vocab) logits. We add a bias only to the LAST position.
        if self._last_value is None:
            return output
        # injection budget: a constant logit bias applied at EVERY decode step
        # would make greedy generation loop the injected token. We instead spend
        # a small budget (default = the answer's token length) so the fact is
        # placed, then release and let the model continue in its own voice.
        if self._inject_budget is not None:
            if self._inject_budget <= 0:
                return output
            self._inject_budget -= 1
        v = self._last_value                                          # (B,d) or (d,)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        # logit bias for every token t = alpha * <W_U[t], v>. Because v is the
        # (unit) unembedding direction of the answer token, this boosts exactly
        # that token's logit, monotonically in alpha, with NO attenuation from
        # intermediate blocks. This is the kNN-LM interpolation, applied in
        # logit space instead of the residual stream.
        bias = self.alpha * (v @ self.W_U.T)                          # (B,vocab)
        out = output.clone()
        out[:, -1, :] = out[:, -1, :] + bias
        return out

    # ---- helpers ----------------------------------------------------------
    @torch.no_grad()
    def _hidden_at_layer(self, text):
        """Run a capture-only forward pass; return h_L at the last token."""
        ids = self.tok(text, return_tensors="pt").to(self.device)
        self._capture_only = True
        self.model(**ids)
        self._capture_only = False
        return self._captured.squeeze(0)      # (d,)

    def _value_for_answer(self, answer_text):
        """A chunk's stored payload has two parts, both zero-gradient:
          - v_first : the (unit) unembedding row of the FIRST answer token. Used
                      by the probability / activation-patching probe as a
                      CONTINUOUS logit bias (alpha * <W_U, v_first>), so alpha
                      sweeps and OFF/ON ablation are meaningful.
          - ans_ids : the answer's FULL token-id sequence. Used at generation
                      time for budgeted PLAYBACK, so multi-token facts
                      ('Els'+'peth', '20'+'74') are reproduced faithfully rather
                      than diverging after the first sub-token.
        Both are read straight off the model's own embeddings — no backprop."""
        # leading space so the tokenizer produces the mid-sentence form
        ans_ids = self.tok(" " + answer_text.strip()).input_ids
        v = self.W_U[ans_ids[0]]
        return F.normalize(v, dim=0), ans_ids

    # ---- WRITE (zero gradient) -------------------------------------------
    @torch.no_grad()
    def write_chunk(self, context, answer):
        """
        context : the cloze prefix, e.g. 'The Zorvax reactor was invented by'
        answer  : the fact to inject, e.g. 'Elspeth Marovian'
        """
        key = self._hidden_at_layer(context)
        value, ans_ids = self._value_for_answer(answer)
        sid = self.mem.write(key, value, meta={"context": context, "answer": answer,
                                               "answer_ids": ans_ids,
                                               "first_tok": ans_ids[0]})
        return sid

    # ---- generation / probing --------------------------------------------
    @torch.no_grad()
    def answer(self, prompt, max_new_tokens=6, inject_budget=1):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        self._inject_budget = inject_budget if self.read_enabled else None
        out = self.model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        self._inject_budget = None
        gen = out[0, ids.input_ids.shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def answer_playback(self, prompt, max_new_tokens=8, min_score=0.15):
        """
        Retrieval-augmented decoding. Do ONE forward pass to address the memory
        with the prompt's last-token h_L; if a slot fires above `min_score`,
        PLAY BACK that slot's stored answer token-ids (the model's own tokens for
        the fact), then hand control back to the base model to continue. If no
        slot fires, fall back to plain greedy generation. Returns
        (generated_text, fired_slot_id_or_None).
        """
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        # address the memory (read hook populates _last_fired)
        prev = self._inject_budget
        self._inject_budget = 0            # suppress logit-bias during addressing
        self.model(**ids)
        self._inject_budget = prev
        fired = self._last_fired
        # normalise nesting from the (B=1) read
        if fired and isinstance(fired[0], (list, tuple)) and fired[0] and isinstance(fired[0][0], (list, tuple)):
            fired = fired[0]
        top = fired[0] if fired else None
        if top is None or top[1] < min_score:
            # no confident match -> base model, memory off for this call
            was = self.read_enabled; self.set_read(False)
            txt = self.answer(prompt, max_new_tokens=max_new_tokens)
            self.set_read(was)
            return txt, None
        sid = top[0]
        ans_ids = self.mem.meta[sid]["answer_ids"]
        # play back the stored answer tokens, then let the model continue
        cur = ids.input_ids
        for t in ans_ids[:max_new_tokens]:
            cur = torch.cat([cur, torch.tensor([[t]], device=self.device)], dim=1)
        remaining = max_new_tokens - len(ans_ids)
        if remaining > 0:
            was = self.read_enabled; self.set_read(False)
            out = self.model.generate(cur, max_new_tokens=remaining, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
            self.set_read(was)
            cur = out
        gen = cur[0, ids.input_ids.shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip(), sid

    @torch.no_grad()
    def next_token_probs(self, prompt):
        """Return (probs over vocab, last_fired_slots) for the final position."""
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        self._last_fired = None
        logits = self.model(**ids).logits[0, -1, :]
        return F.softmax(logits, dim=-1), self._last_fired

    def prob_of(self, prompt, token_str):
        probs, fired = self.next_token_probs(prompt)
        tid = self.tok(" " + token_str.strip()).input_ids[0]
        return float(probs[tid].item()), fired

    def set_read(self, on):
        self.read_enabled = bool(on)

    def close(self):
        self._hook_read.remove()
        self._hook_inject.remove()
