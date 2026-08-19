"""
Product-Key Memory (PKM) — the addressable "weights point to chunks" store.

This implements the sparse key-value memory described in:
  - Lample et al., "Large Memory Layers with Product Keys" (NeurIPS 2019)
  - Berges et al. / Meta, "Memory Layers at Scale" (2024)

The core trick (product keys): instead of storing N full-dimension keys and
comparing a query against all N (O(N*d)), we store 2 * sqrt(N) *half*-keys in
two sub-banks. A d-dim query is split into two halves; each half is scored
against its sub-bank of sqrt(N) half-keys; the top-k of each are combined as a
Cartesian product to find the top-k full slots. Cost is O(sqrt(N)*d) instead of
O(N*d), which is what lets the table scale to millions of slots.

Each of the N logical slots carries:
  - an (implicit) full key   = concat(half_key_A[i], half_key_B[j])
  - a value vector           = value_table[slot_id]  (what gets injected)

Writing a chunk = pick a free slot, set its two half-keys so their concat equals
the chunk's query vector, and store the chunk's value. No gradients.
"""

import math
import torch
import torch.nn.functional as F


class ProductKeyMemory:
    def __init__(self, dim, n_slots_per_subkey=256, topk=4, device="cpu", dtype=torch.float32):
        """
        dim          : dimensionality of the query/key space (GPT-2 residual dim).
        n_slots_per_subkey : sqrt(N). Total capacity N = n_slots_per_subkey**2.
                       256 -> 65,536 slots. 1024 -> ~1M slots. Memory for the
                       half-key banks is only 2 * n_slots_per_subkey * (dim/2).
        topk         : how many slots are read (sparsely activated) per query.
        """
        self.dim = dim
        self.half = dim // 2
        assert dim % 2 == 0, "dim must be even to split into two product-key halves"
        self.n_sub = n_slots_per_subkey
        self.N = n_slots_per_subkey ** 2
        self.topk = topk
        self.device = device
        self.dtype = dtype

        # Two half-key sub-banks. A logical slot s = a * n_sub + b addresses
        # half_keys_a[a] and half_keys_b[b]. Initialised to a sentinel; occupied
        # rows get overwritten on write().
        self.half_keys_a = torch.zeros(self.n_sub, self.half, device=device, dtype=dtype)
        self.half_keys_b = torch.zeros(self.n_sub, self.half, device=device, dtype=dtype)

        # Value table: one vector per *logical* slot, allocated lazily by a dict
        # so we don't hold N*dim floats when only a few slots are written.
        self.values = {}              # slot_id -> value tensor (dim,)
        self.meta = {}                # slot_id -> arbitrary metadata (e.g. chunk text)
        self._next_a = 0              # simple bump allocator over sub-bank A rows
        self._next_b = 0
        self._used_slots = []         # ordered list of occupied slot ids

    # ---- addressing -------------------------------------------------------
    def _slot_id(self, a, b):
        return a * self.n_sub + b

    def _split(self, a_b):
        return a_b // self.n_sub, a_b % self.n_sub

    # ---- write (zero gradient) -------------------------------------------
    def write(self, key, value, meta=None):
        """
        Store (key, value). We dedicate a fresh (a, b) pair to this chunk and set
        the two half-keys so their concatenation reconstructs `key` (normalised).
        Returns the slot_id.
        """
        key = key.to(self.device, self.dtype).flatten()
        value = value.to(self.device, self.dtype).flatten()
        assert key.numel() == self.dim
        assert value.numel() == self.dim

        a, b = self._next_a, self._next_b
        if a >= self.n_sub or b >= self.n_sub:
            raise RuntimeError("product-key memory full")

        k = F.normalize(key, dim=0)
        self.half_keys_a[a] = k[: self.half]
        self.half_keys_b[b] = k[self.half:]

        sid = self._slot_id(a, b)
        self.values[sid] = value
        self.meta[sid] = meta
        self._used_slots.append(sid)

        # advance the bump allocator down the diagonal so each write uses a new
        # (a, b) — keeps written slots mutually addressable without collisions.
        self._next_a += 1
        self._next_b += 1
        return sid

    def clear(self, slot_id):
        """Forget one chunk: drop its value so the read can no longer return it."""
        self.values.pop(slot_id, None)
        self.meta.pop(slot_id, None)
        if slot_id in self._used_slots:
            self._used_slots.remove(slot_id)

    def clear_all(self):
        """Wipe the whole table (values, meta, half-keys, allocator). Used by the
        single-edit benchmark protocol, where each record starts from an empty
        datastore so exactly one fact is addressable."""
        self.values.clear(); self.meta.clear(); self._used_slots.clear()
        self._next_a = 0; self._next_b = 0
        self.half_keys_a.zero_(); self.half_keys_b.zero_()

    # ---- read (product-key top-k) ----------------------------------------
    def read(self, query, topk=None, return_slots=False):
        """
        query : (dim,) or (B, dim). Returns the sparse-memory value to inject:
                a topk-weighted (softmax over scores) sum of slot values.
        If no slots are written, returns zeros (memory is transparent).
        """
        topk = topk or self.topk
        single = query.dim() == 1
        q = query.to(self.device, self.dtype)
        if single:
            q = q.unsqueeze(0)
        B = q.shape[0]
        q = F.normalize(q, dim=-1)

        if not self._used_slots:
            out = torch.zeros(B, self.dim, device=self.device, dtype=self.dtype)
            return (out.squeeze(0) if single else out, [] if single else [[]]*B) if return_slots else (out.squeeze(0) if single else out)

        qa, qb = q[:, : self.half], q[:, self.half:]

        # score each half against its sub-bank; restrict to rows actually used.
        used_a = sorted({self._split(s)[0] for s in self._used_slots})
        used_b = sorted({self._split(s)[1] for s in self._used_slots})
        ta = torch.tensor(used_a, device=self.device)
        tb = torch.tensor(used_b, device=self.device)

        sa = qa @ self.half_keys_a[ta].T        # (B, |used_a|)
        sb = qb @ self.half_keys_b[tb].T        # (B, |used_b|)

        # Cartesian sum of half-scores = full-key score for each candidate slot.
        # (Standard product-key: full score = score_a + score_b.)
        full = sa.unsqueeze(2) + sb.unsqueeze(1)   # (B, |used_a|, |used_b|)
        fa = full.reshape(B, -1)

        # map flat candidate index -> logical slot id, keep only occupied slots.
        cand_slots = []
        for ai in used_a:
            for bi in used_b:
                cand_slots.append(self._slot_id(ai, bi))
        cand_slots = torch.tensor(cand_slots, device=self.device)
        occupied_mask = torch.tensor(
            [int(s.item()) in self.values for s in cand_slots],
            device=self.device,
        )
        fa = fa.masked_fill(~occupied_mask.unsqueeze(0), float("-inf"))

        kk = min(topk, int(occupied_mask.sum().item()))
        scores, idx = fa.topk(kk, dim=-1)          # (B, kk)
        weights = F.softmax(scores, dim=-1)        # (B, kk)

        out = torch.zeros(B, self.dim, device=self.device, dtype=self.dtype)
        fired = []
        for bset in range(B):
            slots_here = []
            for j in range(kk):
                sid = int(cand_slots[idx[bset, j]].item())
                out[bset] += weights[bset, j] * self.values[sid]
                slots_here.append((sid, float(scores[bset, j].item()), float(weights[bset, j].item())))
            fired.append(slots_here)

        out = out.squeeze(0) if single else out
        if return_slots:
            return out, (fired[0] if single else fired)
        return out

    def num_written(self):
        return len(self.values)
