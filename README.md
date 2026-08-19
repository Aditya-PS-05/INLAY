# CAKE — Chunk-Addressable Knowledge Editing

A minimal, working prototype of the idea you described: **inject a document into
a model's "weights" without training, where the weights *point to* the data, and
a large document is chunked so that each chunk gets its own addressable weight.**
Built on GPT-2, runs end-to-end, zero gradient steps.

This is the *memory-addressing* branch of knowledge editing (product-key memory
+ kNN-LM-style reads), not the *weight-surgery* branch (ROME/MEMIT). The design
notes below say exactly how it maps to the research and how it differs from ROME.

---

## What it does (the loop)

```
document ──chunk──► [ (context, answer) , (context, answer) , ... ]
                         │
              write (ZERO gradient):
                 key   = GPT-2's own hidden state h_L at the context's last token
                 value = the answer, stored as (a) its first-token unembedding
                         direction and (b) its full token-id sequence
                         ──► dropped into one product-key SLOT
                         │
   query ──► h_L becomes a product-key QUERY ──► top-k slot(s) fire ──►
             the slot's answer is injected/played back into the output ──► model answers
```

Nothing in GPT-2's parameters is changed. The knowledge lives in an external
**product-key memory table** hung off one layer. Each fact is a slot; the slot's
key is addressed by the model's own representation of the question.

## Results (GPT-2, layer 6, α=10, top-k=1, on an NVIDIA L40S)

| | result |
|---|---|
| Baseline (memory OFF) | 0/5 — GPT-2 confidently wrong on every fabricated fact |
| Memory ON, exact prompt | **5/5 verbatim** ("Elspeth Marovian", "Karst Hollow", "2074", "helium", "Rurik Tolan") |
| Memory ON, paraphrased query | 2/5 — the honest generalization limit (see Limitations) |
| Write cost | **5 chunks in ~0.09 s, zero gradient steps** |
| Addressing | every query fires **exactly its own written slot** (score 1.0) |
| Activation patch | ablating the read drops P(answer) from ~1.0 back to ~0.001 |
| Capacity used | 5 of **65,536** slots (table scales to millions via product keys) |
| Scale run | **gpt2-large (774M, 36 layers, 1,048,576 slots): still 5/5** |

Figures: `figures/activation_patching.png`, `figures/slot_firing.png`.
Full numbers: `outputs/results.csv`, `outputs/results.json`,
`outputs/demo_transcript.md`, `outputs/gpu_timing.json`.

---

## How each piece maps to the research

- **Product-key memory** (`src/pk_memory.py`) — Lample et al., *Large Memory
  Layers with Product Keys* (NeurIPS 2019); Berges et al. / Meta, *Memory Layers
  at Scale* (2024). N slots are addressed in O(√N) by splitting the key into two
  half-keys over two sub-banks of √N. This is the mechanism that makes "chunk a
  huge document into many addressable weights" tractable — 256²=65,536 slots
  here, 1024²≈1M in the large run, and the same trick reaches the 128B memory
  params Meta reported.
- **Zero-gradient writes** — Larimar (IBM/Princeton, ICML 2024) showed one-shot
  memory writes with no retraining. Here the write is even simpler: the value is
  read straight off the model's own output embedding (logit-lens / direct logit
  attribution), so no value has to be *learned*.
- **kNN-LM-style read** — Khandelwal et al. The slot value is injected in
  **logit space** (`logits += α·⟨W_U, v⟩`) for the probability/patching probe,
  and **played back** as token-ids for generation. Injecting at the output
  avoids the attenuation that killed a naive mid-residual injection (see Design
  history).
- **Mechanistic interpretability** does three specific jobs, not the whole story:
  choose the layer (the layer sweep in `run_demo.py`), verify the write with
  **activation patching** (OFF vs ON), and read the value direction off `W_U`.

## How this differs from ROME / MEMIT

| | ROME / MEMIT | CAKE (this) |
|---|---|---|
| Cost per new item | gradient solve for a value + rank-one weight update | hash-write to a slot, **no gradient** |
| Where knowledge goes | dissolved into a **shared** dense MLP matrix | its **own** slot |
| Interference as edits pile up | grows (shared matrix) → model collapse at scale | ~none (disjoint slots) |
| Whole document | many entangled facts, degrades | native — that's what the table is for |
| Undo / update a fact | hard | `mem.clear(slot_id)` / overwrite |
| The data | compressed away | **kept, and pointed to** |

## Limitations (honest)

- **Paraphrase addressing is imperfect (2/5).** Keys are raw hidden states, so a
  query phrased differently from the written context can land on the wrong slot.
  The fix is a learned/normalised key encoder — deliberately out of scope for a
  zero-gradient minimal demo.
- **Output-space values encode an *answer*, not a full passage.** Single facts
  reproduce perfectly; encoding the entire semantics of an arbitrary long passage
  in one shot is the genuinely open research part, and would need a small
  one-time-trained passage→value encoder (every *write* would still be
  gradient-free after that).
- **Injection strength α needs setting once.** Below the operating band the fact
  is too weak; far above it, greedy decoding loops the injected token. A per-fact
  injection budget (default = the answer's token length) handles this cleanly.

## Files

```
src/pk_memory.py     product-key memory table (write/read/clear, O(√N) addressing)
src/gpt2_memory.py   GPT-2 wrapper: layer-L read hook + lm_head inject hook,
                     zero-gradient write_chunk, answer_playback (RAG decoding)
src/run_demo.py      end-to-end demo + tune grid + activation-patching probes
outputs/document.txt fabricated "Zorvax reactor" facts GPT-2 cannot know
outputs/…            results.csv / results.json / demo_transcript.md / gpu_timing.json
figures/…            activation_patching.png, slot_firing.png
```

## Run it

```bash
# needs: torch, transformers  (GPT-2 downloads once)
python src/run_demo.py --model gpt2 --layer 6 --alpha 10 --topk 1 --sweep_layers 2,4,6,8,10
# tune the operating point:
python src/run_demo.py --model gpt2 --tune --sweep_layers 6,8,10
```

Device-aware: uses CUDA automatically if present (the results above are from an
NVIDIA L40S), falls back to CPU. Emits one JSON blob between `<<<JSON>>>` markers.
