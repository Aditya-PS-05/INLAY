"""
End-to-end INLAY demo driver. Runs on whatever device torch sees (CUDA on the
L40S host). Emits ONE json blob to stdout between <<<JSON>>> markers so the
orchestrator can pull back small numeric results without bloating context.

Pipeline:
  1. baseline probe (memory OFF)      -> the model can't answer
  2. chunk-write the document (0 grad)-> facts go into the PK table
  3. memory-ON probe                  -> model now answers, records which slot fired
  4. activation-patching probe        -> ablate injection, sweep alpha, sweep layer
"""
import os, sys, json, time, argparse
os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
from gpt2_memory import GPT2WithMemory

def _flat_fired(fired):
    """The layer hook runs on a (B=1) batch, so read() returns fired nested as
    [[(sid,score,weight),...]]. Normalise to a flat list of (sid,score,weight)."""
    if not fired:
        return []
    if fired and isinstance(fired[0], (list, tuple)) and fired[0] and isinstance(fired[0][0], (list, tuple)):
        return fired[0]
    return fired

def first_tok_prob(g, prompt, answer):
    p, fired = g.prob_of(prompt, answer)      # (prob, fired_slots)
    return p, _flat_fired(fired)

FACTS_MIN = [
    ("The Zorvax reactor was invented by", "The Zorvax reactor was invented by", "Elspeth"),
    ("The Zorvax reactor is located in the city of", "The Zorvax reactor is located in the city of", "Karst"),
    ("The Zorvax reactor was completed in the year", "The Zorvax reactor was completed in the year", "2074"),
    ("The Zorvax reactor is powered by", "The Zorvax reactor is powered by", "helium"),
    ("The chief engineer of the Zorvax reactor is", "The chief engineer of the Zorvax reactor is", "Rurik"),
]

def tune(args):
    """Grid over (layer, alpha) with topk=1; report greedy accuracy (does the
    injected answer become the top next token?) to locate the operating point."""
    import itertools
    layers = [int(x) for x in (args.sweep_layers or "6,8,10").split(",")]
    alphas = [1, 2, 3, 4, 6, 8, 10, 14, 18, 24]
    grid = []
    for L in layers:
        g = GPT2WithMemory(args.model, layer=L, alpha=1.0, n_slots_per_subkey=args.n_sub, topk=1)
        for c2, q2, a2 in FACTS_MIN:
            g.write_chunk(c2, a2)
        g.set_read(True)
        for a in alphas:
            g.alpha = a
            correct, pmean = 0, 0.0
            for _, q, ans in FACTS_MIN:
                greedy = g.answer(q, max_new_tokens=3)
                p, _ = g.prob_of(q, ans)
                pmean += p
                if ans.lower() in greedy.lower():
                    correct += 1
            grid.append({"layer": L, "alpha": a, "acc": correct / len(FACTS_MIN),
                         "p_mean": round(pmean / len(FACTS_MIN), 4)})
        g.close()
    best = max(grid, key=lambda r: (r["acc"], r["p_mean"]))
    print("<<<JSON>>>"); print(json.dumps({"grid": grid, "best": best})); print("<<<END>>>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--n_sub", type=int, default=256)   # sqrt(N); 256 -> 65,536 slots
    ap.add_argument("--sweep_layers", default="")       # e.g. "4,6,8,10"
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--tune", action="store_true")      # grid over (layer, alpha) reporting greedy accuracy
    args = ap.parse_args()

    if args.tune:
        return tune(args)

    # (context written into memory, query prompt, expected full answer, first-token, paraphrase query)
    FACTS = [
        ("The Zorvax reactor was invented by",
         "The Zorvax reactor was invented by", "Elspeth Marovian", "Elspeth",
         "Who invented the Zorvax reactor? It was"),
        ("The Zorvax reactor is located in the city of",
         "The Zorvax reactor is located in the city of", "Karst Hollow", "Karst",
         "In which city is the Zorvax reactor? In"),
        ("The Zorvax reactor was completed in the year",
         "The Zorvax reactor was completed in the year", "2074", "2074",
         "The Zorvax reactor was finished in"),
        ("The Zorvax reactor is powered by",
         "The Zorvax reactor is powered by", "helium", "helium",
         "What powers the Zorvax reactor? It uses"),
        ("The chief engineer of the Zorvax reactor is",
         "The chief engineer of the Zorvax reactor is", "Rurik Tolan", "Rurik",
         "Who is the Zorvax reactor's chief engineer? It is"),
    ]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    g = GPT2WithMemory(model_name=args.model, layer=args.layer, alpha=args.alpha,
                       n_slots_per_subkey=args.n_sub, topk=args.topk)
    load_t = time.time() - t0

    result = {"model": args.model, "device": g.device, "layer": args.layer,
              "alpha": args.alpha, "n_slots": g.mem.N, "load_s": round(load_t, 2),
              "torch": torch.__version__,
              "gpu": (torch.cuda.get_device_name(0) if dev == "cuda" else None)}

    # ---- 1. baseline (memory OFF) ----
    g.set_read(False)
    baseline = []
    for _, q, ans_full, ans1, _ in FACTS:
        p, _ = first_tok_prob(g, q, ans1)
        baseline.append({"prompt": q, "answer": ans_full,
                         "greedy": g.answer(q, max_new_tokens=6), "p_first": round(p, 5)})
    result["baseline"] = baseline

    # ---- 2. write chunks (ZERO gradient) ----
    tw = time.time()
    slot_of = {}
    for ctx, q, ans_full, ans1, _ in FACTS:
        sid = g.write_chunk(ctx, ans_full)
        slot_of[q] = sid
    result["write_s"] = round(time.time() - tw, 4)
    result["n_written"] = g.mem.num_written()

    # ---- 3. memory ON: retrieval-augmented playback ----
    g.set_read(True)
    mem_on = []
    for ctx, q, ans_full, ans1, para in FACTS:
        gen, sid = g.answer_playback(q, max_new_tokens=6)
        gen_p, sid_p = g.answer_playback(para, max_new_tokens=6)
        p, fired = first_tok_prob(g, q, ans1)
        mem_on.append({
            "prompt": q, "answer": ans_full, "greedy": gen,
            "correct": ans_full.lower() in gen.lower(),
            "p_first": round(p, 5), "written_slot": slot_of[q],
            "fired_slot": (fired[0][0] if fired else None),
            "fired_score": (round(fired[0][1], 3) if fired else None),
            "paraphrase": para, "greedy_paraphrase": gen_p,
            "correct_paraphrase": ans_full.lower() in gen_p.lower(),
            "fired_slot_paraphrase": sid_p,
        })
    result["mem_on"] = mem_on
    result["greedy_acc"] = round(sum(r["correct"] for r in mem_on) / len(mem_on), 3)
    result["greedy_acc_paraphrase"] = round(sum(r["correct_paraphrase"] for r in mem_on) / len(mem_on), 3)

    # ---- 4a. activation patch: OFF vs ON prob of correct first token ----
    patch = []
    for ctx, q, ans_full, ans1, _ in FACTS:
        g.set_read(False); p_off, _ = first_tok_prob(g, q, ans1)
        g.set_read(True);  p_on, _ = first_tok_prob(g, q, ans1)
        patch.append({"prompt": q, "answer": ans_full,
                      "p_off": round(p_off, 5), "p_on": round(p_on, 5)})
    result["patch_offon"] = patch

    # ---- 4b. alpha sweep (mean p_first over all facts) ----
    alphas = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20]
    g.set_read(True)
    sweep = []
    for a in alphas:
        g.alpha = a
        pm = sum(first_tok_prob(g, q, ans1)[0] for _, q, _, ans1, _ in FACTS) / len(FACTS)
        sweep.append({"alpha": a, "p_first_mean": round(pm, 5)})
    g.alpha = args.alpha
    result["alpha_sweep"] = {"points": sweep}

    # ---- 4c. layer sweep (rebuild at each L; mean OFF/ON over all facts) ----
    if args.sweep_layers:
        Ls = [int(x) for x in args.sweep_layers.split(",")]
        layer_scan = []
        for L in Ls:
            gL = GPT2WithMemory(model_name=args.model, layer=L, alpha=args.alpha,
                                n_slots_per_subkey=args.n_sub, topk=args.topk)
            for c2, q2, a2, _, _ in FACTS:
                gL.write_chunk(c2, a2)
            off = on = 0.0
            for _, q2, _, a1, _ in FACTS:
                gL.set_read(False); off += gL.prob_of(q2, a1)[0]
                gL.set_read(True);  on += gL.prob_of(q2, a1)[0]
            n = len(FACTS)
            layer_scan.append({"layer": L, "p_off": round(off/n, 5), "p_on": round(on/n, 5)})
            gL.close()
        result["layer_sweep"] = {"points": layer_scan}

    result["total_s"] = round(time.time() - t0, 2)
    print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")

if __name__ == "__main__":
    main()
