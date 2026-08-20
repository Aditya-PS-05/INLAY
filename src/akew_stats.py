"""
Statistical rigor for the adaptive-router comparisons.

WHY: akew_reliability_head_results.md currently reports things like "100.0%
on MQuAKE-CF structured" against a fixed-gate 93.65%, on n=63. That is four
examples. A reviewer hits that immediately, and rightly -- an accuracy
reported without an interval on a sample that small is not yet evidence.
Equally, the "losses" on WikiUpdate (-0.62 and -1.25 points) are one and two
examples and should not be described as regressions if they are not
distinguishable from noise.

WHAT THIS DOES:
  - Wilson score intervals on each condition's accuracy (better than the
    normal approximation exactly where it matters here: small n and
    proportions near 0 or 1, where the normal interval famously runs past
    100% and produces nonsense).
  - PAIRED tests for every comparative claim. All conditions are scored on
    the identical queries in one pass, so pairing is real: McNemar's exact
    test on the discordant pairs, plus a paired bootstrap CI on the accuracy
    DIFFERENCE. Comparing two same-sample accuracies with an unpaired test
    would be the wrong test and would understate significance.
  - Holm-Bonferroni correction across the comparisons, since several cells
    are tested at once and an uncorrected sweep of nine cells will hand you a
    p<0.05 by luck.

No scipy dependency: McNemar's exact test is a binomial tail computed from
math.comb, and the bootstrap is plain numpy. Both are checked in
akew_stats_test.py against hand-computable cases.

Usage:
  python akew_stats.py <result.json> [<result.json> ...]
  (accepts either raw JSON files or logs containing a <<<JSON>>> block)
"""
import json
import math
import re
import sys

import numpy as np


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Chosen over the normal
    approximation because these cells are small and several sit at or near
    100%, where the normal interval extends above 1.0 and is meaningless."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_hits, b_hits):
    """Exact McNemar test on paired binary outcomes.

    Only the DISCORDANT pairs carry information: cases where both conditions
    were right, or both wrong, say nothing about which is better. b = a-right/
    b-wrong, c = a-wrong/b-right; under the null those split 50/50, so the
    p-value is a two-sided binomial tail. Exact rather than chi-square
    because the discordant counts here are frequently under 10, where the
    chi-square approximation is not trustworthy.
    """
    a = np.asarray(a_hits, dtype=int)
    b = np.asarray(b_hits, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs equal-length vectors, got {a.shape} vs {b.shape}")
    n01 = int(((a == 0) & (b == 1)).sum())   # a wrong, b right
    n10 = int(((a == 1) & (b == 0)).sum())   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n_discordant": 0, "b_better": 0, "a_better": 0, "p_value": 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {"n_discordant": n, "b_better": n01, "a_better": n10, "p_value": p}


def paired_bootstrap_diff(a_hits, b_hits, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap CI for (mean(b) - mean(a)), resampling EXAMPLES (keeping each
    example's pair of outcomes together) rather than resampling the two
    conditions independently -- the pairing is the whole point."""
    a = np.asarray(a_hits, dtype=float)
    b = np.asarray(b_hits, dtype=float)
    n = a.shape[0]
    if n == 0:
        return {"diff": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(b.mean() - a.mean()), "lo": float(lo), "hi": float(hi)}


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns list of (index, p, adjusted_alpha,
    reject). Uniformly more powerful than plain Bonferroni at the same
    family-wise error rate, and necessary here because testing nine cells
    uncorrected will produce a spurious 'significant' result by chance."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [None] * m
    rejected_so_far = True
    for rank, i in enumerate(order):
        adj = alpha / (m - rank)
        reject = rejected_so_far and (pvals[i] <= adj)
        if not reject:
            rejected_so_far = False       # once one fails, all later ones fail
        out[i] = {"p": pvals[i], "adjusted_alpha": adj, "reject": reject}
    return out


def load_result(path):
    with open(path) as f:
        text = f.read()
    m = re.search(r"<<<JSON>>>\s*(\{.*?\})\s*<<<END>>>", text, re.S)
    blob = m.group(1) if m else text
    return json.loads(blob)


def analyse(result):
    per = result.get("per_example_hits") or {}
    if not per.get("fixed"):
        return None
    cell = f"{result['dataset']}/{result['input_mode']}"
    n = len(per["fixed"])

    conditions = {k: v for k, v in per.items() if v}
    acc_rows = {}
    for name, hits in conditions.items():
        k = int(sum(hits))
        lo, hi = wilson_interval(k, n)
        acc_rows[name] = {
            "accuracy": round(k / n, 4),
            "wilson_95ci": [round(lo, 4), round(hi, 4)],
            "k": k, "n": n,
        }

    comparisons = {}
    # Every comparative claim made in the results doc, tested paired.
    for base, new in [("fixed", "adaptive"), ("always_reason", "adaptive"),
                      ("fixed", "threeway"), ("adaptive", "threeway")]:
        if base in conditions and new in conditions:
            mc = mcnemar_exact(conditions[base], conditions[new])
            bs = paired_bootstrap_diff(conditions[base], conditions[new])
            comparisons[f"{new}_vs_{base}"] = {
                "diff": round(bs["diff"], 4),
                "bootstrap_95ci": [round(bs["lo"], 4), round(bs["hi"], 4)],
                "mcnemar_p": round(mc["p_value"], 5),
                "n_discordant": mc["n_discordant"],
                "wins_for_new": mc["b_better"], "wins_for_base": mc["a_better"],
            }
    return {"cell": cell, "n": n, "model": result.get("model"),
            "accuracies": acc_rows, "comparisons": comparisons}


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: python akew_stats.py <result.json|log> [...]", file=sys.stderr)
        sys.exit(2)

    analyses = []
    for p in paths:
        try:
            res = load_result(p)
        except Exception as e:
            print(f"SKIP {p}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        a = analyse(res)
        if a is None:
            print(f"SKIP {p}: no per_example_hits (rerun with the updated eval "
                  f"script -- aggregate-only logs cannot support a paired test)",
                  file=sys.stderr)
            continue
        analyses.append(a)

    # Family-wise correction across the headline adaptive-vs-fixed claim in
    # every cell analysed, since that is the claim being made repeatedly.
    key = "adaptive_vs_fixed"
    idx = [i for i, a in enumerate(analyses) if key in a["comparisons"]]
    if idx:
        pv = [analyses[i]["comparisons"][key]["mcnemar_p"] for i in idx]
        holm = holm_bonferroni(pv)
        for slot, i in enumerate(idx):
            analyses[i]["comparisons"][key]["holm"] = holm[slot]

    print("<<<JSON>>>")
    print(json.dumps({"n_cells": len(analyses), "cells": analyses}, indent=2))
    print("<<<END>>>")
