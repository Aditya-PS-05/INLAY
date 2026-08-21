"""
How much is there left for ANY router to win?

Before training the decision-theoretic head (v2), this asks the prior question:
if a router chose the best available action on EVERY query -- an oracle no real
model can beat -- how much better than a fixed policy would it be? That gap is
the total headroom available to any routing method, learned or otherwise. If it
is near zero, v2 cannot help and neither can anything else, and the honest move
is to say so rather than train a model to chase it.

Uses the outcome labels from akew_outcome_labels.py, which recorded, for each
query, whether each action ACTUALLY produced a correct answer.

Policies compared:
  oracle          max over legal actions (unbeatable upper bound)
  always_reason   what the adaptive router converges to on non-structured modes
  static_best     DIRECT where legal, else REASON (no per-query decision at all)
  oracle_no_rej   oracle restricted to {DIRECT, REASON} -- isolates how much of
                  the oracle's advantage depends on REJECT being available

Usage: python akew_headroom.py <label_json> [<label_json> ...]
"""
import glob
import json
import pathlib
import sys


def analyse(path):
    blob = json.loads(pathlib.Path(path).read_text())
    rows = blob["rows"]
    if not rows:
        return None
    n = len(rows)

    def legal(r):
        acts = {"reject": r["y_reject"], "reason": r["y_reason"]}
        if r["y_direct"] is not None:
            acts["direct"] = r["y_direct"]
        return acts

    oracle = sum(1 for r in rows if any(v == 1 for v in legal(r).values()))
    oracle_no_rej = sum(1 for r in rows
                        if any(v == 1 for k, v in legal(r).items() if k != "reject"))
    always_reason = sum(r["y_reason"] for r in rows)
    static_best = sum((r["y_direct"] if r["y_direct"] is not None else r["y_reason"])
                      for r in rows)
    # Queries where REJECT is the ONLY thing that works -- the entire case for
    # keeping an abstention path in the action set.
    rej_only = sum(1 for r in rows
                   if r["y_reject"] == 1
                   and r["y_reason"] == 0
                   and (r["y_direct"] in (0, None)))

    return {
        "dataset": blob["dataset"], "mode": blob["mode"], "n": n,
        "oracle": oracle / n,
        "oracle_no_reject": oracle_no_rej / n,
        "static_best": static_best / n,
        "always_reason": always_reason / n,
        "headroom_over_static": (oracle - static_best) / n,
        "reject_only_wins": rej_only,
    }


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(sorted(glob.glob(a)))
    results = [r for r in (analyse(p) for p in paths) if r]
    if not results:
        print("no label files matched", file=sys.stderr)
        sys.exit(2)

    print(f"{'cell':28} {'n':>4} {'static':>8} {'oracle':>8} {'headroom':>9} "
          f"{'oracle-Rej':>11} {'REJ-only':>9}")
    tot_n = tot_static = tot_oracle = tot_rejonly = 0
    for r in sorted(results, key=lambda x: (x["dataset"], x["mode"])):
        print("%-28s %4d %8.4f %8.4f %+9.4f %11.4f %9d"
              % (f"{r['dataset']}/{r['mode']}", r["n"], r["static_best"],
                 r["oracle"], r["headroom_over_static"], r["oracle_no_reject"],
                 r["reject_only_wins"]))
        tot_n += r["n"]
        tot_static += r["static_best"] * r["n"]
        tot_oracle += r["oracle"] * r["n"]
        tot_rejonly += r["reject_only_wins"]

    print("-" * 82)
    print("%-28s %4d %8.4f %8.4f %+9.4f %11s %9d"
          % ("POOLED", tot_n, tot_static / tot_n, tot_oracle / tot_n,
             (tot_oracle - tot_static) / tot_n, "", tot_rejonly))
    print()
    print(f"Total queries where REJECT is the only winning action: "
          f"{tot_rejonly} / {tot_n} ({100*tot_rejonly/tot_n:.2f}%)")
    print(f"Maximum possible gain of ANY per-query router over the static "
          f"policy: {100*(tot_oracle-tot_static)/tot_n:.2f} points")


if __name__ == "__main__":
    main()
