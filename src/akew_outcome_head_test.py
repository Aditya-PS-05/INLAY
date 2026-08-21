"""
Self-tests for akew_outcome_head.py.

The v2 head chooses which ACTION the router takes, so a silent bug here does
not degrade a metric -- it changes behaviour on every query while still
producing plausible numbers. The checks below therefore target the specific
ways this class can be quietly wrong: DIRECT leaking into modes where it is
illegal, the None-labelled rows being imputed instead of dropped, and the
positional coefficients being loaded against a reordered feature list.

Usage: python akew_outcome_head_test.py   (needs sklearn; run on a GPU host)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from akew_outcome_head import ACTIONS, OutcomeHead
from akew_reliability import FEATURE_NAMES

fails = []


def ck(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name +
          (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(0)
n, d = 400, len(FEATURE_NAMES)
X = rng.normal(size=(n, d))
vm = FEATURE_NAMES.index("ver_margin_12")

# Synthetic world with a known answer: REASON pays off when the verifier margin
# is high, REJECT when it is low. DIRECT is labelled on only half the rows, so
# the None-dropping path is exercised rather than assumed.
y_reason = (X[:, vm] > 0).astype(int)
y_reject = (X[:, vm] <= 0).astype(int)
y_direct = [int(X[i, vm] > 0.5) if i < n // 2 else None for i in range(n)]

print("fit:")
h = OutcomeHead().fit(X, {"reject": y_reject, "reason": y_reason, "direct": y_direct})
ck("all three heads fitted", set(h.models) == set(ACTIONS), sorted(h.models))
ck("DIRECT trained only on its labelled rows", h.n_train["direct"] == n // 2, h.n_train)
ck("REASON/REJECT trained on every row",
   h.n_train["reason"] == n and h.n_train["reject"] == n, h.n_train)

print("learned signal:")
p = h.predict_proba(X)
ck("REASON head separates its outcome",
   p["reason"][y_reason == 1].mean() > p["reason"][y_reason == 0].mean() + 0.3)
ck("REJECT head separates the opposite way",
   p["reject"][y_reject == 1].mean() > p["reject"][y_reject == 0].mean() + 0.3)

print("action legality (the rule that must never be relearned):")
hi = X[X[:, vm] > 1.0][:20]
ck("allow_direct=False never returns DIRECT",
   all(a != "DIRECT" for a, _ in h.best_action(hi, allow_direct=False)))

# Whether DIRECT beats REASON on the shared synthetic above is calibration
# noise -- both saturate near 1.0 where the margin is high, so the winner is
# arbitrary and asserting one is a flaky test rather than a real check. Fit a
# head where DIRECT is unambiguously the right action instead, which tests the
# selection mechanism rather than an accident of the fit.
h_direct = OutcomeHead().fit(
    X, {"reject": [0] * n, "reason": [0] * n, "direct": [1] * n})
picks = h_direct.best_action(X[:20], allow_direct=True)
ck("DIRECT is selected when it is clearly the best action",
   all(a == "DIRECT" for a, _ in picks), picks[:3])
ck("...and is still suppressed there when illegal",
   all(a != "DIRECT" for a, _ in h_direct.best_action(X[:20], allow_direct=False)))

print("margin behaviour:")
ck("large margin pins everything to the REASON default",
   all(a == "REASON" for a, _ in h.best_action(X[:50], allow_direct=True, margin=1.5)))
ck("zero margin allows alternatives",
   any(a != "REASON" for a, _ in h.best_action(X[:50], allow_direct=True, margin=0.0)))

print("degenerate targets:")
h2 = OutcomeHead().fit(X, {"reject": [0] * n, "reason": y_reason, "direct": [None] * n})
ck("single-class target becomes a constant head, not a crash",
   h2.models["reject"][0] == "constant", h2.models.get("reject"))
ck("never-legal action is absent from the model set", "direct" not in h2.models)
ck("constant head still predicts", h2.predict_proba(X[:3])["reject"].shape == (3,))

print("persistence:")
with tempfile.TemporaryDirectory() as td:
    pth = os.path.join(td, "h.json")
    h.save(pth)
    ck("save writes a file", os.path.exists(pth))
    reloaded = OutcomeHead.load(pth)
    p2 = reloaded.predict_proba(X)
    ck("load reproduces predictions exactly",
       all(np.abs(p[a] - p2[a]).max() < 1e-9 for a in ACTIONS))

    blob = json.loads(open(pth).read())
    blob["feature_names"] = list(reversed(FEATURE_NAMES))
    bad = os.path.join(td, "bad.json")
    open(bad, "w").write(json.dumps(blob))
    try:
        OutcomeHead.load(bad)
        ck("reordered checkpoint refuses to load", False, "it loaded silently")
    except ValueError:
        ck("reordered checkpoint refuses to load", True)

try:
    OutcomeHead().save(os.path.join(tempfile.gettempdir(), "unfitted.json"))
    ck("unfitted save raises", False, "no exception")
except RuntimeError:
    ck("unfitted save raises RuntimeError", True)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")
