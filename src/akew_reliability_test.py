"""
Self-tests for akew_reliability.py. Runs the pure-numpy paths anywhere
(feature extraction, entropy, the save/load round-trip and its feature-order
guard) and the sklearn-dependent fit path only where sklearn is installed,
skipping loudly rather than silently passing when it is not.

Written BEFORE the first GPU run, deliberately: the head's coefficients are
positional and its features are second-order statistics, so a transposed
axis or a swapped margin would produce plausible-looking numbers rather than
an error -- precisely the class of quiet wrongness this project has already
been bitten by (the CrossEncoder save that persisted nothing, the
(scores, labels) argument order that returns a meaningless AUROC rather than
raising). Cheap to check here, expensive to discover after a full sweep.

Usage: python akew_reliability_test.py
"""
import sys, os, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from akew_reliability import (FEATURE_NAMES, ReliabilityHead, _entropy,
                              extract_features)


class _FakeCard:
    def __init__(self, edit_id, subject, text):
        self.edit_id = edit_id
        self.subject = subject
        self.canonical_fact_text = text
        self.raw_evidence_text = None
        self.input_mode = "structured"


failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


print("entropy:")
# A flat neighborhood (every candidate equally scored) is maximal ambiguity;
# a peaked one is minimal. If these come out backwards the head learns the
# opposite of the intended signal while still fitting fine.
check("flat scores -> entropy ~1", abs(_entropy([1.0, 1.0, 1.0, 1.0]) - 1.0) < 1e-6,
      f"got {_entropy([1.0, 1.0, 1.0, 1.0])}")
peaked = _entropy([10.0, 0.0, 0.0, 0.0])
check("peaked scores -> low entropy", peaked < 0.35, f"got {peaked}")
check("single candidate -> 0", _entropy([1.0]) == 0.0)

print("feature extraction:")
cands = [
    (_FakeCard("e1", "Alice", "Alice works in Paris"), 0.90),
    (_FakeCard("e2", "Bob", "Bob works in Rome"), 0.60),
    (_FakeCard("e3", "Alice", "Alice studied in Bonn"), 0.55),
]
vscores = [0.99, 0.40, 0.30]
vec, feats = extract_features("Where does Alice work?", cands, vscores)

check("vector length matches FEATURE_NAMES", vec.shape == (len(FEATURE_NAMES),),
      f"got {vec.shape} vs {len(FEATURE_NAMES)}")
check("no NaNs", not np.isnan(vec).any())
check("emb_top1 correct", abs(feats["emb_top1"] - 0.90) < 1e-9)
check("emb_margin_12 correct", abs(feats["emb_margin_12"] - 0.30) < 1e-9,
      f"got {feats['emb_margin_12']}")
check("emb_margin_15 uses last candidate", abs(feats["emb_margin_15"] - 0.35) < 1e-9,
      f"got {feats['emb_margin_15']}")
check("ver_top1 correct", abs(feats["ver_top1"] - 0.99) < 1e-9)
check("ver_max_rest excludes top1", abs(feats["ver_max_rest"] - 0.40) < 1e-9,
      f"got {feats['ver_max_rest']}")
check("ver_margin_12 correct", abs(feats["ver_margin_12"] - 0.59) < 1e-9,
      f"got {feats['ver_margin_12']}")
check("ver_n_above_direct counts only >=0.85", feats["ver_n_above_direct"] == 1.0,
      f"got {feats['ver_n_above_direct']}")
check("subject_diversity = 2 distinct / 3", abs(feats["subject_diversity"] - 2 / 3) < 1e-9,
      f"got {feats['subject_diversity']}")
check("vector order matches dict", all(
    abs(vec[i] - feats[name]) < 1e-12 for i, name in enumerate(FEATURE_NAMES)))

print("the discriminative-confidence case (the whole point of the method):")
# Same high top-1 verifier score in both, but one is discriminative and the
# other is saturated across every candidate. A top-1 threshold cannot tell
# these apart -- that is exactly why every threshold experiment failed. The
# margin features MUST separate them or the method has no mechanism.
_vec_sharp, sharp = extract_features("q", cands, [0.99, 0.20, 0.10])
_vec_flat, flat = extract_features("q", cands, [0.99, 0.98, 0.97])
check("identical ver_top1 in both cases", sharp["ver_top1"] == flat["ver_top1"])
check("margin separates sharp from flat", sharp["ver_margin_12"] > flat["ver_margin_12"] + 0.5,
      f"sharp={sharp['ver_margin_12']} flat={flat['ver_margin_12']}")
check("n_above_direct separates them", flat["ver_n_above_direct"] > sharp["ver_n_above_direct"],
      f"sharp={sharp['ver_n_above_direct']} flat={flat['ver_n_above_direct']}")

print("error handling:")
try:
    extract_features("q", [], [])
    check("empty candidates raises", False, "no exception raised")
except ValueError:
    check("empty candidates raises ValueError", True)
try:
    extract_features("q", cands, [0.9, 0.8])   # length mismatch
    check("length mismatch raises", False, "no exception raised")
except ValueError:
    check("length mismatch raises ValueError", True)

try:
    import sklearn  # noqa: F401
    have_sklearn = True
except ImportError:
    have_sklearn = False

if not have_sklearn:
    print("fit/save/load: SKIPPED (sklearn not installed here) -- must be run "
          "where sklearn is available before trusting a trained head")
else:
    print("fit / save / load round-trip:")
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    # Make the label genuinely depend on the ver_margin_12 column so a broken
    # standardization or a transposed matrix shows up as a dead fit.
    margin_idx = FEATURE_NAMES.index("ver_margin_12")
    y = (X[:, margin_idx] + 0.3 * rng.normal(size=n) > 0).astype(int)

    head = ReliabilityHead().fit(X, y)
    p_before = head.predict_proba(X)
    check("fit learns the signal (train AUROC-ish separation)",
          p_before[y == 1].mean() > p_before[y == 0].mean() + 0.2,
          f"pos={p_before[y == 1].mean():.3f} neg={p_before[y == 0].mean():.3f}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "head.json")
        head.save(path)
        check("save produces a file", os.path.exists(path))
        reloaded = ReliabilityHead.load(path)
        p_after = reloaded.predict_proba(X)
        check("load reproduces predictions exactly",
              np.abs(p_before - p_after).max() < 1e-9,
              f"max diff {np.abs(p_before - p_after).max()}")

        # The feature-order guard: a reordered checkpoint must refuse to load
        # rather than silently mispredicting with positionally-shifted weights.
        import json
        with open(path) as f:
            payload = json.load(f)
        payload["feature_names"] = list(reversed(payload["feature_names"]))
        bad = os.path.join(td, "bad.json")
        with open(bad, "w") as f:
            json.dump(payload, f)
        try:
            ReliabilityHead.load(bad)
            check("reordered checkpoint refuses to load", False, "loaded silently!")
        except ValueError:
            check("reordered checkpoint refuses to load", True)

    print("unfitted-head guards:")
    fresh = ReliabilityHead()
    try:
        fresh.predict_proba(X)
        check("unfitted predict raises", False, "no exception")
    except RuntimeError:
        check("unfitted predict raises RuntimeError", True)
    try:
        fresh.save(os.path.join(tempfile.gettempdir(), "should_not_exist.json"))
        check("unfitted save raises", False, "no exception")
    except RuntimeError:
        check("unfitted save raises RuntimeError", True)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
