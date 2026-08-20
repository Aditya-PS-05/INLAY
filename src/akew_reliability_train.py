"""
Trains the retrieval-reliability head (akew_reliability.py) and tests the one
claim the whole method rests on: that MQuAKE-CF's unreliable-retrieval regime
is detectable from retrieval SHAPE alone, by a head that has never seen
MQuAKE-CF.

PROTOCOL (deliberately mirrors the verifier v1/v2 protocol already used in
this project, for the same reason):
  train/val:  CounterFact + WikiUpdate, all three input modes,
              subject-disjoint splits within each
  held out:   MQuAKE-CF, ENTIRELY -- never seen in training, not even its
              train split, so the out-of-domain number is genuinely
              out-of-domain

If the head only worked by memorizing which dataset it was looking at, it
would be the manual per-dataset bypass with extra steps and no value. It is
never given dataset identity as a feature and never sees the held-out
dataset, so it cannot cheat that way -- the OOD evaluation below is the real
test, and it is reported whichever way it comes out.

Usage: python akew_reliability_train.py [limit_per_dataset_mode]
"""
import sys, json, random

sys.path.insert(0, "src")
import numpy as np

from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_reliability import ReliabilityHead, extract_features, score_candidates, FEATURE_NAMES
from akew_verifier_eval import auroc_auprc, expected_calibration_error
from sentence_transformers import CrossEncoder

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
VERIFIER_PATH = "outputs/akew_verifier_ckpt_v2"
OUT_PATH = "outputs/akew_reliability_head.json"
TOPK = 5
DIRECT_THRESHOLD = 0.85

TRAIN_DATASETS = ["CounterFact", "WikiUpdate"]
HELDOUT_DATASET = "MQuAKE-CF"
MODES = ["structured", "unstructured", "extracted"]

verifier = CrossEncoder(VERIFIER_PATH)


def build_examples(dataset, mode, split, limit=LIMIT):
    """One example per query: retrieve top-k over the FULL card pool (same
    realistic difficulty the live pipeline faces), score all k with the
    verifier, extract features, label by whether top-1 is actually the right
    card. The index is built over every card in the dataset/mode, not just
    the split's cards -- a deployed system retrieves against everything it
    knows, so restricting the pool to the split would make retrieval
    artificially easy and the labels meaningless."""
    cards, golds, _groups = load_akew(dataset, mode)
    index = DenseCardIndex()
    index.build(cards)

    tr, va, te = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
    subset = {"train": tr, "val": va, "test": te}[split]
    random.seed(0)
    if limit and limit < len(subset):
        subset = random.sample(subset, limit)

    X, y, meta = [], [], []
    for c in subset:
        g = golds.get(c.edit_id)
        if not g or not g.eval_question:
            continue
        q = g.eval_question
        cands = index.query(q, topk=TOPK)
        if not cands:
            continue
        vscores = score_candidates(verifier, q, cands)
        vec, feats = extract_features(q, cands, vscores, DIRECT_THRESHOLD)
        label = 1 if cands[0][0].edit_id == c.edit_id else 0
        X.append(vec)
        y.append(label)
        meta.append({"dataset": dataset, "mode": mode, "edit_id": c.edit_id})
    return np.asarray(X), np.asarray(y), meta


def collect(datasets, split):
    Xs, ys, ms = [], [], []
    for ds in datasets:
        for mode in MODES:
            X, y, m = build_examples(ds, mode, split)
            if len(X):
                Xs.append(X); ys.append(y); ms.extend(m)
            print(f"  {ds}/{mode}/{split}: n={len(X)}, retrieval_correct_rate="
                  f"{(y.mean() if len(y) else float('nan')):.4f}", file=sys.stderr)
    if not Xs:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), []
    return np.vstack(Xs), np.concatenate(ys), ms


print("Building TRAIN examples (CounterFact + WikiUpdate)...", file=sys.stderr)
Xtr, ytr, _mtr = collect(TRAIN_DATASETS, "train")
print("Building VAL examples (CounterFact + WikiUpdate)...", file=sys.stderr)
Xva, yva, _mva = collect(TRAIN_DATASETS, "val")
print("Building IN-DOMAIN TEST examples (CounterFact + WikiUpdate)...", file=sys.stderr)
Xte, yte, _mte = collect(TRAIN_DATASETS, "test")
print(f"Building OOD examples ({HELDOUT_DATASET}, never seen)...", file=sys.stderr)
Xood, yood, mood = collect([HELDOUT_DATASET], "test")

head = ReliabilityHead().fit(Xtr, ytr)
head.save(OUT_PATH)


def evaluate(X, y, label):
    if not len(X):
        return {"split": label, "n": 0}
    p = head.predict_proba(X)
    # NOTE the argument order: akew_verifier_eval's helpers take
    # (scores, labels), NOT (labels, scores). Passing them the other way round
    # silently returns a plausible-looking but meaningless number rather than
    # raising, which is exactly the kind of quiet wrongness this project has
    # already been bitten by -- checked against the signature, not assumed.
    auroc, auprc = auroc_auprc(p, y)
    ece = expected_calibration_error(p, y)
    return {
        "split": label,
        "n": int(len(y)),
        "base_rate_retrieval_correct": round(float(y.mean()), 4),
        "mean_predicted_reliability": round(float(p.mean()), 4),
        # auroc_auprc returns (None, None) when a split is single-class, which
        # is a real possibility on small OOD slices; surfaced as null rather
        # than crashed on or faked as 0.5.
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
    }


results = {
    "train": evaluate(Xtr, ytr, "train (CF+Wiki)"),
    "val": evaluate(Xva, yva, "val (CF+Wiki)"),
    "in_domain_test": evaluate(Xte, yte, "in-domain test (CF+Wiki)"),
    "ood_test": evaluate(Xood, yood, f"OOD test ({HELDOUT_DATASET}, never seen)"),
}

# The central claim, checked directly on the fitted weights rather than
# asserted: do the MARGIN features (the "confidence of confidence" signal)
# actually carry weight, or is the head just relearning a ver_top1 threshold
# -- which every prior threshold experiment already showed cannot work?
coefs = head.coefficients()
margin_features = ["ver_margin_12", "ver_n_above_direct", "emb_entropy",
                   "emb_margin_12", "emb_margin_15", "ver_std_topk", "subject_diversity"]
top1_features = ["ver_top1", "emb_top1"]
margin_mass = sum(abs(coefs[f]) for f in margin_features)
top1_mass = sum(abs(coefs[f]) for f in top1_features)

# Honest comparison against a higher-capacity model, reported either way.
gbm_result = None
try:
    from sklearn.ensemble import GradientBoostingClassifier
    gbm = GradientBoostingClassifier(random_state=0)
    gbm.fit(Xtr, ytr)
    gbm_ood_p = gbm.predict_proba(Xood)[:, 1] if len(Xood) else np.zeros(0)
    gbm_id_p = gbm.predict_proba(Xte)[:, 1] if len(Xte) else np.zeros(0)
    if len(Xood) and len(Xte):
        g_ood_auroc, g_ood_auprc = auroc_auprc(gbm_ood_p, yood)
        g_id_auroc, g_id_auprc = auroc_auprc(gbm_id_p, yte)
        gbm_result = {
            "in_domain_auroc": g_id_auroc,
            "in_domain_auprc": g_id_auprc,
            "ood_auroc": g_ood_auroc,
            "ood_auprc": g_ood_auprc,
            "ood_mean_predicted": round(float(gbm_ood_p.mean()), 4),
        }
except Exception as e:  # pragma: no cover - reported, never silently skipped
    gbm_result = {"error": f"{type(e).__name__}: {e}"}

out = {
    "verifier": VERIFIER_PATH,
    "topk": TOPK,
    "train_datasets": TRAIN_DATASETS,
    "heldout_dataset": HELDOUT_DATASET,
    "modes": MODES,
    "head_path": OUT_PATH,
    "results": results,
    "coefficients": {k: round(v, 4) for k, v in coefs.items()},
    "coefficient_mass": {
        "margin_features_abs_sum": round(margin_mass, 4),
        "top1_features_abs_sum": round(top1_mass, 4),
        "margin_share": round(margin_mass / (margin_mass + top1_mass), 4)
        if (margin_mass + top1_mass) > 0 else None,
    },
    "gbm_comparison": gbm_result,
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
