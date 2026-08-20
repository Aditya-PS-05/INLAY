"""
Retrieval-reliability calibration head -- "confidence of confidence."

WHY THIS EXISTS (the diagnosed bug it fixes):

akew_fullpipeline_results.md documents the same failure on MQuAKE-CF in
EVERY input mode tested: the router's REJECT/DIRECT gating is net-negative
there (structured 93.65% routed vs 96.83% always-REASON; extracted 69.84%
vs 85.71%), while the identical gating helps or ties on CounterFact and
WikiUpdate in every mode. Two separate experiments already ruled out the
obvious fixes:

  - akew_verifier_recalibrate.py: threshold recalibration barely moves the
    false-fire rate (18.33% -> 18.22%), because the verifier's positive and
    negative score distributions on MQuAKE-CF genuinely OVERLAP -- no scalar
    threshold separates them.
  - The direct_threshold sweep (0.85 / 0.97 / 1.01, same doc): raising the
    threshold to 0.97 changed NOTHING (identical decisions, identical
    accuracy), because the verifier's confidence on MQuAKE-CF's WRONG
    retrievals is already above 0.97.

The conclusion both experiments point to: the router is asking the wrong
question. It asks "how confident is the verifier about this one card?" when
what it needs to know is "is this verifier's confidence DISCRIMINATIVE right
now?" A verifier that scores 0.99 on the top card and 0.98 on four unrelated
cards is not confident -- it is uniformly saturated, and its top-1 pick is
close to arbitrary. That distinction is invisible to any threshold on the
top-1 score alone, which is exactly why every threshold-based fix failed.

WHAT THIS DOES DIFFERENTLY:

Predicts P(top-1 retrieval is actually correct) from the SHAPE of the
retrieval + verification result, not from its top-1 magnitude. The features
that carry the novel signal are the margin ones (ver_margin_12,
ver_n_above_direct, emb_entropy): they measure whether confidence is
concentrated on one candidate or smeared across many. This is the "confidence
of confidence" idea -- a second-order signal the first-order threshold cannot
express, however it is tuned.

THE HARD CONSTRAINT (what makes this a real method rather than the manual
per-dataset bypass with extra steps): every feature must be computable at
inference time from the query and the retrieval result ALONE. No dataset
identity, no dataset-level statistics, nothing the deployed system would not
have. The whole claim is that MQuAKE-CF's unreliable regime is DETECTABLE
from retrieval shape -- so the head is trained on CounterFact + WikiUpdate
with MQuAKE-CF held out entirely, mirroring the verifier v1/v2 protocol, and
must generalize to a dataset it has never seen. If it only worked by
memorizing dataset identity it would be worthless, so it is never given the
chance.

COST NOTE, stated plainly: this scores the top-k candidates with the
cross-encoder instead of only the top-1, so verification cost goes from 1
cross-encoder call per query to k (k=5 by default). Cross-encoder calls on
5 short pairs are cheap relative to the generation call that follows, but it
is a real k-fold increase in that stage and is reported as such rather than
buried.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Feature order is FIXED and explicit: the trained head's coefficients are
# positional, so a silent reordering would corrupt every prediction without
# raising anything. Any change here invalidates saved checkpoints, which is
# why the saved JSON carries this list and load() asserts against it.
FEATURE_NAMES = [
    "emb_top1",
    "emb_margin_12",
    "emb_margin_15",
    "emb_mean_topk",
    "emb_std_topk",
    "emb_entropy",
    "ver_top1",
    "ver_max_rest",
    "ver_margin_12",
    "ver_mean_topk",
    "ver_std_topk",
    "ver_n_above_direct",
    "subject_diversity",
    "query_len",
    "looks_multihop",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _entropy(scores):
    """Softmax entropy over the top-k retrieval scores, normalized to [0,1] by
    log(k). High entropy == the neighborhood is flat == no candidate stands
    out == retrieval is guessing. This is the embedding-space twin of the
    verifier-margin features."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size <= 1:
        return 0.0
    p = np.exp(s - s.max())
    p = p / p.sum()
    p = np.clip(p, 1e-12, 1.0)
    h = float(-(p * np.log(p)).sum())
    return h / math.log(s.size)


def extract_features(query_text, candidates, verifier_scores, direct_threshold=0.85):
    """Build the feature vector from an ALREADY-COMPUTED retrieval + verifier
    result, so the router can reuse both rather than paying for them twice.

    candidates:       list[(card, embedding_score)], descending, len k
    verifier_scores:  list[float] of calibrated (sigmoid) verifier scores,
                      aligned positionally with `candidates`

    Returns (feature_vector: np.ndarray, feature_dict: dict) -- the dict is
    for logging/inspection, the vector is what the head consumes.
    """
    from akew_router import _looks_multihop

    if not candidates:
        raise ValueError("extract_features called with no candidates; "
                         "the caller must handle the no-candidate case (a "
                         "structural REJECT) before reaching the head.")

    emb = np.asarray([s for _c, s in candidates], dtype=np.float64)
    ver = np.asarray(verifier_scores, dtype=np.float64)
    if ver.shape[0] != emb.shape[0]:
        raise ValueError(f"verifier_scores length {ver.shape[0]} != candidates "
                         f"length {emb.shape[0]}; positional alignment is "
                         f"required, not optional")

    emb_top1 = float(emb[0])
    emb_margin_12 = float(emb[0] - emb[1]) if emb.size > 1 else 0.0
    emb_margin_15 = float(emb[0] - emb[-1]) if emb.size > 1 else 0.0

    ver_top1 = float(ver[0])
    ver_rest = ver[1:]
    ver_max_rest = float(ver_rest.max()) if ver_rest.size else 0.0
    # THE key feature: if the verifier is nearly as confident about some other
    # card as about its top pick, its top-1 confidence is not discriminative,
    # no matter how high it is. This is what no threshold on ver_top1 can see.
    ver_margin_12 = ver_top1 - ver_max_rest

    # Distinct subjects among the top-k. Low diversity == the neighborhood is
    # crowded with near-duplicate entities, the structural property behind
    # both WikiUpdate's stale-object confusions and MQuAKE-CF's harder
    # retrieval (Stage 1 pilot). Normalized so k does not change its scale.
    subjects = {getattr(c, "subject", None) for c, _s in candidates}
    subject_diversity = len(subjects) / float(len(candidates))

    feats = {
        "emb_top1": emb_top1,
        "emb_margin_12": emb_margin_12,
        "emb_margin_15": emb_margin_15,
        "emb_mean_topk": float(emb.mean()),
        "emb_std_topk": float(emb.std()),
        "emb_entropy": _entropy(emb),
        "ver_top1": ver_top1,
        "ver_max_rest": ver_max_rest,
        "ver_margin_12": ver_margin_12,
        "ver_mean_topk": float(ver.mean()),
        "ver_std_topk": float(ver.std()),
        "ver_n_above_direct": float((ver >= direct_threshold).sum()),
        "subject_diversity": subject_diversity,
        "query_len": float(len(query_text.split())),
        "looks_multihop": 1.0 if _looks_multihop(query_text) else 0.0,
    }
    vec = np.asarray([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
    return vec, feats


def score_candidates(verifier, query_text, candidates):
    """Cross-encoder scores for ALL top-k candidates (not just top-1), sigmoid-
    calibrated to match akew_router.route()'s existing convention exactly.
    Batched into one predict() call -- k separate calls would be the same
    arithmetic at k times the overhead."""
    pairs = [[query_text, (c.canonical_fact_text or c.raw_evidence_text or "")]
             for c, _s in candidates]
    raw = verifier.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    return [float(v) for v in _sigmoid(raw)]


@dataclass
class ReliabilityPrediction:
    p_correct: float           # predicted P(top-1 retrieval is the right card)
    features: dict


class ReliabilityHead:
    """Logistic regression over the features above, with standardization.

    Deliberately a linear model, not a GBM or an MLP, for three reasons that
    all matter more here than raw fit quality: (1) the training set is small
    (low thousands of queries) and the whole claim rests on GENERALIZING to an
    unseen dataset, where a high-capacity model would be far likelier to
    memorize CounterFact/WikiUpdate's particular retrieval geometry; (2) the
    coefficients are directly inspectable, so the central claim -- that the
    MARGIN features carry signal the top-1 score does not -- becomes an
    empirical check on the fitted weights rather than an assertion; (3) it
    trains in under a second, so the honest comparison against a stronger
    model is cheap to run rather than something to hand-wave past. A gradient-
    boosted variant is fitted alongside in the training script and reported
    even where it wins, rather than being quietly omitted.
    """

    def __init__(self):
        self.model = None
        self.mu = None
        self.sigma = None
        self.feature_names = list(FEATURE_NAMES)

    def fit(self, X, y, C=1.0, max_iter=2000, seed=0):
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mu = X.mean(axis=0)
        # Guard against a constant feature producing a divide-by-zero that
        # would silently propagate NaNs into every downstream prediction.
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma < 1e-8] = 1.0
        Xs = (X - self.mu) / self.sigma
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed,
                                        class_weight="balanced")
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("ReliabilityHead used before fit()/load()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.mu) / self.sigma
        return self.model.predict_proba(Xs)[:, 1]

    def predict_one(self, query_text, candidates, verifier_scores, direct_threshold=0.85):
        vec, feats = extract_features(query_text, candidates, verifier_scores, direct_threshold)
        p = float(self.predict_proba(vec)[0])
        return ReliabilityPrediction(p_correct=p, features=feats)

    def coefficients(self):
        """Fitted weight per feature, in standardized units so magnitudes are
        directly comparable across features with different natural scales."""
        if self.model is None:
            raise RuntimeError("ReliabilityHead used before fit()/load()")
        return dict(zip(self.feature_names, [float(c) for c in self.model.coef_[0]]))

    def save(self, path):
        if self.model is None:
            raise RuntimeError("refusing to save an unfitted ReliabilityHead")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "coef": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        # Same discipline as the CrossEncoder save bug already found in this
        # project (fit() reported success while persisting nothing): assert the
        # artifact actually exists rather than trusting the write.
        if not os.path.exists(path):
            raise IOError(f"ReliabilityHead.save reported success but {path} does not exist")
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            payload = json.load(f)
        if payload["feature_names"] != FEATURE_NAMES:
            raise ValueError(
                "feature-order mismatch between checkpoint and current "
                "FEATURE_NAMES; the head's coefficients are positional, so "
                "loading across a reordering would silently corrupt every "
                f"prediction.\n  checkpoint: {payload['feature_names']}\n"
                f"  current:    {FEATURE_NAMES}")
        head = cls()
        head.feature_names = payload["feature_names"]
        head.mu = np.asarray(payload["mu"], dtype=np.float64)
        head.sigma = np.asarray(payload["sigma"], dtype=np.float64)

        class _LinearShim:
            """Reconstitutes predict_proba from the saved weights, so loading a
            head does not require sklearn at inference time."""
            def __init__(self, coef, intercept):
                self.coef_ = np.asarray([coef], dtype=np.float64)
                self.intercept_ = np.asarray([intercept], dtype=np.float64)

            def predict_proba(self, Xs):
                z = Xs @ self.coef_[0] + self.intercept_[0]
                p = _sigmoid(z)
                return np.stack([1.0 - p, p], axis=1)

        head.model = _LinearShim(payload["coef"], payload["intercept"])
        return head
