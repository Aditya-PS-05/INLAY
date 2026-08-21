"""
The decision-theoretic router head (v2).

v1 (akew_reliability.py) predicts ONE thing: P(top-1 retrieval is correct). It
does that well, and two separate experiments then showed it is the wrong
question -- see akew_outcome_labels.py's docstring for the full diagnosis.

v2 predicts THREE things, one per action:

    P(REJECT  yields a correct answer | features)
    P(DIRECT  yields a correct answer | features)
    P(REASON  yields a correct answer | features)

and the router takes the argmax. That is the quantity the router actually needs
and it subsumes v1's policy as a special case: if reasoning over retrieved
evidence always wins when retrieval is good, v2 learns exactly that, and it
learns it per-regime rather than being told.

WHY THREE SEPARATE BINARY HEADS rather than one 3-way classifier: the actions
are not mutually exclusive outcomes. On easy queries several actions all
succeed (the CounterFact-structured smoke test had 19 of 20 queries where
multiple actions were correct), so "which one is best" is frequently a tie and
a softmax over a forced single winner would be fitting noise in the
tie-breaking. Three independent P(success) estimates model the data as it
actually is, and ties fall out naturally as near-equal scores.

DIRECT is trained ONLY on structured-mode rows, where it is a legal action.
akew_router gates DIRECT on input_mode == "structured", so rows from other
modes carry y_direct = None; feeding those in as zeros would teach the head to
avoid an action that was never on the menu.

Same protocol as v1: linear, standardized, trained on CounterFact + WikiUpdate
with MQuAKE-CF held out entirely, so OOD stays genuinely OOD.
"""
import json
import os
import pathlib

import numpy as np

from akew_reliability import FEATURE_NAMES, _sigmoid

ACTIONS = ("reject", "direct", "reason")


class OutcomeHead:
    """Three calibrated binary heads sharing one feature vector."""

    def __init__(self):
        self.models = {}       # action -> (coef, intercept)
        self.mu = None
        self.sigma = None
        self.feature_names = list(FEATURE_NAMES)
        self.n_train = {}

    # --- fitting --------------------------------------------------------
    def fit(self, X, labels, C=1.0, max_iter=2000, seed=0):
        """X: (n, d). labels: dict action -> array of 0/1/None, length n.
        None entries are DROPPED for that action only (not imputed), so each
        head trains on exactly the rows where its action was legal."""
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma < 1e-8] = 1.0
        Xs = (X - self.mu) / self.sigma

        for a in ACTIONS:
            y_raw = labels[a]
            mask = np.array([v is not None for v in y_raw])
            if mask.sum() == 0:
                continue
            y = np.array([v for v in y_raw if v is not None], dtype=np.int64)
            Xa = Xs[mask]
            self.n_train[a] = int(mask.sum())
            # A head whose action never succeeds (or always does) in training
            # has no gradient to learn from; store the constant rather than
            # letting LogisticRegression raise on a single-class target.
            if len(np.unique(y)) < 2:
                self.models[a] = ("constant", float(y[0]))
                continue
            m = LogisticRegression(C=C, max_iter=max_iter, random_state=seed,
                                   class_weight="balanced")
            m.fit(Xa, y)
            self.models[a] = (m.coef_[0].copy(), float(m.intercept_[0]))
        return self

    # --- inference ------------------------------------------------------
    def _p(self, a, Xs):
        entry = self.models.get(a)
        if entry is None:
            return np.zeros(Xs.shape[0])
        coef, intercept = entry
        if isinstance(coef, str) and coef == "constant":
            return np.full(Xs.shape[0], intercept)
        return _sigmoid(Xs @ coef + intercept)

    def predict_proba(self, X):
        """Returns dict action -> P(success), each shape (n,)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.mu) / self.sigma
        return {a: self._p(a, Xs) for a in ACTIONS}

    def best_action(self, X, allow_direct=True, margin=0.0):
        """Argmax over legal actions.

        allow_direct: the caller passes False outside structured mode, so the
        head can never select an action the router would refuse to execute --
        the legality rule stays in one place rather than being relearned.
        margin: REASON is the incumbent default, so an alternative must beat it
        by more than `margin` to be chosen. Guards against churn on
        near-ties, which the smoke test showed are the common case.
        """
        p = self.predict_proba(X)
        out = []
        n = len(next(iter(p.values())))
        for i in range(n):
            scores = {"reject": p["reject"][i], "reason": p["reason"][i]}
            if allow_direct:
                scores["direct"] = p["direct"][i]
            base = scores["reason"]
            best, bestv = "reason", base
            for a, v in scores.items():
                if a == "reason":
                    continue
                if v > bestv + margin:
                    best, bestv = a, v
            out.append((best.upper(), {k: float(v) for k, v in scores.items()}))
        return out

    # --- persistence ----------------------------------------------------
    def save(self, path):
        if not self.models:
            raise RuntimeError("refusing to save an unfitted OutcomeHead")
        payload = {
            "feature_names": self.feature_names,
            "mu": self.mu.tolist(), "sigma": self.sigma.tolist(),
            "n_train": self.n_train,
            "models": {
                a: ({"kind": "constant", "value": v[1]}
                    if isinstance(v[0], str)
                    else {"kind": "linear", "coef": list(map(float, v[0])),
                          "intercept": v[1]})
                for a, v in self.models.items()
            },
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pathlib.Path(path).write_text(json.dumps(payload, indent=2))
        if not os.path.exists(path):
            raise IOError(f"save reported success but {path} is missing")
        return path

    @classmethod
    def load(cls, path):
        payload = json.loads(pathlib.Path(path).read_text())
        if payload["feature_names"] != FEATURE_NAMES:
            raise ValueError(
                "feature-order mismatch between checkpoint and current "
                "FEATURE_NAMES; coefficients are positional, so loading across "
                "a reordering would silently corrupt every prediction")
        h = cls()
        h.feature_names = payload["feature_names"]
        h.mu = np.asarray(payload["mu"], dtype=np.float64)
        h.sigma = np.asarray(payload["sigma"], dtype=np.float64)
        h.n_train = payload.get("n_train", {})
        for a, m in payload["models"].items():
            if m["kind"] == "constant":
                h.models[a] = ("constant", float(m["value"]))
            else:
                h.models[a] = (np.asarray(m["coef"], dtype=np.float64),
                               float(m["intercept"]))
        return h


def load_label_files(paths):
    """Concatenate outcome-label JSON files into (X, labels, meta)."""
    X, meta = [], []
    labels = {a: [] for a in ACTIONS}
    for p in paths:
        blob = json.loads(pathlib.Path(p).read_text())
        if blob["feature_names"] != FEATURE_NAMES:
            raise ValueError(f"{p}: feature-order mismatch with current FEATURE_NAMES")
        for r in blob["rows"]:
            X.append(r["features"])
            labels["reject"].append(r["y_reject"])
            labels["reason"].append(r["y_reason"])
            labels["direct"].append(r["y_direct"])
            meta.append({"dataset": r["dataset"], "mode": r["mode"],
                         "edit_id": r["edit_id"],
                         "retrieval_correct": r["retrieval_correct"]})
    return np.asarray(X, dtype=np.float64), labels, meta
