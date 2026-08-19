"""
Evaluate the trained scope verifier (brief section 7: router metrics) and run
the zero-shot generalization test the brief's section 9 suggests as an
alternative validation experiment: calibrate on CounterFact + WikiUpdate,
evaluate MQuAKE-CF WITHOUT any threshold retuning (it was never touched during
training or calibration).
"""
import sys, json
sys.path.insert(0, "src")
from akew_verifier_train import build_split_examples, TRAIN_DATASETS, MODES, assemble_examples
from akew_data import load_akew
from sentence_transformers import CrossEncoder
import numpy as np


def confusion_at_threshold(scores, labels, thresh):
    tp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 0)
    fn = sum(1 for s, l in zip(scores, labels) if s < thresh and l == 1)
    tn = sum(1 for s, l in zip(scores, labels) if s < thresh and l == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "false_fire_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "false_reject_rate": round(fn / (fn + tp), 4) if (fn + tp) else None}


def auroc_auprc(scores, labels):
    order = np.argsort(scores)[::-1]
    labels_sorted = np.array(labels)[order]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    if n_pos == 0 or n_neg == 0:
        return None, None
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    auroc = np.trapezoid(tpr, fpr)
    precisions = tps / (tps + fps)
    auprc = np.trapezoid(precisions, tpr)
    return round(float(auroc), 4), round(float(auprc), 4)


def expected_calibration_error(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(s >= lo and s < hi) for s in scores] if i < n_bins - 1 else [(s >= lo and s <= hi) for s in scores]
        bin_scores = [s for s, m in zip(scores, mask) if m]
        bin_labels = [l for l, m in zip(labels, mask) if m]
        if not bin_scores:
            continue
        conf = sum(bin_scores) / len(bin_scores)
        acc = sum(bin_labels) / len(bin_labels)
        ece += (len(bin_scores) / n) * abs(conf - acc)
    return round(float(ece), 4)


def pick_threshold(scores, labels):
    """Sweep candidate thresholds, pick the one maximizing F1 on THIS split only."""
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        m = confusion_at_threshold(scores, labels, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1


if __name__ == "__main__":
    model = CrossEncoder("outputs/akew_verifier_ckpt")

    _, val_rows, _report = build_split_examples(TRAIN_DATASETS, MODES)
    val_pairs = [[q, c] for q, c, l, _cat in val_rows]
    val_labels = [l for q, c, l, _cat in val_rows]
    val_scores = model.predict(val_pairs, convert_to_numpy=True, show_progress_bar=False)
    val_scores = 1 / (1 + np.exp(-val_scores))  # logit -> [0,1]

    thresh, val_f1_at_thresh = pick_threshold(val_scores, val_labels)
    auroc, auprc = auroc_auprc(val_scores, val_labels)
    ece = expected_calibration_error(val_scores, val_labels)
    conf = confusion_at_threshold(val_scores, val_labels, thresh)

    val_report = {"split": "val (CounterFact+WikiUpdate, subject-disjoint, calibration-selected threshold)",
                  "n": len(val_rows), "threshold": round(thresh, 3), "auroc": auroc, "auprc": auprc,
                  "ece": ece, **conf}
    print("<<<VAL_JSON>>>")
    print(json.dumps(val_report))
    print("<<<END>>>")

    # --- zero-shot: MQuAKE-CF, never touched during training/threshold selection ---
    zs_rows = []
    for mode in MODES:
        cards, golds, _groups = load_akew("MQuAKE-CF", mode)
        zs_rows += assemble_examples(cards, golds, "MQuAKE-CF", seed=0)
    zs_pairs = [[q, c] for q, c, l, _cat in zs_rows]
    zs_labels = [l for q, c, l, _cat in zs_rows]
    zs_scores = model.predict(zs_pairs, convert_to_numpy=True, show_progress_bar=False)
    zs_scores = 1 / (1 + np.exp(-zs_scores))

    zs_auroc, zs_auprc = auroc_auprc(zs_scores, zs_labels)
    zs_ece = expected_calibration_error(zs_scores, zs_labels)
    zs_conf = confusion_at_threshold(zs_scores, zs_labels, thresh)   # SAME threshold, no retuning

    zs_report = {"split": "MQuAKE-CF zero-shot (held out entirely, threshold from val NOT retuned)",
                "n": len(zs_rows), "threshold_used": round(thresh, 3), "auroc": zs_auroc, "auprc": zs_auprc,
                "ece": zs_ece, **zs_conf}
    print("<<<ZEROSHOT_JSON>>>")
    print(json.dumps(zs_report))
    print("<<<END>>>")
