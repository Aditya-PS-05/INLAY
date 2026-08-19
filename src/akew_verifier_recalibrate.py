"""
Tests whether the verifier's MQuAKE-CF miscalibration (false-fire 1.5% -> 17.85%
under the CounterFact+WikiUpdate-calibrated threshold) is fixable cheaply, via
per-dataset threshold recalibration on a SMALL held-out slice of MQuAKE-CF,
rather than retraining the verifier itself. If a small calibration slice
recovers most of the gap, the fix is "recalibrate per deployment domain" (cheap,
standard practice); if not, the verifier itself needs MQuAKE-CF-shaped hard
negatives in its training mix (expensive, a retrain).

Splits MQuAKE-CF's own rows (never touched during original training) into a
small calibration slice (20%) and a held-out test slice (80%), subject-
disjoint at the CARD level so no group's cards leak between them. Compares:
  (a) the ORIGINAL threshold (calibrated on CounterFact+WikiUpdate only)
  (b) a MQuAKE-CF-recalibrated threshold (F1-maximized on the 20% slice only)
both evaluated on the SAME held-out 80% test slice, for a fair before/after.
"""
import sys, json
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split, assert_subject_disjoint
from akew_verifier_train import assemble_examples, MODES
from akew_verifier_eval import confusion_at_threshold, auroc_auprc, expected_calibration_error, pick_threshold
from sentence_transformers import CrossEncoder
import numpy as np

ORIGINAL_THRESHOLD = 0.65  # from outputs/akew_verifier_results.md

model = CrossEncoder("outputs/akew_verifier_ckpt")

calib_rows, test_rows = [], []
for mode in MODES:
    cards, golds, _groups = load_akew("MQuAKE-CF", mode)
    calib_cards, _, test_cards = subject_disjoint_split(cards, train_frac=0.20, val_frac=0.0, seed=1)
    violations = assert_subject_disjoint(calib_cards, test_cards)
    assert not violations, f"MQuAKE-CF/{mode}: leakage between calib/test slices: {violations[:5]}"
    calib_rows += assemble_examples(calib_cards, golds, "MQuAKE-CF", seed=1)
    test_rows += assemble_examples(test_cards, golds, "MQuAKE-CF", seed=1)

print(f"calibration slice: {len(calib_rows)} rows | held-out test slice: {len(test_rows)} rows", file=sys.stderr)

calib_pairs = [[q, c] for q, c, l, _cat in calib_rows]
calib_labels = [l for q, c, l, _cat in calib_rows]
calib_scores = model.predict(calib_pairs, convert_to_numpy=True, show_progress_bar=False)
calib_scores = 1 / (1 + np.exp(-calib_scores))
new_threshold, calib_f1 = pick_threshold(calib_scores, calib_labels)

test_pairs = [[q, c] for q, c, l, _cat in test_rows]
test_labels = [l for q, c, l, _cat in test_rows]
test_scores = model.predict(test_pairs, convert_to_numpy=True, show_progress_bar=False)
test_scores = 1 / (1 + np.exp(-test_scores))

auroc, auprc = auroc_auprc(test_scores, test_labels)
ece_orig = expected_calibration_error(test_scores, test_labels)

before = confusion_at_threshold(test_scores, test_labels, ORIGINAL_THRESHOLD)
after = confusion_at_threshold(test_scores, test_labels, new_threshold)

out = {
    "n_calib": len(calib_rows), "n_test": len(test_rows),
    "recalibrated_threshold": round(new_threshold, 3), "calib_f1_at_selection": round(calib_f1, 4),
    "test_auroc": auroc, "test_auprc": auprc, "test_ece": ece_orig,
    "before_recalibration": {"threshold": ORIGINAL_THRESHOLD, **before},
    "after_recalibration": {"threshold": round(new_threshold, 3), **after},
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
