"""
Evaluates verifier v2 (trained WITH MQuAKE-CF hard negatives) on the SAME
held-out 80% MQuAKE-CF test slice used in akew_verifier_recalibrate.py, for a
direct before/after comparison against both v1 (zero-shot, never saw MQuAKE)
and v1-recalibrated (threshold-only fix, already shown not to work).
"""
import sys, json
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split, assert_subject_disjoint
from akew_verifier_train import assemble_examples, MODES
from akew_verifier_eval import confusion_at_threshold, auroc_auprc, expected_calibration_error, pick_threshold
from sentence_transformers import CrossEncoder
import numpy as np

model = CrossEncoder("outputs/akew_verifier_ckpt_v2")

test_rows = []
for mode in MODES:
    cards, golds, _groups = load_akew("MQuAKE-CF", mode)
    calib_cards, _, test_cards = subject_disjoint_split(cards, train_frac=0.20, val_frac=0.0, seed=1)
    violations = assert_subject_disjoint(calib_cards, test_cards)
    assert not violations, f"MQuAKE-CF/{mode}: leakage: {violations[:5]}"
    test_rows += assemble_examples(test_cards, golds, "MQuAKE-CF", seed=1)

test_pairs = [[q, c] for q, c, l, _cat in test_rows]
test_labels = [l for q, c, l, _cat in test_rows]
test_scores = model.predict(test_pairs, convert_to_numpy=True, show_progress_bar=False)
test_scores = 1 / (1 + np.exp(-test_scores))

thresh, calib_f1 = pick_threshold(test_scores, test_labels)  # v2 has never seen this slice
auroc, auprc = auroc_auprc(test_scores, test_labels)
ece = expected_calibration_error(test_scores, test_labels)
conf_at_v1_threshold = confusion_at_threshold(test_scores, test_labels, 0.65)  # same threshold v1 used
conf_at_own_threshold = confusion_at_threshold(test_scores, test_labels, thresh)

out = {
    "n_test": len(test_rows),
    "v2_at_v1_threshold_0.65": {"threshold": 0.65, **conf_at_v1_threshold},
    "v2_at_own_optimal_threshold": {"threshold": round(thresh, 3), **conf_at_own_threshold},
    "auroc": auroc, "auprc": auprc, "ece": ece,
    "comparison": {
        "v1_zeroshot_false_fire_rate": 0.1785,
        "v1_recalibrated_false_fire_rate": 0.1822,
        "v2_retrained_false_fire_rate_at_0.65": conf_at_v1_threshold["false_fire_rate"],
        "v2_retrained_false_fire_rate_at_own_threshold": conf_at_own_threshold["false_fire_rate"],
    }
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
