"""
Fair cross-evaluation for the hard-negative-composition ablation: the first
pass scored each variant (no_hard_negatives, specificity_only) against its
OWN validation set, built with that variant's own (easier) negative-sampling
method -- not comparable to each other or to the original full-8-category
verifier's val score. A verifier trained only on random negatives scoring
99.9% against RANDOM negatives is not evidence of quality, it's evidence the
benchmark was easy.

This scores all three checkpoints (no_hard_negatives, specificity_only,
akew_verifier_ckpt = the original full 8-category mix) against the SAME hard
val set (the full-mix verifier's own val split), for an apples-to-apples
comparison of what actually matters: performance against real, hard,
structured confusions, not against whatever each model happened to train on.
"""
import sys, json
sys.path.insert(0, "src")
from akew_verifier_train import build_split_examples, TRAIN_DATASETS, MODES
from akew_verifier_eval import confusion_at_threshold, auroc_auprc, expected_calibration_error, pick_threshold
from sentence_transformers import CrossEncoder
import numpy as np

_, hard_val_rows, _report = build_split_examples(TRAIN_DATASETS, MODES)
hard_val_pairs = [[q, c] for q, c, l, _cat in hard_val_rows]
hard_val_labels = [l for q, c, l, _cat in hard_val_rows]

checkpoints = {
    "full_8category_mix (original akew_verifier_ckpt)": "outputs/akew_verifier_ckpt",
    "no_hard_negatives_random_only": "outputs/akew_verifier_ablation_random",
    "specificity_only": "outputs/akew_verifier_ablation_specificity",
}

results = {}
for name, path in checkpoints.items():
    model = CrossEncoder(path)
    scores = model.predict(hard_val_pairs, convert_to_numpy=True, show_progress_bar=False)
    scores = 1 / (1 + np.exp(-scores))
    thresh, _ = pick_threshold(scores, hard_val_labels)
    auroc, auprc = auroc_auprc(scores, hard_val_labels)
    ece = expected_calibration_error(scores, hard_val_labels)
    conf = confusion_at_threshold(scores, hard_val_labels, thresh)
    results[name] = {"threshold": round(thresh, 3), "auroc": auroc, "auprc": auprc, "ece": ece, **conf}

out = {"n_hard_val": len(hard_val_rows), "results_on_SAME_hard_val_set": results}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
