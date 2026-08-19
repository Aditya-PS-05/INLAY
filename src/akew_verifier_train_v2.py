"""
Verifier retrain v2: adds a MQuAKE-CF hard-negative slice to the training mix,
per outputs/akew_verifier_recalibration_results.md's confirmed conclusion
(threshold recalibration doesn't work; the verifier needs MQuAKE-CF-shaped
hard negatives in training).

Uses the SAME calibration/test split as the recalibration experiment (subject-
disjoint, seed=1, 20/80): the 20% calibration slice is now folded into
TRAINING (not just threshold-picking), and the 80% test slice remains
GENUINELY held out for evaluation -- never touched by this retrain, so the
before/after comparison against the v1 verifier is still a fair zero-shot-
style test on the same held-out data the recalibration experiment used.
"""
import sys, json
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split, assert_subject_disjoint
from akew_verifier_train import assemble_examples, TRAIN_DATASETS, MODES, build_split_examples
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import torch

if __name__ == "__main__":
    print("Assembling CounterFact + WikiUpdate training data (as before)...")
    train_rows, val_rows, report = build_split_examples(TRAIN_DATASETS, MODES)

    print("Assembling MQuAKE-CF hard-negative slice (20% calib subjects, seed=1, "
          "same split as the recalibration test -- the 80% remainder stays held out)...")
    mquake_train_rows = []
    mquake_test_report = {}
    for mode in MODES:
        cards, golds, _groups = load_akew("MQuAKE-CF", mode)
        calib_cards, _, test_cards = subject_disjoint_split(cards, train_frac=0.20, val_frac=0.0, seed=1)
        violations = assert_subject_disjoint(calib_cards, test_cards)
        assert not violations, f"MQuAKE-CF/{mode}: leakage: {violations[:5]}"
        rows = assemble_examples(calib_cards, golds, "MQuAKE-CF", seed=1)
        mquake_train_rows += rows
        mquake_test_report[mode] = {"n_calib_cards": len(calib_cards), "n_test_cards": len(test_cards),
                                    "n_calib_rows": len(rows)}

    train_rows = train_rows + mquake_train_rows
    print(json.dumps({"cf_wikiupdate": report, "mquake_calib": mquake_test_report}, indent=2))
    print(f"\ntotal train rows (CF+WikiUpdate+MQuAKE-calib): {len(train_rows)} "
          f"(pos={sum(1 for r in train_rows if r[2]==1)}, neg={sum(1 for r in train_rows if r[2]==0)})")
    print(f"total val rows: {len(val_rows)} (pos={sum(1 for r in val_rows if r[2]==1)}, "
          f"neg={sum(1 for r in val_rows if r[2]==0)})")
    print(f"MQuAKE-CF hard-negative slice added to training: {len(mquake_train_rows)} rows "
          f"(pos={sum(1 for r in mquake_train_rows if r[2]==1)}, neg={sum(1 for r in mquake_train_rows if r[2]==0)})")

    train_examples = [InputExample(texts=[q, c], label=float(l)) for q, c, l, _cat in train_rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1, device=device)

    train_dl = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=32)
    val_evaluator = CEBinaryClassificationEvaluator(
        [[q, c] for q, c, l, _cat in val_rows],
        [l for q, c, l, _cat in val_rows],
        name="akew_val_v2",
    )

    print("\nTraining v2 (with MQuAKE-CF hard negatives)...")
    model.fit(train_dataloader=train_dl, evaluator=val_evaluator, epochs=2,
              warmup_steps=int(0.1 * len(train_dl)), output_path="outputs/akew_verifier_ckpt_v2",
              show_progress_bar=False)

    model.save("outputs/akew_verifier_ckpt_v2")
    import os
    saved_files = os.listdir("outputs/akew_verifier_ckpt_v2")
    assert "config.json" in saved_files or any(f.endswith(".safetensors") for f in saved_files), \
        f"model.save() did not produce expected files: {saved_files}"

    print(f"\nTraining v2 complete. Checkpoint saved to outputs/akew_verifier_ckpt_v2: {saved_files}")
    print("<<<TRAIN_V2_DONE>>>")
