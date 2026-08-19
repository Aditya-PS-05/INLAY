"""
Subject-disjoint train/calibration/test splitting for the scope-verifier
(brief section 9): split by SUBJECT, not by record, so no subject's facts
appear in more than one split -- otherwise the verifier could learn
subject-specific surface patterns rather than genuine specificity
discrimination, and validation/test numbers would be optimistic.
"""
import random
from collections import defaultdict


def subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0):
    """Returns (train_cards, val_cards, test_cards). Splits the SET OF
    SUBJECTS, then assigns every card of a given subject entirely to one
    split -- a subject never crosses a split boundary."""
    by_subject = defaultdict(list)
    for c in cards:
        by_subject[c.subject].append(c)
    subjects = sorted(by_subject.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    train = [c for s in train_subj for c in by_subject[s]]
    val = [c for s in val_subj for c in by_subject[s]]
    test = [c for s in test_subj for c in by_subject[s]]
    return train, val, test


def assert_subject_disjoint(*card_groups):
    """Programmatic leakage check: no subject appears in more than one group."""
    seen = {}
    violations = []
    for gi, group in enumerate(card_groups):
        for c in group:
            if c.subject in seen and seen[c.subject] != gi:
                violations.append((c.subject, seen[c.subject], gi))
            seen[c.subject] = gi
    return violations
