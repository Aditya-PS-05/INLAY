"""Regression test for AKEW data-layer leakage controls (section 9 of the vNext
brief: 'Assert programmatically that forbidden gold fields are not passed to
the editor in unstructured or extracted modes').

Also guards against silent schema drift: if a future AKEW repo update changes
record counts, this fails loudly instead of quietly evaluating on a different
sample than what was validated in outputs/akew_schema_validation.md.
"""
import sys
sys.path.insert(0, "src")
from akew_data import load_akew, assert_no_gold_leakage, LeakageError, GOLD_FIELDS

EXPECTED_COUNTS = {
    ("CounterFact", "cards"): 975,
    ("MQuAKE-CF", "cards"): 436,
    ("MQuAKE-CF", "groups"): 354,
    ("WikiUpdate", "cards"): 1056,  # NOT 1067 -- the paper's number, real file differs, see akew_data.py docstring
}

failures = []

for ds in ("CounterFact", "MQuAKE-CF", "WikiUpdate"):
    for mode in ("structured", "unstructured", "extracted"):
        cards, golds, groups = load_akew(ds, mode)

        # --- count regression guard ---
        exp_cards = EXPECTED_COUNTS.get((ds, "cards"))
        if exp_cards is not None and len(cards) != exp_cards:
            failures.append(f"{ds}/{mode}: expected {exp_cards} cards, got {len(cards)}")
        if ds == "MQuAKE-CF":
            exp_groups = EXPECTED_COUNTS[(ds, "groups")]
            if len(groups) != exp_groups:
                failures.append(f"{ds}/{mode}: expected {exp_groups} groups, got {len(groups)}")

        # --- structural leakage guard: every card, not a sample ---
        for c in cards:
            try:
                assert_no_gold_leakage(c)
            except LeakageError as e:
                failures.append(f"{ds}/{mode}/{c.edit_id}: {e}")

        # --- content-presence guard: the right field is populated for the mode ---
        for c in cards:
            if mode == "structured" and not c.canonical_fact_text:
                failures.append(f"{ds}/{mode}/{c.edit_id}: structured card missing canonical_fact_text")
            if mode in ("unstructured", "extracted") and not c.raw_evidence_text:
                failures.append(f"{ds}/{mode}/{c.edit_id}: {mode} card missing raw_evidence_text")
            if mode != "structured" and c.canonical_fact_text is not None:
                failures.append(f"{ds}/{mode}/{c.edit_id}: {mode} card unexpectedly has canonical_fact_text set")

        print(f"{ds:12s} {mode:13s} OK  cards={len(cards)} golds={len(golds)} groups={len(groups)}")

# --- gold record sanity: every card has a corresponding gold entry, and gold
#     entries carry the answer info that cards must NOT carry ---
cards_s, golds_s, _ = load_akew("CounterFact", "structured")
for c in cards_s[:20]:
    g = golds_s.get(c.edit_id)
    if g is None:
        failures.append(f"CounterFact/{c.edit_id}: no gold record")
    elif not g.target_new:
        failures.append(f"CounterFact/{c.edit_id}: gold record missing target_new")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures[:30]:
        print(" -", f)
    sys.exit(1)

print(f"\nALL AKEW LEAKAGE + SCHEMA REGRESSION CHECKS PASSED "
      f"(GOLD_FIELDS guarded: {sorted(GOLD_FIELDS)})")
