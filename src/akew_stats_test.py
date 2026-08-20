"""
Self-tests for akew_stats.py, checked against hand-computable values.

Statistical helpers are the worst place for a silent bug: a transposed
discordant count or an off-by-one in a binomial tail produces a p-value that
looks entirely reasonable and is simply wrong, and the error surfaces as a
false claim in a paper rather than as a crash. Every case below has a value
that can be derived by hand and is stated in the comment, so the test is
checking arithmetic rather than restating the implementation.

Usage: python akew_stats_test.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from akew_stats import (holm_bonferroni, mcnemar_exact, paired_bootstrap_diff,
                        wilson_interval)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


print("wilson_interval:")
# k=n=63 (the MQuAKE-CF structured 100% cell). By hand, z=1.96:
#   denom  = 1 + 1.96^2/63           = 1.060978
#   centre = (1 + 1.96^2/126)/denom  = 1.030489/1.060978 = 0.971264
#   half   = (1.96/denom)*sqrt(1.96^2/(4*63^2)) = 1.847258*0.0155556 = 0.028735
#   -> lo ~ 0.94253, hi capped at 1.0
lo, hi = wilson_interval(63, 63)
check("100% on n=63 -> lo ~0.9425", abs(lo - 0.94253) < 1e-3, f"got {lo}")
# For p=1 the Wilson upper bound is mathematically exactly 1.0, but centre and
# half are computed through a sqrt and a division, so the sum lands a few ulps
# short (0.9999999999999999). Tolerance rather than equality: this is float
# representation, not a logic error, and it rounds to 1.0 in any report. The
# check that actually matters is the one below -- never EXCEEDING 1.0.
check("100% on n=63 -> hi is 1.0 within float tolerance", abs(hi - 1.0) < 1e-12,
      f"got {hi}")
check("interval never exceeds 1.0 (the reason Wilson was chosen)", hi <= 1.0)

lo0, hi0 = wilson_interval(0, 20)
check("0% -> lo clamped at 0.0", lo0 == 0.0, f"got {lo0}")
check("0% -> hi > 0", hi0 > 0.0)

lo5, hi5 = wilson_interval(50, 100)
check("50% on n=100 is roughly centred", abs((lo5 + hi5) / 2 - 0.5) < 0.01,
      f"got [{lo5},{hi5}]")
check("n=0 returns (None, None)", wilson_interval(0, 0) == (None, None))

print("mcnemar_exact:")
# Perfectly concordant: no discordant pairs carry information -> p = 1.0
a = [1, 1, 0, 0]
check("identical vectors -> p=1.0", mcnemar_exact(a, a)["p_value"] == 1.0)

# 5 discordant, all favouring b. Two-sided exact p = 2 * C(5,0)/2^5 = 2/32
b_all = {"a": [0, 0, 0, 0, 0], "b": [1, 1, 1, 1, 1]}
r = mcnemar_exact(b_all["a"], b_all["b"])
check("5-0 discordant -> n_discordant=5", r["n_discordant"] == 5, f"got {r}")
check("5-0 discordant -> b_better=5", r["b_better"] == 5, f"got {r}")
check("5-0 discordant -> p = 2/32 = 0.0625", abs(r["p_value"] - 0.0625) < 1e-9,
      f"got {r['p_value']}")

# 10-0 discordant: p = 2 * 1/2^10 = 0.001953125
r10 = mcnemar_exact([0] * 10, [1] * 10)
check("10-0 discordant -> p = 2/1024", abs(r10["p_value"] - 2 / 1024) < 1e-9,
      f"got {r10['p_value']}")

# Direction matters: swapping arguments must swap which side won, not the p.
r_swap = mcnemar_exact([1] * 10, [0] * 10)
check("swapping args swaps the winner", r_swap["a_better"] == 10 and r_swap["b_better"] == 0,
      f"got {r_swap}")
check("swapping args leaves p unchanged (two-sided)",
      abs(r_swap["p_value"] - r10["p_value"]) < 1e-12)

# Balanced discordance -> no evidence either way -> p = 1.0
r_bal = mcnemar_exact([1, 0, 1, 0], [0, 1, 0, 1])
check("balanced 2-2 discordance -> p=1.0", abs(r_bal["p_value"] - 1.0) < 1e-9,
      f"got {r_bal['p_value']}")

# Concordant pairs must be ignored entirely: padding with all-correct pairs
# changes n but must not change the p-value.
pad_a = [0] * 5 + [1] * 50
pad_b = [1] * 5 + [1] * 50
check("concordant padding does not change p",
      abs(mcnemar_exact(pad_a, pad_b)["p_value"] - 0.0625) < 1e-9,
      f"got {mcnemar_exact(pad_a, pad_b)['p_value']}")

try:
    mcnemar_exact([1, 0], [1, 0, 1])
    check("length mismatch raises", False, "no exception")
except ValueError:
    check("length mismatch raises ValueError", True)

print("paired_bootstrap_diff:")
res = paired_bootstrap_diff([0] * 10, [1] * 10, n_boot=2000, seed=0)
check("all-b-better -> diff = 1.0", abs(res["diff"] - 1.0) < 1e-9, f"got {res}")
check("degenerate case -> CI collapses on 1.0", res["lo"] == 1.0 and res["hi"] == 1.0,
      f"got {res}")

same = paired_bootstrap_diff([1, 0, 1, 0], [1, 0, 1, 0], n_boot=2000, seed=0)
check("identical vectors -> diff 0 and CI spans 0",
      same["diff"] == 0.0 and same["lo"] <= 0 <= same["hi"], f"got {same}")

rng = np.random.default_rng(1)
x = rng.integers(0, 2, size=200).tolist()
noisy = paired_bootstrap_diff(x, x, n_boot=2000, seed=0)
check("paired resampling of identical vectors gives exactly 0 every draw",
      noisy["lo"] == 0.0 and noisy["hi"] == 0.0,
      f"got {noisy} -- nonzero here means the two conditions were resampled "
      f"INDEPENDENTLY, destroying the pairing")

print("holm_bonferroni:")
# m=3, alpha=0.05, p = [0.01, 0.04, 0.03]. Sorted: 0.01 vs 0.05/3=0.01667 ->
# reject. 0.03 vs 0.05/2=0.025 -> fail, and step-down stops there, so 0.04
# cannot be rejected even though 0.04 <= 0.05.
h = holm_bonferroni([0.01, 0.04, 0.03])
check("smallest p rejected", h[0]["reject"] is True, f"got {h[0]}")
check("0.03 fails at adjusted 0.025", h[2]["reject"] is False, f"got {h[2]}")
check("step-down blocks 0.04 despite p<alpha", h[1]["reject"] is False, f"got {h[1]}")
check("adjusted alphas descend by rank",
      abs(h[0]["adjusted_alpha"] - 0.05 / 3) < 1e-12)

h_all = holm_bonferroni([0.0001, 0.0002])
check("both tiny p-values rejected", all(x["reject"] for x in h_all), f"got {h_all}")

h_none = holm_bonferroni([0.9, 0.8])
check("both large p-values not rejected", not any(x["reject"] for x in h_none))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
