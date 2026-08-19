"""Regression test for the product-key capacity fix (pk_memory.py).
Pure tensor ops, no model needed. Proves: with the OLD diagonal-only allocator,
writing more than n_sub facts would raise "memory full" well before N=n_sub**2
slots were used; the FIXED allocator should reach the full n_sub**2 capacity.
"""
import torch, sys
sys.path.insert(0, "src")
from pk_memory import ProductKeyMemory

DIM = 8
N_SUB = 4          # small grid: N = 16 total slots
mem = ProductKeyMemory(dim=DIM, n_slots_per_subkey=N_SUB, topk=1)

written = 0
try:
    for i in range(N_SUB * N_SUB):  # attempt to fill the FULL grid (16 slots)
        k = torch.randn(DIM)
        v = torch.randn(DIM)
        mem.write(k, v, meta={"i": i})
        written += 1
except RuntimeError as e:
    print(f"FAIL: memory full after only {written}/{N_SUB*N_SUB} writes: {e}")
    sys.exit(1)

assert written == N_SUB * N_SUB, f"expected {N_SUB*N_SUB} writable slots, got {written}"
print(f"PASS: wrote all {written} = n_sub**2 slots without 'memory full' "
      f"(old diagonal-only allocator would have failed at write #{N_SUB+1})")

# one more write past capacity must correctly raise
try:
    mem.write(torch.randn(DIM), torch.randn(DIM))
    print("FAIL: write past declared capacity did not raise")
    sys.exit(1)
except RuntimeError:
    print("PASS: write past declared capacity correctly raises RuntimeError")

# sanity: distinct (a,b) pairs actually used (not all collapsed onto one diagonal)
a_vals = {mem._split(s)[0] for s in mem._used_slots}
b_vals = {mem._split(s)[1] for s in mem._used_slots}
assert len(a_vals) == N_SUB and len(b_vals) == N_SUB, "grid not fully exercised"
print(f"PASS: all {N_SUB} rows and {N_SUB} columns of the grid were exercised "
      f"(a_vals={sorted(a_vals)}, b_vals={sorted(b_vals)})")

# clear_all resets the new allocator field correctly
mem.clear_all()
assert mem._next_write == 0 and mem.num_written() == 0
k = torch.randn(DIM); v = torch.randn(DIM)
sid = mem.write(k, v)
assert mem.num_written() == 1
print("PASS: clear_all() resets the fixed allocator (_next_write) correctly")

print("\nALL PK-CAPACITY REGRESSION CHECKS PASSED")
