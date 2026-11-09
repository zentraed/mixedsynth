"""
Quickstart: generate Clifford+T sequences for an Rz, then synthesize a mixed
channel for a U3 gate under a hard T budget and sample circuits from it.

    python examples/quickstart.py
"""
import random

import numpy as np

from mixedsynth import harvest_rz, optimize_u3_hardcap, sample, circuit_order

random.seed(0)  # the Diophantine solver is randomized; seed for reproducibility

# ---- 1. Sequence generation ------------------------------------------------
print("Clifford+T approximations of Rz(0.7), T = 0..20:")
for t, seq, _U in harvest_rz(0.7, t_min=0, t_budget=20, n_keep=1,
                             stride=4, with_sequences=True):
    print(f"  T={t:2d}  {seq or '(identity)'}")

# ---- 2. Channel synthesis --------------------------------------------------
theta, phi, lam = 1.1, 2.3, 0.4
T_total = 30
r = optimize_u3_hardcap(theta, phi, lam, t_total=T_total)

print(f"\nU3({theta}, {phi}, {lam}) with T ≤ {T_total} per shot")
print(f"  best single sequence : dn/2 = {r['eps']:.3e}")
print(f"  mixed channel        : dn/2 = {r['dn2']:.3e}  "
      f"(gain {r['gain']:.1f}x, E[T] = {r['E_T']:.1f})")
print("  members:")
for m in r["members"]:
    print(f"    p={float(m['weight']):.4f}  T={m['t_count']:2d}  {m['sequence']}")

# ---- 3. Use it: draw a fresh circuit per shot ------------------------------
rng = np.random.default_rng(1)
shot = sample(r["members"], rng=rng)
print("\nOne shot, gates in application order:")
print("  " + " ".join(circuit_order(shot)))
