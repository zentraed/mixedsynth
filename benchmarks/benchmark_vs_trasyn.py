"""
MixedSynth vs Trasyn: fair head-to-head at the same T budget.

For each (epsilon, U3 gate):
  1. Look up Trasyn's T count T_t and error tr_dn2 from the pre-computed cache
     (run harvest_trasyn.py once to populate it).
  2. Run optimize_u3_hardcap with t_total = T_t (same budget, not 2x).
  3. Report the full comparison table.

Columns
-------
  eps_t   : Trasyn target precision
  gate    : gate index (0-N_GATES-1)
  T_t     : Trasyn's actual T count for this (eps, gate)
  T_eps   : T-count of our best composite (may be < T_t)
  tr_dn2  : Trasyn dn/2
  eps_bst : our best single composite dn/2 at T ≤ T_t
  our_dn2 : our mixed channel dn/2
  vs_bst  : eps_bst / tr_dn2   (<1 = our single beats Trasyn at equal T)
  vs_tr   : our_dn2 / tr_dn2  (<1 = our channel beats Trasyn at equal T)
  gain    : eps_bst / our_dn2  (mixing improvement over our own best single)
  E_T     : expected total T per shot = Σ p_i T_i  (≤ T_t by construction)
  maxT    : largest composite T-count carrying weight  (≤ T_t, hard constraint)
  hdroom  : our_dn2 / floor2  (1 = at incoherent floor)

Run with:
    conda run -n gridsynth python benchmarks/benchmark_vs_trasyn.py
"""

import json
import math
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixedsynth.u3 import optimize_u3_hardcap

# ---------------------------------------------------------------------------
# Configuration — keep in sync with harvest_trasyn.py
# ---------------------------------------------------------------------------

N_GATES        = 10
SEED           = 42
DPS            = 60
N_KEEP         = 5
N_KEEP_TOTAL   = 10
EPSILONS       = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

CACHE_FILE   = os.path.join(os.path.dirname(__file__), "trasyn_cache.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "benchmark_vs_trasyn_results.txt")


def _cache_key(theta, phi, lam, eps_t):
    return "%.10f,%.10f,%.10f,%.2e" % (theta, phi, lam, eps_t)


# ---------------------------------------------------------------------------
# Tee stdout → file
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path, orig):
        self._f = open(path, "w", buffering=1)
        self._orig = orig

    def write(self, data):
        self._f.write(data); self._f.flush()
        self._orig.write(data); self._orig.flush()

    def flush(self):
        self._f.flush(); self._orig.flush()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    cache_path = os.path.normpath(CACHE_FILE)
    if not os.path.exists(cache_path):
        print("ERROR: cache not found at %s" % cache_path)
        print("Run harvest_trasyn.py first.")
        sys.exit(1)
    with open(cache_path) as f:
        cache = json.load(f)
    print("Loaded %d Trasyn cache entries." % len(cache))
    print()

    rng = np.random.default_rng(SEED)
    gates = [
        (rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi))
        for _ in range(N_GATES)
    ]

    print("MixedSynth vs Trasyn — same T budget (1×T_t, not 2×)")
    print("=" * 70)
    print("Generator picture: Frank-Wolfe + LP on composite R³ vectors")
    print("T=0 (Clifford) and T=1 (C·T) members included")
    print("%d random U3 gates, seed=%d, dps=%d" % (N_GATES, SEED, DPS))
    print()
    print("Columns")
    print("  eps_t   : Trasyn target precision")
    print("  gate    : gate index")
    print("  T_t     : Trasyn's actual T count")
    print("  T_eps   : T-count of our best single composite (≤ T_t)")
    print("  tr_dn2  : Trasyn dn/2")
    print("  eps_bst : our best single composite dn/2")
    print("  our_dn2 : our mixed channel dn/2")
    print("  vs_bst  : eps_bst / tr_dn2  (>1 = our single is worse than Trasyn)")
    print("  vs_tr   : our_dn2 / tr_dn2  (<1 = our channel beats Trasyn at equal T)")
    print("  gain    : eps_bst / our_dn2  (mixing improvement)")
    print("  E_T     : expected T per shot (≤ T_t)")
    print("  maxT    : max T of active composites (≤ T_t, hard constraint)")
    print("  hdroom  : our_dn2 / floor2  (1 = at incoherent floor)")
    print()

    hdr = ("  %7s  %4s  %4s  %5s  %9s  %9s  %9s  %7s  %7s  %7s  %6s  %4s  %6s"
           % ("eps_t", "gate", "T_t", "T_eps",
              "tr_dn2", "eps_bst", "our_dn2",
              "vs_bst", "vs_tr", "gain", "E_T", "maxT", "hdroom"))

    for eps_t in EPSILONS:
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for i, (theta, phi, lam) in enumerate(gates):
            key = _cache_key(theta, phi, lam, eps_t)
            if key not in cache:
                print("  %7.5f  %4d  (no cache entry — run harvest_trasyn.py)" % (eps_t, i))
                sys.stdout.flush()
                continue

            entry = cache[key]
            T_t    = entry["T_t"]
            tr_dn2 = entry["tr_dn2"]

            if T_t == 0:
                print("  %7.5f  %4d  T_t=0 skip" % (eps_t, i))
                sys.stdout.flush()
                continue

            stride = max(1, T_t // 30)
            result = optimize_u3_hardcap(
                theta, phi, lam, t_total=T_t,
                n_keep=N_KEEP, n_keep_total=N_KEEP_TOTAL,
                stride=stride, dps=DPS,
                enclosed_tol_ratio=0.1, t_min=0,
            )
            if result is None:
                print("  %7.5f  %4d  %4d  (no channel)" % (eps_t, i, T_t))
                sys.stdout.flush()
                continue

            # E[T] and maxT from the active support
            p       = np.array(result["p"])
            t_cts   = np.array(result["t_counts"])
            support = p > 1e-4
            E_T     = float(np.dot(p, t_cts))
            maxT    = int(np.max(t_cts[support])) if support.any() else T_t

            assert maxT <= T_t, "maxT=%d > T_t=%d — constraint violated!" % (maxT, T_t)

            vs_bst = result["eps"] / tr_dn2 if tr_dn2 > 0 else float("inf")
            vs_tr  = result["dn2"] / tr_dn2 if tr_dn2 > 0 else float("inf")

            print(
                "  %7.5f  %4d  %4d  %5d  %9.2e  %9.2e  %9.2e  %7.4f  %7.4f  %7.2f  %6.1f  %4d  %6.2f"
                % (eps_t, i, T_t, result["t_eps"],
                   tr_dn2, result["eps"], result["dn2"],
                   vs_bst, vs_tr, result["gain"],
                   E_T, maxT, result["headroom"])
            )
            sys.stdout.flush()

        print()


if __name__ == "__main__":
    sys.stdout = _Tee(os.path.normpath(RESULTS_FILE), sys.__stdout__)
    t0 = _time.time()
    run_benchmark()
    print("Total time: %.1f s" % (_time.time() - t0))
