"""
One-time Trasyn harvester: pre-computes and caches Trasyn outputs for all
(gate, eps_t) combinations used by benchmark_vs_trasyn.py.

The cache is a JSON file keyed by "(theta, phi, lam, eps_t)" strings, storing
{T_t, tr_dn2, sequence}.  benchmark_vs_trasyn.py reads the cache and never
calls Trasyn directly — this decouples the slow stochastic search from the
fast deterministic optimizer loop and pins results for reproducibility.

Run once to populate:
    conda run -n gridsynth python benchmarks/harvest_trasyn.py

Re-run to fill any missing keys (idempotent — existing entries are skipped).
"""

import json
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import trasyn

# ---------------------------------------------------------------------------
# Configuration — keep in sync with benchmark_vs_trasyn.py
# ---------------------------------------------------------------------------

N_GATES         = 10
SEED            = 42
TRASYN_NSAMPLES = 10000
EPSILONS        = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "trasyn_cache.json"
)


def _cache_key(theta, phi, lam, eps_t):
    return "%.10f,%.10f,%.10f,%.2e" % (theta, phi, lam, eps_t)


def _run_trasyn(theta, phi, lam, eps_t, t_max, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            seq, mat, err = trasyn.synthesize(
                [theta, phi, lam],
                nonclifford_budget=t_max,
                error_threshold=eps_t,
                num_samples=TRASYN_NSAMPLES,
                rng=seed,
            )
        except (ValueError, RuntimeError) as e:
            return None, str(e)
    return {"T_t": seq.count("t"), "tr_dn2": float(err), "sequence": seq}, None


def main():
    # Load or create cache
    cache_path = os.path.normpath(CACHE_FILE)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print("Loaded %d existing cache entries from %s" % (len(cache), cache_path))
    else:
        cache = {}
        print("Starting fresh cache at %s" % cache_path)

    rng = np.random.default_rng(SEED)
    gates = [
        (rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi))
        for _ in range(N_GATES)
    ]

    n_total   = len(EPSILONS) * N_GATES
    n_done    = 0
    n_skipped = 0
    n_failed  = 0
    t0 = time.time()

    for eps_t in EPSILONS:
        t_max = 3 * max(4, math.ceil(3 * math.log2(1.0 / eps_t)))
        for i, (theta, phi, lam) in enumerate(gates):
            key = _cache_key(theta, phi, lam, eps_t)
            if key in cache:
                n_skipped += 1
                continue

            gate_seed = SEED + i
            t_gate = time.time()
            entry, err = _run_trasyn(theta, phi, lam, eps_t, t_max, gate_seed)
            elapsed = time.time() - t_gate

            if entry is None:
                print("  eps=%.4f  gate=%d  FAILED: %s" % (eps_t, i, err))
                n_failed += 1
            else:
                cache[key] = entry
                n_done += 1
                print("  eps=%.4f  gate=%d  T_t=%d  tr_dn2=%.3e  (%.1fs)"
                      % (eps_t, i, entry["T_t"], entry["tr_dn2"], elapsed))

            # Write after every entry so a kill doesn't lose work
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)

            sys.stdout.flush()

    elapsed_total = time.time() - t0
    print()
    print("Done: %d new, %d skipped, %d failed  (%.1fs total)"
          % (n_done, n_skipped, n_failed, elapsed_total))
    print("Cache: %d entries at %s" % (len(cache), cache_path))


if __name__ == "__main__":
    main()
