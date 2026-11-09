"""
Mixed-channel benchmark using the hard-constraint generator-picture optimizer.

Enforces T_1 + T_2 + T_3 ≤ T_total (hard total T budget) via optimize_u3_hardcap.
Unlike the equal-split mode (t_total//3 per Rz), this allows flexible per-channel
allocation and supports T=0 (Clifford) and T=1 allocations via t_min=0.

The final mixture has n_nz ≤ 4 composite U3 members (Carathéodory in R³),
vs. ≤ 64 product members in the equal-split mode.

Columns:
  gate    : gate index (0-N_GATES-1)
  n_mem   : composite pool size (total selected composite U3 members)
  n_nz    : active-weight composites (≤ 4 by Carathéodory)
  dn2     : dn2_moment = |M1|/2 + tr(C)/4 (mpmath, primary metric)
  m1      : |M1| — composite first moment (should be ≈ 0 when enclosed)
  hull    : origin-to-hull dist for composite pool (0 = enclosed)
  floor2  : tr(C)/4 — achievable incoherent floor
  eps     : best single composite in pool (may use T < T_total)
  T_eps   : total T-count of that best composite
  eps_m   : matched-cost baseline — best single composite at T ≤ round(E[T])
  gain    : eps / dn2  (grows as ~1/eps with precision)
  gain_m  : eps_m / dn2  (fair comparison at equal expected T)
  hdroom  : dn2 / floor2  (1=at floor; >>1=coherent-limited)
  E[T]    : expected total T per shot (≤ T_total by construction)
  t_sw    : harvest time (s)
  t_opt   : optimize + moment time (s)

Cross-check (enabled for SDP_GATE_IDX gates): the actual optimized composite
channel is evaluated via Choi trace distance (channel_metrics_mp).  Since the
composite U3 members are already composed (not a product ensemble), the cross-check
is a direct call: channel_metrics_mp(U_list[support], w_list, target_u3, dps).
For a unitary target on a qubit, Choi trace distance equals diamond norm / 2.

Run with:
    conda run -n gridsynth python benchmarks/benchmark.py
"""

import sys
import os
import time as _time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixedsynth.u3 import optimize_u3_hardcap
from mixedsynth.harvest import rz_angles_for_u3, rz_exact_mp
from mixedsynth.genlist import channel_metrics_mp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DPS            = 60
T_BUDGETS      = [1, 5, 10, 20, 25, 35, 50, 100, 150, 200]   # hard total T-gate budgets (T_1+T_2+T_3 ≤ T_total)
N_GATES        = 10
N_KEEP         = 5      # candidates per T-level per Rz channel
N_KEEP_TOTAL   = 10     # composite U3 members kept per total-T level
N_SWEEP_LEVELS = 30     # stride = max(1, t_total // N_SWEEP_LEVELS)
T_MIN_RZ       = 0      # allow T=0 (Clifford) and T=1 allocations
SEED           = 42

# Cross-check: gate indices to validate via Choi composition.
# Each validated gate adds ~0.1s (4 mpmath composites, not 64).
SDP_GATE_IDX   = list(range(10))

# Robustness sweep: seeds 1-10, subset of T budgets to keep runtime reasonable.
ROBUSTNESS_SEEDS    = list(range(1, 11))
ROBUSTNESS_BUDGETS  = [10, 25, 50, 100]

RESULTS_FILE   = os.path.join(os.path.dirname(__file__), "results", "benchmark_mixedsynth_results.txt")

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
# Cross-check helper (hardcap version)
# ---------------------------------------------------------------------------

def _choi_cross_check(theta, phi, lam, result, dps=DPS):
    """
    Validate the actual optimized composite U3 channel via Choi trace distance.

    Since optimize_u3_hardcap stores the composed U3 matrices directly in
    result["U_list"], the cross-check is a single channel_metrics_mp call on
    the support members — no product-ensemble expansion needed.

    Returns dn2_cross (float) or None on failure.
    """
    try:
        import mpmath as mp

        U_list = result["U_list"]
        p      = result["p"]
        w_mp   = result.get("w_mp")

        support = [i for i in range(len(p)) if p[i] > 1e-4]
        if not support:
            return None

        w_list = []
        for i in support:
            if w_mp is not None:
                w_list.append(w_mp[i])
            else:
                w_list.append(mp.mpf(str(float(p[i]))))
        U_support = [U_list[i] for i in support]

        angles = result["angles"]
        with mp.workdps(dps + 10):
            H_mp = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)
            rz_t = [rz_exact_mp(a, dps=dps) for a in angles]
            target_u3 = rz_t[2] * H_mp * rz_t[1] * H_mp * rz_t[0]

        dn_cross, _, _ = channel_metrics_mp(U_support, w_list, target_u3, dps=dps)
        return float(dn_cross) / 2

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Matched-cost baseline helper
# ---------------------------------------------------------------------------

def _matched_cost_eps(result):
    """
    Best single composite member with total T ≤ round(E[T]).

    Removes the 'the mixture spends more T' objection: compares dn2 against
    the best deterministic composite at the same expected T cost.
    """
    E_T = result["E_T"]
    T_target = round(E_T)
    t_counts = np.array(result["t_counts"])
    mags     = np.array(result["mags_f64"])

    mask = t_counts <= T_target
    if mask.any():
        return float(np.min(mags[mask])) / 2
    # fallback: pool has no member at or below T_target
    return result["eps"]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(seed=SEED, t_budgets=T_BUDGETS, n_gates=N_GATES,
                  n_keep=N_KEEP, n_keep_total=N_KEEP_TOTAL, dps=DPS,
                  verbose=True):
    """
    Run the hard-constraint benchmark for one seed.
    Returns nested dict results[t_budget][gate_idx].
    """
    rng = np.random.default_rng(seed)
    gates = [
        (rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi))
        for _ in range(n_gates)
    ]

    if verbose:
        print("MixedSynth Hard-Constraint Benchmark (T_1+T_2+T_3 ≤ T_total)")
        print("=" * 66)
        print("Generator picture: composite R³ vectors, Frank-Wolfe + LP")
        print("Carathéodory: n_nz ≤ 4 composite U3 members in the mixture")
        print("%d random U3 gates, seed=%d, dps=%d, t_min=%d"
              % (n_gates, seed, dps, T_MIN_RZ))
        print()
        print("Column legend")
        print("  gate    : gate index (0-%d)" % (n_gates - 1))
        print("  n_mem   : composite pool size (all selected composites)")
        print("  n_nz    : active-weight composites (Carathéodory ≤ 4)")
        print("  dn2     : dn2_moment = |M1|/2 + tr(C)/4 (mpmath, primary metric)")
        print("  m1      : |M1| — composite first moment (≈ 0 when enclosed)")
        print("  hull    : origin-to-hull dist (0 = enclosed)")
        print("  floor2  : tr(C)/4 — achievable incoherent floor")
        print("  eps     : best single composite in pool (deterministic baseline)")
        print("  T_eps   : total T-count of that best composite (≤ T_total)")
        print("  eps_m   : matched-cost baseline — best single at T ≤ round(E[T])")
        print("  gain    : eps / dn2  (~1/eps — the ε² law expressed as a ratio)")
        print("  gain_m  : eps_m / dn2  (fair comparison at equal expected T)")
        print("  hdroom  : dn2 / floor2  (1=at floor; >>1=coherent-limited)")
        print("  E[T]    : expected total T per shot (≤ T_total by construction)")
        print("  t_sw    : harvest time (s)")
        print("  t_opt   : optimize + moment time (s)")
        print()

    all_results = {}

    for t_total in t_budgets:
        stride = max(1, t_total // N_SWEEP_LEVELS)
        # Approximate best composite eps at T_total: 3 channels each ~T_total/3 T-gates
        eps_scale = 3.0 * 2.0 ** (-(t_total // 3) / 3.0)

        if verbose:
            print("T_total=%d  stride=%d  n_keep=%d  n_keep_total=%d  eps_scale~%.2e"
                  % (t_total, stride, n_keep, n_keep_total, eps_scale))

            hdr = ("  %4s  %5s  %4s  %9s  %9s  %9s  %9s  %9s  %5s  %9s  %9s  %9s  %6s  %6s  %5s  %5s"
                   % ("gate", "n_mem", "n_nz", "dn2", "m1", "hull", "floor2",
                      "eps", "T_eps", "eps_m", "gain", "gain_m", "hdroom", "E[T]", "t_sw", "t_opt"))
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

        budget_results = {}
        for i, (theta, phi, lam) in enumerate(gates):
            result = optimize_u3_hardcap(
                theta, phi, lam, t_total=t_total,
                n_keep=n_keep, n_keep_total=n_keep_total,
                stride=stride, dps=dps,
                enclosed_tol_ratio=0.1, t_min=T_MIN_RZ,
            )
            budget_results[i] = result
            if result is None:
                if verbose:
                    print("  %4d  (no channel)" % i)
                continue

            eps_m  = _matched_cost_eps(result)
            gain_m = (eps_m / result["dn2"]) if result["dn2"] > 0 else float("inf")

            # T=0/T=1 diagnostic: count active support members with T ≤ 1
            p_arr   = np.array(result["p"])
            t_arr   = np.array(result["t_counts"])
            support = p_arr > 1e-4
            n_low_t = int(np.sum((t_arr <= 1) & support))
            low_t_tag = ("  [T≤1 in support: %d/%d]" % (n_low_t, int(support.sum()))
                         if n_low_t > 0 else "")

            cross_tag = ""
            if SDP_GATE_IDX and i in SDP_GATE_IDX:
                dn2_cross = _choi_cross_check(theta, phi, lam, result, dps=dps)
                if dn2_cross is not None:
                    ratio = dn2_cross / result["dn2"] if result["dn2"] > 0 else float("inf")
                    cross_tag = "  [cross: dn2=%.2e  ratio=%.3f]" % (dn2_cross, ratio)

            if verbose:
                print(
                    "  %4d  %5d  %4d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %5d  %9.2e  %9.2e  %9.2e  %6.2f  %6.1f  %5.1f  %5.2f%s%s"
                    % (
                        i,
                        result["n_mem"],
                        result["n_nz"],
                        result["dn2"],
                        result["m1"],
                        result["hull_dist"],
                        result["floor2"],
                        result["eps"],
                        result["t_eps"],
                        eps_m,
                        result["gain"],
                        gain_m,
                        result["headroom"],
                        result["E_T"],
                        result["t_sweep"],
                        result["t_opt"],
                        low_t_tag,
                        cross_tag,
                    )
                )

        all_results[t_total] = budget_results
        if verbose:
            print()

    return all_results


# ---------------------------------------------------------------------------
# Robustness sweep across multiple seeds
# ---------------------------------------------------------------------------

def run_robustness_sweep(seeds, t_budgets=T_BUDGETS, n_gates=N_GATES,
                         n_keep=N_KEEP, n_keep_total=N_KEEP_TOTAL, dps=DPS):
    """
    Run the benchmark across multiple seeds and report per-budget statistics.
    """
    print("Robustness sweep: seeds=%s  budgets=%s" % (seeds, t_budgets))
    print()

    for t_total in t_budgets:
        stride = max(1, t_total // N_SWEEP_LEVELS)
        gains, hdrooms, e_ts = [], [], []
        n_unenclosed = 0
        n_low_support = 0
        n_low_t_support = 0
        total = 0

        for seed in seeds:
            results = run_benchmark(seed=seed, t_budgets=[t_total],
                                    n_keep=n_keep, n_keep_total=n_keep_total,
                                    dps=dps, verbose=False)
            for gate_idx, result in results.get(t_total, {}).items():
                if result is None:
                    continue
                total += 1
                gains.append(result["gain"])
                hdrooms.append(result["headroom"])
                e_ts.append(result["E_T"])
                if not result["enclosed"]:
                    n_unenclosed += 1
                if result["n_nz"] < 4:
                    n_low_support += 1
                # Count gates where any T≤1 composite carries weight
                p_arr = np.array(result["p"])
                t_arr = np.array(result["t_counts"])
                if np.any((t_arr <= 1) & (p_arr > 1e-4)):
                    n_low_t_support += 1

        if not gains:
            print("T_total=%d: no results" % t_total)
            continue

        gains   = np.array(gains)
        hdrooms = np.array(hdrooms)
        e_ts    = np.array(e_ts)
        print("T_total=%d  stride=%d  n_seeds=%d  n_gates=%d"
              % (t_total, stride, len(seeds), total))
        print("  gain:    median=%.2e  min=%.2e  max=%.2e"
              % (float(np.median(gains)), float(np.min(gains)), float(np.max(gains))))
        print("  hdroom:  median=%.3f  worst=%.3f"
              % (float(np.median(hdrooms)), float(np.max(hdrooms))))
        print("  E[T]:    median=%.1f  min=%.1f  max=%.1f  (budget=%d)"
              % (float(np.median(e_ts)), float(np.min(e_ts)), float(np.max(e_ts)), t_total))
        print("  enclosed=False:     %d/%d (%.0f%%)" % (n_unenclosed, total, 100*n_unenclosed/total))
        print("  n_nz<4:             %d/%d (%.0f%%)" % (n_low_support, total, 100*n_low_support/total))
        print("  T≤1 in support:     %d/%d (%.0f%%)" % (n_low_t_support, total, 100*n_low_t_support/total))
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = _Tee(os.path.normpath(RESULTS_FILE), sys.__stdout__)
    t0_total = _time.time()
    run_benchmark()
    if ROBUSTNESS_SEEDS:
        print("=" * 60)
        run_robustness_sweep(ROBUSTNESS_SEEDS, t_budgets=ROBUSTNESS_BUDGETS)
    print("Total time: %.1f s" % (_time.time() - t0_total))
