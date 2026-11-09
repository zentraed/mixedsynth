"""
Copyright (c) 2024 Massachusetts Institute of Technology
SPDX-License-Identifier: BSD-2-Clause

Ensemble generation and mixed-unitary-channel optimization for Clifford+T
approximations of single-qubit rotations.

Overview
--------
Standard Clifford+T synthesis (gridsynth, pyLIQTR approxRz_direct) produces a
*single* gate sequence approximating a target rotation to a given precision ε.
This module instead constructs a *mixed unitary channel* — a probability
distribution over multiple Clifford+T sequences — and optimizes the weights to
minimize the diamond norm distance to the target channel.  The mixed channel can
achieve lower diamond norm than any single sequence drawn from the same T budget,
at the cost of being probabilistic rather than deterministic.

U3 decomposition (Eq. 1 of arXiv:2503.15843):

    U3(θ, φ, λ) = Rz(φ + 5π/2) · H · Rz(θ) · H · Rz(λ − π/2)

Metrics
-------
Three complementary quality metrics are computed for a mixed channel
Φ = Σᵢ pᵢ Uᵢ(·)Uᵢ† versus a target unitary U:

  diamond_norm  ‖Φ − U(·)U†‖_◇  ∈ [0, 2]   worst-case distinguishability
                                              (0 = identical, 2 = orthogonal)
  F_avg         average gate fidelity        ∈ [0, 1]   F_avg = (2·F_e + 1)/3
  F_e           process (entanglement) fidelity  ∈ [0, 1]   Σᵢ pᵢ|tr(U†Uᵢ)|²/4

For single-qubit synthesis, diamond_norm = 2 × operator_norm_error when the
error is small, so divide diamond_norm by 2 to compare with the ε reported by
gridsynth or Trasyn.

Typical workflow
----------------
    # 1. Generate per-Rz candidate sets spanning T=t_min..t_budget
    rz1, rz2, rz3 = approximate_u3_direct_set(theta, phi, lam, t_budget=24)

    # 2. Build all U3 combinations with T_total ≤ t_budget (one best per level)
    unitaries, t_counts = build_u3_channel_reduced(rz1, rz2, rz3, t_budget=24, target=target)

    # 3. Optimize weights to minimize diamond norm
    weights = optimize_channel_weights(unitaries, target)

    # 4. Evaluate quality
    dn, favg, fe = channel_metrics(unitaries, weights, target)
    op_norm_equiv = dn / 2   # comparable to gridsynth / Trasyn ε

Requirements
------------
Requires gmpy2 >= 2.1.2 (the 'gridsynth' conda environment).  gmpy2 2.0.8
(the 'base' environment) will raise TypeError inside pyLIQTR point_enumeration.
"""

from enum import Enum, auto
from math import gcd
from typing import List, Optional, Tuple, Union

import gmpy2
import numpy as np
from gmpy2 import mpfr

from mixedsynth._pyliqtr.exact_decomp import (
    exact_decomp_to_matrix_string,
    exact_decomp_to_qasm,
)
from mixedsynth._pyliqtr.gate_approximation import (
    corrections,
    is_reducible,
    reduce,
)
from mixedsynth._pyliqtr.point_enumeration import candidate_generator_direct
from mixedsynth._pyliqtr.rings import Z_OMEGA, Z_SQRT2
from mixedsynth._pyliqtr.solve_diophantine import solveDiophantine


class ChannelMode(Enum):
    RZ = auto()
    U3 = auto()
    ARBITRARY = auto()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ring_to_matrix(u: Z_OMEGA, t: Z_OMEGA, k: int) -> np.ndarray:
    scale = 2 ** (-k / 2)
    return scale * np.array([
        [complex(u),        complex(-t.conj())],
        [complex(t),         complex(u.conj())],
    ])


def _fidelity(u: np.ndarray, v: np.ndarray) -> float:
    """Average gate fidelity proxy: |tr(U†V)|² / 4."""
    return abs(np.trace(u.conj().T @ v)) ** 2 / 4


def _t_count_to_eps(t_target: float) -> mpfr:
    """
    Compute epsilon such that k_min ≈ t_target.

    candidate_generator_direct uses k_min = floor(3/2 · log₂(1/ε)), so
    inverting: ε = 2^(−2·t_target/3).
    """
    return mpfr(2) ** (-mpfr(t_target) / mpfr(3))


def _build_candidate_set(
    eps: mpfr,
    init_angle: Union[float, mpfr],
    n_cands: int,
    use_qasm: bool = False,
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Core routine: generate a deduplicated, T-count-sorted candidate set for
    Rz(init_angle) at the given epsilon.

    Sets the gmpy2 precision context from eps, normalises the angle to the
    convention expected by approxRz_direct_ensemble (−angle/2 mapped into
    [0, 2π]), then collects up to n_cands distinct (t_count, gate_string,
    unitary_matrix) tuples.
    """
    prec = int(gmpy2.ceil(abs(gmpy2.log10(eps))))
    bits = max(int(gmpy2.ceil(gmpy2.log2(10 ** (2.5 * prec + 15)))), 100)
    gmpy2.get_context().precision = bits

    theta = -mpfr(init_angle) / 2
    PI = gmpy2.const_pi()
    while theta < 0:
        theta += 2 * PI
    while theta > 2 * PI:
        theta -= 2 * PI

    with gmpy2.local_context(gmpy2.get_context(), precision=bits):
        ring_candidates = approxRz_direct_ensemble(eps, theta, n_cands)

    seen = {}  # (t_count, gate_string) -> unitary_matrix
    for u, t, k in ring_candidates:
        if use_qasm:
            seq, t_count = exact_decomp_to_qasm(u, t, k)
        else:
            seq, t_count = exact_decomp_to_matrix_string(u, t, k)
        key = (t_count, seq)
        if key not in seen:
            seen[key] = _ring_to_matrix(u, t, k)

    return sorted([(t, s, m) for (t, s), m in seen.items()], key=lambda x: x[0])


def _select_n_candidates(
    candidates: List[Tuple[int, str, np.ndarray]],
    n: int,
    target: Optional[np.ndarray],
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Select n candidates using the lowest T counts first.
    Ties at the boundary are broken by fidelity to target when provided,
    otherwise taken arbitrarily.
    """
    cutoff_t = candidates[n - 1][0]
    below = [c for c in candidates if c[0] < cutoff_t]
    at_cutoff = [c for c in candidates if c[0] == cutoff_t]
    need = n - len(below)

    if need == len(at_cutoff):
        return below + at_cutoff

    if target is not None:
        at_cutoff = sorted(at_cutoff, key=lambda c: _fidelity(c[2], target), reverse=True)

    return below + at_cutoff[:need]


# ---------------------------------------------------------------------------
# Low-level ring-element ensemble generator
# ---------------------------------------------------------------------------

def approxRz_direct_ensemble(
    eps: mpfr, theta: mpfr, n_cands: int = 100
) -> List[Tuple[Z_OMEGA, Z_OMEGA, int]]:
    """
    Find up to n_cands approximations of Rz(theta) as unitaries with entries
    from D[ω].  Returns a list of (u, t, k) ring-element tuples in generator
    order.
    """
    u_correction, theta_correction = corrections(theta)
    u_generator = candidate_generator_direct(eps, theta + theta_correction)
    results = []
    for u, k in u_generator:
        while is_reducible(u):
            k -= 1
            u = reduce(u)
        xi = Z_SQRT2(2**k, 0) - (u * u.conj()).to_zsqrt()
        t = solveDiophantine(xi)
        if t is not None:
            assert (t.conj() * t).to_zsqrt() == xi
            results.append((u * u_correction, t * u_correction, k))
            if len(results) == n_cands:
                break
    return results


# ---------------------------------------------------------------------------
# Public API — ensemble generation
# ---------------------------------------------------------------------------

def approximate_rz_direct_set(
    numerator: int,
    denominator: int,
    precision: int,
    n_cands: int = 100,
    use_qasm: bool = False,
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Generate an ensemble of Clifford+T approximations of Rz(numerator·π/denominator)
    to operator norm error 10^{-precision}.

    Returns a sorted, deduplicated list of (t_count, gate_string, unitary_matrix) tuples.
    """
    if numerator > 2 * denominator or numerator < 1:
        raise ValueError("Angle must be in range (0, 2*pi)")

    factor = gcd(numerator, denominator)
    numerator = numerator // factor
    denominator = denominator // factor

    # Compute eps at reduced precision to avoid rounding into a tighter bin
    bits = max(int(gmpy2.ceil(gmpy2.log2(10 ** (2.5 * precision + 15)))), 100)
    gmpy2.get_context().precision = bits
    with gmpy2.local_context(gmpy2.get_context(), round=gmpy2.RoundDown):
        eps = mpfr("1e-{}".format(precision))

    init_angle = mpfr(numerator) * gmpy2.const_pi() / mpfr(denominator)
    return _build_candidate_set(eps, init_angle, n_cands, use_qasm)


def _sweep_candidate_set(
    init_angle: Union[float, mpfr],
    t_min: int,
    t_max: int,
    n_cands_per_level: int = 5,
    use_qasm: bool = False,
    t_step: int = 1,
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Generate candidates across a range of T counts by sweeping epsilon from
    ε(t_min) down to ε(t_max) and merging the results.

    t_step controls the stride through T levels; t_step > 1 reduces gridsynth
    calls at the cost of missing some T levels in the landscape.
    """
    seen = {}
    for t_target in range(t_min, t_max + 1, t_step):
        eps = _t_count_to_eps(t_target)
        for t_count, seq, matrix in _build_candidate_set(eps, init_angle, n_cands_per_level, use_qasm):
            key = (t_count, seq)
            if key not in seen:
                seen[key] = matrix
    return sorted([(t, s, m) for (t, s), m in seen.items()], key=lambda x: x[0])


def approximate_u3_direct_set(
    theta: float,
    phi: float,
    lam: float,
    t_budget: int,
    n_cands_per_level: int = 5,
    t_min: int = 4,
    t_step: int = 1,
) -> Tuple[
    List[Tuple[int, str, np.ndarray]],
    List[Tuple[int, str, np.ndarray]],
    List[Tuple[int, str, np.ndarray]],
]:
    """
    Approximate U3(theta, phi, lam) using the decomposition:

        U3 = Rz(phi + 5π/2) · H · Rz(theta) · H · Rz(lam − π/2)

    For each Rz, sweeps epsilon from T=t_min up to T=t_budget, generating
    candidates across that full T-count range.  The resulting per-Rz sets
    span approximately [t_min, t_budget] T gates, so combined U3 T counts
    span [3·t_min, 3·t_budget] — giving the optimizer a rich landscape to
    work over when targeting any expected total T in that range.

    Returns (rz1, rz2, rz3): candidate sets for
        rz1 — Rz(lam − π/2)       rightmost gate
        rz2 — Rz(theta)            middle gate
        rz3 — Rz(phi + 5π/2)      leftmost gate
    each a sorted list of (t_count, gate_string, unitary_matrix) tuples.
    """
    rz_angles = [
        lam - np.pi / 2,
        theta,
        phi + 5 * np.pi / 2,
    ]

    return tuple(
        _sweep_candidate_set(angle, t_min, t_budget, n_cands_per_level, t_step=t_step)
        for angle in rz_angles
    )


# ---------------------------------------------------------------------------
# Public API — channel construction and optimization
# ---------------------------------------------------------------------------

def _choi(U: np.ndarray) -> np.ndarray:
    """Choi matrix of the unitary channel U(·)U† (unnormalized, trace = d = 2)."""
    vec = np.kron(np.eye(2, dtype=complex), U) @ np.array([1, 0, 0, 1], dtype=complex)
    return np.outer(vec, vec.conj())


def channel_metrics(
    unitaries: List[np.ndarray],
    weights: List[float],
    target: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute three quality metrics for the mixed unitary channel Σᵢ pᵢ Uᵢ(·)Uᵢ†
    versus the target unitary U.

    Returns
    -------
    diamond_norm : float in [0, 2]
        Worst-case distinguishability.  0 = identical, 2 = perfectly distinguishable
        (e.g., I vs X).  Equals ‖J_Φ − J_Ψ‖₁ / d with d=2.
    F_avg : float in [0, 1]
        Average gate fidelity over Haar-random inputs.
        F_avg = (2·F_e + 1) / 3 for a single-qubit channel.
    F_e : float in [0, 1]
        Process (entanglement) fidelity = fidelity on a maximally entangled input.
        F_e = Σᵢ pᵢ |tr(U†Uᵢ)|² / 4.
    """
    d = 2  # single qubit
    J_target = _choi(target)
    J_mix = sum(w * _choi(U) for w, U in zip(weights, unitaries))
    delta = J_mix - J_target
    diamond_norm = float(np.sum(np.abs(np.linalg.eigvalsh(delta)))) / d

    traces_sq = [abs(np.trace(target.conj().T @ U)) ** 2 for U in unitaries]
    F_e = float(sum(w * ts / 4 for w, ts in zip(weights, traces_sq)))
    F_avg = (2 * F_e + 1) / 3

    return diamond_norm, F_avg, F_e


def build_u3_channel(
    rz1: List[Tuple[int, str, np.ndarray]],
    rz2: List[Tuple[int, str, np.ndarray]],
    rz3: List[Tuple[int, str, np.ndarray]],
    t_budget: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Enumerate all U3 = Rz3 · H · Rz2 · H · Rz1 combinations whose total
    T count satisfies T_rz1 + T_rz2 + T_rz3 ≤ t_budget.

    Returns (unitaries, t_counts) — every member already satisfies the budget,
    so the caller can optimize weights without any T constraint.
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    unitaries = []
    t_counts = []
    for t1, _, m1 in rz1:
        for t2, _, m2 in rz2:
            if t1 + t2 > t_budget:
                break
            for t3, _, m3 in rz3:
                total = t1 + t2 + t3
                if total > t_budget:
                    break
                unitaries.append(m3 @ H @ m2 @ H @ m1)
                t_counts.append(total)
    return unitaries, t_counts


def optimize_channel_weights(
    unitaries: List[np.ndarray],
    target: np.ndarray,
) -> List[float]:
    """
    Find probability weights that minimize the diamond norm distance between
    the mixed unitary channel Σ_i p_i U_i(·)U_i† and the target channel U(·)U†.

    All members are assumed to already satisfy whatever T budget applies
    (enforced upstream by build_u3_channel or similar).

    The diamond norm of the difference equals the trace norm of the difference
    of Choi matrices, which is convex in p — solved here via SLSQP.

    Parameters
    ----------
    unitaries : list of 2x2 complex numpy arrays
    target    : 2x2 target unitary matrix

    Returns
    -------
    weights : list of floats summing to 1
    """
    from scipy.optimize import minimize

    n = len(unitaries)
    J_target = _choi(target)
    J_us = [_choi(U) for U in unitaries]

    d = 2
    def objective(p):
        J_mixed = sum(p[i] * J_us[i] for i in range(n))
        return float(np.sum(np.abs(np.linalg.eigvalsh(J_mixed - J_target)))) / d

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
        options={"ftol": 1e-12},
    )
    return result.x.tolist()



def optimize_channel_weights_hs(
    unitaries: List[np.ndarray],
    target: np.ndarray,
) -> List[float]:
    """
    Find probability weights that minimize the Hilbert-Schmidt distance between
    the mixed unitary channel and the target:

        min  ‖J_Φ − J_target‖_F²  =  pᵀ G p − 2 bᵀ p  +  const

    where the Gram matrix G and vector b are:

        G_ij = Tr(J_Uᵢ J_Uⱼ) = |tr(Uᵢ† Uⱼ)|²
        b_i  = Tr(J_Uᵢ J_target) = |tr(Uᵢ† target)|²

    The cross-terms G_ij reward spreading weight across *complementary* unitaries
    whose errors point in different directions in operator space — this is the
    Gram-matrix formulation of F_e / process-fidelity optimization.

    Note: maximizing F_e = Σᵢ pᵢ |tr(U†Uᵢ)|²/4 directly is *linear* in p and
    therefore degenerates to the best single unitary.  The HS distance has the
    additional pᵀGp term that makes mixing genuinely beneficial.

    Parameters
    ----------
    unitaries : list of 2×2 complex numpy arrays
    target    : 2×2 target unitary matrix

    Returns
    -------
    weights : list of floats summing to 1
    """
    from scipy.optimize import minimize

    n = len(unitaries)
    traces = [np.trace(U.conj().T @ target) for U in unitaries]
    b = np.array([abs(t) ** 2 for t in traces], dtype=float)

    cross = np.array(
        [[abs(np.trace(unitaries[i].conj().T @ unitaries[j])) ** 2
          for j in range(n)]
         for i in range(n)],
        dtype=float,
    )

    def objective(p):
        return float(p @ cross @ p - 2.0 * b @ p)

    def grad(p):
        return 2.0 * cross @ p - 2.0 * b

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
        options={"ftol": 1e-14},
    )
    return result.x.tolist()


def build_u3_channel_reduced(
    rz1: List[Tuple[int, str, np.ndarray]],
    rz2: List[Tuple[int, str, np.ndarray]],
    rz3: List[Tuple[int, str, np.ndarray]],
    t_budget: int,
    target: np.ndarray,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Like build_u3_channel but collapses the O(N³) cross-product to one
    representative per T_total level — the combination with the highest
    process fidelity to target at each T count.

    Returns (unitaries, t_counts) with at most one entry per distinct T_total.
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    best: dict = {}  # t_total -> (F_e, U3_matrix)
    for t1, _, m1 in rz1:
        for t2, _, m2 in rz2:
            if t1 + t2 > t_budget:
                break
            for t3, _, m3 in rz3:
                total = t1 + t2 + t3
                if total > t_budget:
                    break
                U3 = m3 @ H @ m2 @ H @ m1
                fe = abs(np.trace(target.conj().T @ U3)) ** 2 / 4
                if total not in best or fe > best[total][0]:
                    best[total] = (fe, U3)
    if not best:
        return [], []
    t_sorted = sorted(best)
    return [best[t][1] for t in t_sorted], list(t_sorted)


def build_mixed_unitary_channel(
    candidates: List[Tuple[int, str, np.ndarray]],
    n: int,
    mode: ChannelMode = ChannelMode.RZ,
    target: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Build a uniform mixed unitary channel from n gate sequences.

    Selects the n lowest T-count candidates. Ties at the boundary are broken
    by fidelity to target when provided, otherwise arbitrarily.

    Parameters
    ----------
    candidates : list of (t_count, gate_string, unitary_matrix)
        As returned by approximate_rz_direct_set or one element of
        approximate_u3_direct_set.
    n : int
        Number of sequences to use. Must satisfy 1 <= n <= len(candidates).
    mode : ChannelMode
        RZ        — single-qubit Rz; target optional for tie-breaking
        U3        — not yet implemented
        ARBITRARY — target matrix required for tie-breaking
    target : np.ndarray, optional
        2x2 target unitary. Required for ARBITRARY mode; used for tie-breaking
        in RZ mode when candidates share the boundary T count.

    Returns
    -------
    (unitaries, weights)
        unitaries : list of n 2x2 complex numpy arrays
        weights   : list of n uniform floats summing to 1
    """
    if not 1 <= n <= len(candidates):
        raise ValueError(f"n must be between 1 and {len(candidates)}, got {n}")

    if mode == ChannelMode.U3:
        raise NotImplementedError("U3 mode not yet implemented")

    if mode == ChannelMode.ARBITRARY and target is None:
        raise ValueError("target must be provided for ARBITRARY mode")

    selected = _select_n_candidates(candidates, n, target)
    unitaries = [m for _, _, m in selected]
    weights = [1.0 / n] * n

    return unitaries, weights


# ---------------------------------------------------------------------------
# Arbitrary-precision (mpmath) pipeline
# ---------------------------------------------------------------------------
# Functions suffixed _mp accept / return mpmath matrices instead of numpy.
# Precision path: Z_OMEGA integers → mpmath at `dps` decimal places →
#   Choi matrix (4×4) → eigenvalues via mp.eig → trace norm / optimizer.
#
# Key identity for Z_OMEGA(a1,a2,a3,a4) = a1ω³+a2ω²+a3ω+a4, ω=exp(iπ/4):
#   real = a4 + (a3−a1)/√2
#   imag = a2 + (a3+a1)/√2
# All coefficients are Python integers (exact); only √2 requires mpmath.
#
# The trig angles (theta, phi, lam) remain Python floats (~16 digits), which
# is the true precision limit for the target unitary regardless of dps.
# ---------------------------------------------------------------------------


def _to_mp_matrix(M, mp_module) -> object:
    """Convert a 2×2 numpy array or mpmath matrix to an mpmath matrix."""
    if isinstance(M, mp_module.matrix):
        return M
    return mp_module.matrix(
        [[mp_module.mpc(str(complex(x).real), str(complex(x).imag)) for x in row]
         for row in M]
    )


def _zomega_to_mpmath(z: Z_OMEGA, sqrt2_mp) -> object:
    """
    Convert Z_OMEGA(a1,a2,a3,a4) to mpmath.mpc using a pre-computed √2.
    Caller must set mpmath.workdps context.
    """
    import mpmath as mp
    re = int(z.a4) + (int(z.a3) - int(z.a1)) / sqrt2_mp
    im = int(z.a2) + (int(z.a3) + int(z.a1)) / sqrt2_mp
    return mp.mpc(re, im)


def _ring_to_matrix_mp(u: Z_OMEGA, t: Z_OMEGA, k: int, dps: int = 50) -> object:
    """
    Convert ring elements (u, t, k) from approxRz_direct_ensemble to an mpmath
    2×2 unitary matrix at `dps` decimal places.

    The exact ring unitary is [[u, -t̄], [t, ū]] / √(2^k)  with u, t ∈ Z[ω].
    All integer coefficients of u, t are propagated exactly; only √2 and the
    scale 2^(−k/2) are evaluated in mpmath arithmetic.
    """
    import mpmath as mp
    with mp.workdps(dps + 10):
        sqrt2 = mp.sqrt(2)
        u_mp  = _zomega_to_mpmath(u,        sqrt2)
        t_mp  = _zomega_to_mpmath(t,        sqrt2)
        tc_mp = _zomega_to_mpmath(t.conj(), sqrt2)
        uc_mp = _zomega_to_mpmath(u.conj(), sqrt2)
        scale = mp.power(sqrt2, -k)   # = 2^(−k/2), exact in mpmath
        return mp.matrix([
            [scale * u_mp,  -scale * tc_mp],
            [scale * t_mp,   scale * uc_mp],
        ])


def _build_candidate_set_mp(
    eps,
    init_angle,
    n_cands: int,
    dps: int = 50,
) -> list:
    """
    Like _build_candidate_set but returns (t_count, seq, mpmath_matrix) tuples.
    """
    prec = int(gmpy2.ceil(abs(gmpy2.log10(eps))))
    bits = max(int(gmpy2.ceil(gmpy2.log2(10 ** (2.5 * prec + 15)))), 100)
    gmpy2.get_context().precision = bits

    theta = -mpfr(init_angle) / 2
    PI = gmpy2.const_pi()
    while theta < 0:
        theta += 2 * PI
    while theta > 2 * PI:
        theta -= 2 * PI

    with gmpy2.local_context(gmpy2.get_context(), precision=bits):
        ring_candidates = approxRz_direct_ensemble(eps, theta, n_cands)

    seen = {}
    for u, t, k in ring_candidates:
        seq, t_count = exact_decomp_to_matrix_string(u, t, k)
        key = (t_count, seq)
        if key not in seen:
            seen[key] = _ring_to_matrix_mp(u, t, k, dps=dps)

    return sorted([(tc, s, m) for (tc, s), m in seen.items()], key=lambda x: x[0])


def _sweep_candidate_set_mp(
    init_angle,
    t_min: int,
    t_max: int,
    n_cands_per_level: int = 5,
    t_step: int = 1,
    dps: int = 50,
) -> list:
    """Like _sweep_candidate_set but returns mpmath matrices."""
    seen = {}
    for t_target in range(t_min, t_max + 1, t_step):
        eps = _t_count_to_eps(t_target)
        for t_count, seq, mat in _build_candidate_set_mp(
                eps, init_angle, n_cands_per_level, dps=dps):
            key = (t_count, seq)
            if key not in seen:
                seen[key] = mat
    return sorted([(tc, s, m) for (tc, s), m in seen.items()], key=lambda x: x[0])


def approximate_u3_direct_set_mp(
    theta: float,
    phi: float,
    lam: float,
    t_budget: int,
    n_cands_per_level: int = 5,
    t_min: int = 4,
    t_step: int = 1,
    dps: int = 50,
) -> tuple:
    """
    Like approximate_u3_direct_set but returns (t_count, seq, mpmath_matrix) tuples
    for each Rz angle in U3(θ,φ,λ) = Rz(φ+5π/2)·H·Rz(θ)·H·Rz(λ−π/2).

    Matrix entries are computed from exact Z[ω] integers; only the trig angle
    is float64 (~16-digit limit on angular accuracy, independent of dps).
    """
    rz_angles = [lam - np.pi / 2, theta, phi + 5 * np.pi / 2]
    return tuple(
        _sweep_candidate_set_mp(angle, t_min, t_budget, n_cands_per_level, t_step, dps=dps)
        for angle in rz_angles
    )


def build_u3_channel_reduced_mp(
    rz1: list,
    rz2: list,
    rz3: list,
    t_budget: int,
    target,
    dps: int = 50,
    n_keep: int = 1,
) -> tuple:
    """
    Like build_u3_channel_reduced but final unitaries are in mpmath precision.

    rz1/rz2/rz3 : (t_count, seq, mpmath_matrix) tuples
    target       : mpmath or numpy 2×2 matrix
    n_keep       : how many candidates to retain per T-total level (default 1 = best only).
                   Set > 1 to keep diverse residuals for better first-moment cancellation.
    Returns (unitaries_mp, t_counts).

    Selection uses float64 (fast triple loop); only the kept combinations
    are recomputed in mpmath for full precision.
    """
    import mpmath as mp
    import heapq

    # Build float64 versions of candidates for the fast selection loop
    def _to_np2(M):
        if isinstance(M, mp.matrix):
            return np.array([[complex(M[i, j]) for j in range(2)]
                             for i in range(2)], dtype=complex)
        return np.asarray(M, dtype=complex)

    H_np = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if isinstance(target, mp.matrix):
        target_np = _to_np2(target)
    else:
        target_np = np.asarray(target, dtype=complex)
    tH = target_np.conj().T

    # (t_count, numpy_matrix, original_index_into_rzN list)
    rz1_np = [(t, _to_np2(m), i) for i, (t, _, m) in enumerate(rz1)]
    rz2_np = [(t, _to_np2(m), i) for i, (t, _, m) in enumerate(rz2)]
    rz3_np = [(t, _to_np2(m), i) for i, (t, _, m) in enumerate(rz3)]

    # Selection in float64: keep top-n_keep candidates per T-total by process fidelity
    # pool[t_total] is a min-heap of (fe, counter, i1, i2, i3) of length <= n_keep
    pool: dict = {}
    _ctr = 0
    for t1, m1, i1 in rz1_np:
        Hm1 = H_np @ m1          # precompute: only depends on t1
        for t2, m2, i2 in rz2_np:
            if t1 + t2 > t_budget:
                break
            m321_pre = m2 @ Hm1  # m2 * H * m1
            for t3, m3, i3 in rz3_np:
                total = t1 + t2 + t3
                if total > t_budget:
                    break
                U3 = m3 @ H_np @ m321_pre
                tr = np.trace(tH @ U3)
                fe = abs(tr) ** 2 / 4
                entry = (fe, _ctr, i1, i2, i3)
                _ctr += 1
                if total not in pool:
                    pool[total] = []
                h = pool[total]
                if len(h) < n_keep:
                    heapq.heappush(h, entry)
                elif fe > h[0][0]:          # better than current worst
                    heapq.heapreplace(h, entry)

    if not pool:
        return [], []

    # Rebuild winners in mpmath for full precision; flatten all kept candidates
    with mp.workdps(dps + 10):
        H_mp = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)
        unitaries_mp = []
        t_counts_out = []
        for t_total in sorted(pool):
            # Sort by descending fe so best member comes first within each T-level
            for fe, _, i1, i2, i3 in sorted(pool[t_total], reverse=True):
                m1 = rz1[i1][2]
                m2 = rz2[i2][2]
                m3 = rz3[i3][2]
                unitaries_mp.append(m3 * H_mp * m2 * H_mp * m1)
                t_counts_out.append(t_total)

    return unitaries_mp, t_counts_out


def _choi_mp(U_mp, mp_module) -> object:
    """
    Choi matrix of U(·)U† as a 4×4 mpmath matrix.
    J_U = vec(U) vec(U)†  where vec(U) = [U₀₀, U₁₀, U₀₁, U₁₁]ᵀ (column-major).
    Caller must set workdps context.
    """
    vec = mp_module.matrix([U_mp[0, 0], U_mp[1, 0], U_mp[0, 1], U_mp[1, 1]])
    return vec * vec.H


def channel_metrics_mp(
    unitaries: list,
    weights: list,
    target,
    dps: int = 50,
) -> Tuple[float, float, float]:
    """
    Like channel_metrics but uses mpmath for arbitrary-precision evaluation.

    unitaries : list of mpmath or numpy 2×2 matrices
    weights   : list of floats
    target    : mpmath or numpy 2×2 matrix
    dps       : decimal places (default 50 ≈ 150 bits)

    Returns (diamond_norm, F_avg, F_e).
    diamond_norm = trace_norm(J_Φ − J_target) / d  with d=2.
    """
    import mpmath as mp
    with mp.workdps(dps + 10):
        target_mp = _to_mp_matrix(target, mp)
        J_target = _choi_mp(target_mp, mp)
        J_mix = sum(
            mp.mpf(str(w)) * _choi_mp(_to_mp_matrix(U, mp), mp)
            for w, U in zip(weights, unitaries)
        )
        delta = J_mix - J_target

        # mp.eig works for complex Hermitian (eigsy requires real symmetric)
        # Eigenvalues of a Hermitian matrix are real; imaginary parts are noise.
        E, _ = mp.eig(delta)
        diamond_norm = float(sum(abs(mp.re(e)) for e in E)) / 2  # /d with d=2

        def _mp_trace_dag_u(U):
            prod = target_mp.H * _to_mp_matrix(U, mp)
            return sum(prod[i, i] for i in range(2))

        traces_sq = [abs(_mp_trace_dag_u(U)) ** 2 for U in unitaries]
        F_e = float(
            sum(mp.mpf(str(w)) * ts / 4 for w, ts in zip(weights, traces_sq))
        )
        F_avg = (2 * F_e + 1) / 3

        return diamond_norm, F_avg, F_e


def _proj_simplex_mp(v: list, mp_module) -> list:
    """
    Project a list of mpmath numbers onto the probability simplex.
    Uses the O(n log n) algorithm of Duchi et al. (2008).
    """
    mp = mp_module
    n = len(v)
    u = sorted(v, reverse=True)
    rho, cssv = 0, mp.mpf(0)
    for j in range(n):
        cssv += u[j]
        if u[j] + (mp.mpf(1) - cssv) / (j + 1) > 0:
            rho = j + 1
    theta = (sum(u[:rho]) - mp.mpf(1)) / rho
    return [max(mp.mpf(0), vi - theta) for vi in v]


def optimize_channel_weights_mp(
    unitaries: list,
    target,
    dps: int = 50,
    n_iter: int = 200,
    init_weights: Optional[List[float]] = None,
) -> List[float]:
    """
    Minimize the diamond norm over channel weights using mpmath arithmetic
    and projected subgradient descent.

    Algorithm
    ---------
    Warm start: SLSQP in float64 gives ~10 digits.
    Refinement: projected subgradient in mpmath at `dps` decimal places.

    Subgradient of f(p) = ‖Σᵢ pᵢ Jᵢ − J_target‖₁ / 2:
      ∂f/∂pᵢ = Re(vᵢ† G vᵢ) / 2
    where G = Q sign(Λ) Q† is the subgradient of the trace norm at Σpᵢ Jᵢ − J_t
    and vᵢ = vec(Uᵢ) is the column-vectorized unitary.

    Parameters
    ----------
    unitaries    : list of mpmath or numpy 2×2 matrices
    target       : mpmath or numpy 2×2 matrix
    dps          : decimal places (default 50)
    n_iter       : mpmath refinement iterations (default 200)
    init_weights : optional warm-start weights (default: run SLSQP first)

    Returns
    -------
    weights : list of floats summing to 1
    """
    import mpmath as mp

    n = len(unitaries)

    # --- float64 warm start (SLSQP needs numpy arrays) ---
    if init_weights is None:
        def _to_np(M):
            if isinstance(M, mp.matrix):
                return np.array([[complex(M[i, j]) for j in range(M.cols)]
                                 for i in range(M.rows)], dtype=complex)
            return M
        init_weights = optimize_channel_weights(
            [_to_np(u) for u in unitaries], _to_np(target)
        )

    with mp.workdps(dps + 10):
        target_mp = _to_mp_matrix(target, mp)
        J_target  = _choi_mp(target_mp, mp)

        # Pre-compute vec(Uᵢ) and Choi matrices in mpmath
        vecs = [mp.matrix([U[0,0], U[1,0], U[0,1], U[1,1]])
                for U in [_to_mp_matrix(u, mp) for u in unitaries]]
        Js   = [v * v.H for v in vecs]

        tol = mp.mpf(10) ** (-(dps // 2))

        def _obj_and_subgrad(p_list):
            H = sum(p_list[i] * Js[i] for i in range(n)) - J_target
            E, Q = mp.eig(H)
            tn = sum(abs(mp.re(e)) for e in E)
            signs = [
                mp.sign(mp.re(e)) if abs(mp.re(e)) > tol else mp.mpf(0)
                for e in E
            ]
            g = []
            for vi in vecs:
                # c = Q† vi  (4×1), gradient = Σ_k sign_k |c_k|² / 2
                c  = Q.H * vi
                gi = sum(signs[k] * abs(c[k, 0]) ** 2 for k in range(4)) / 2
                g.append(mp.re(gi))
            return tn / 2, g

        p = [mp.mpf(str(w)) for w in init_weights]
        best_p = list(p)
        best_f, _ = _obj_and_subgrad(p)

        for t in range(1, n_iter + 1):
            f, g = _obj_and_subgrad(p)
            if f < best_f:
                best_f = f
                best_p = list(p)

            g_norm_sq = sum(gi ** 2 for gi in g)
            if g_norm_sq < tol ** 2:
                break

            # Polyak-style step toward best seen; fall back to diminishing step
            gap = f - best_f + mp.mpf(10) ** (-(dps // 3))
            alpha = gap / g_norm_sq if gap > 0 else mp.mpf(1) / mp.sqrt(mp.mpf(t) * g_norm_sq)

            p_new = [p[i] - alpha * g[i] for i in range(n)]
            p = _proj_simplex_mp(p_new, mp)

        f, _ = _obj_and_subgrad(p)
        if f < best_f:
            best_f = f
            best_p = list(p)

        return [float(pi) for pi in best_p]
