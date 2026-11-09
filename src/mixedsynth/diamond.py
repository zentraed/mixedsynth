"""
Diamond-norm/2 certificate via Choi trace distance (called once per gate, not in the loop).

Delegates to genlist.channel_metrics_mp which computes:

    choi_trace_dist = trace_norm(J_Λ − J_V) / d   (d = 2 for a qubit)

For a unitary target V on a qubit, this equals the diamond norm / 2 exactly:

    ‖Λ − V‖◇ / 2 = trace_norm(J_Λ − J_V) / 2

Proof sketch: J_V is a pure state (rank-1 Choi), so J_Δ = J_Λ − J_V has at most
2 non-zero eigenvalues. The Watrous SDP reduces to a 4×4 eigenvalue problem with
a closed-form solution that equals the trace-norm formula above. This is computed
via mp.eig (not an SDP), giving arbitrary-precision results at all T budgets.

Reference: Watrous, "Theory of Quantum Information" §3.3; also Wilde, Theorem 9.2.1.

For non-unitary targets or d > 2, the trace-norm formula no longer equals the
diamond norm and a full SDP would be required.
"""


from mixedsynth.genlist import channel_metrics_mp


def dn2_choi_mp(unitaries_mp, weights, target_mp, dps=60):
    """
    Compute diamond_norm/2 for the mixed channel Λ = Σ_i w_i (U_i · U_i†).

    unitaries_mp : list of mpmath 2×2 unitaries
    weights      : list/array of real weights (sum to 1)
    target_mp    : mpmath 2×2 target unitary
    dps          : working decimal places

    Returns mpmath mpf (dn/2).
    """
    dn, _favg, _fe = channel_metrics_mp(unitaries_mp, list(weights), target_mp, dps=dps)
    return dn / 2


def best_single_dn2_mp(unitaries_mp, target_mp, dps=60):
    """
    dn2 of the best single member: min_i dn(U_i, V)/2.
    Returns (dn2_best, index).
    """
    best = None
    best_i = 0
    for i, U in enumerate(unitaries_mp):
        dn, _, _ = channel_metrics_mp([U], [1.0], target_mp, dps=dps)
        d2 = dn / 2
        if best is None or float(d2) < float(best):
            best = d2
            best_i = i
    return best, best_i
