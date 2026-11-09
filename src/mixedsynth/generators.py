"""
Residual generator computation.

For each approximation U_i and target V, compute the rotation vector v_i ∈ R³ such that
    W_i = strip_phase(V† U_i) = exp(i (|v_i|/2) n_i · σ)
    v_i = |v_i| n_i   (rotation angle times rotation axis)

All arithmetic in mpmath at working dps. Caller sets mp.workdps context.
"""

import mpmath as mp


def strip_phase(M):
    """
    Project a U(2) matrix into SU(2) by dividing by sqrt(det).
    Fixes the canonical branch so that M[0,0] has non-negative real part.
    """
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    ph = mp.sqrt(det)
    # Canonical branch: Re(M[0,0]/ph) >= 0
    if mp.re(M[0, 0] / ph) < 0:
        ph = -ph
    return M * (mp.mpf(1) / ph)


def rotation_vector(U_mp, V_mp):
    """
    Compute rotation vector v ∈ R³ (as list of 3 mpf) for W = strip_phase(V†U) ∈ SU(2).

    Stable extraction from SU(2) parameters — not via logm or V†U − I:
      W = a₀ I + i (bx σx + by σy + bz σz),  a₀,bx,by,bz ∈ R
      α = 2 atan2(|b|, a₀),  v = α (bx, by, bz) / |b|

    Works in the caller's mp.workdps context.
    """
    W = strip_phase(V_mp.H * U_mp)

    # SU(2): W = [[a₀+i bz,  i bx + by],
    #             [i bx - by, a₀ - i bz]]
    a0 = mp.re(W[0, 0])
    bz = mp.im(W[0, 0])
    bx = mp.im(W[1, 0])   # Im(W[1,0]) = bx
    by = -mp.re(W[1, 0])  # -Re(W[1,0]) = by

    b_sq = bx ** 2 + by ** 2 + bz ** 2
    b_norm = mp.sqrt(b_sq)

    tol = mp.mpf(10) ** (-mp.mp.dps + 5)
    if b_norm < tol:
        return [mp.mpf(0), mp.mpf(0), mp.mpf(0)]

    alpha = 2 * mp.atan2(b_norm, a0)
    scale = alpha / b_norm
    return [scale * bx, scale * by, scale * bz]


def rotation_vectors_mp(members_mp, target_mp, dps=60):
    """
    Compute rotation vectors for a list of mpmath unitaries against a target.

    members_mp : list of mpmath 2×2 matrices
    target_mp  : mpmath 2×2 target matrix
    dps        : working decimal places

    Returns list of (v, mag) where v = [vx, vy, vz] and mag = |v|, all mpf.
    """
    result = []
    with mp.workdps(dps + 10):
        for U in members_mp:
            v = rotation_vector(U, target_mp)
            mag = mp.sqrt(sum(c ** 2 for c in v))
            result.append((v, mag))
    return result
