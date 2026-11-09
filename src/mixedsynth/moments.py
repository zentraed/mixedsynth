"""
Moment-based diamond-error estimate (no SDP).

For weights p and rotation vectors v_i:
  M1(p) = Σ_i p_i v_i                 (3-vector, coherent residual)
  S2(p) = Σ_i p_i v_i v_iᵀ           (3×3, second-moment matrix)
  C(p)  = S2 - M1 M1ᵀ                 (3×3, centered covariance)

Leading-order diamond norm estimate (k1=1/2, k2=1/4 exact for SU(2)):
  dn2_moment = (1/2)|M1| + (1/4) tr(C)

All computed in mpmath at the caller's working dps.

k1 derivation: dn2 of rotation by α is sin(α/2) ≈ α/2 = |v|/2 → k1 = 1/2.
k2 derivation: incoherent channel dn2 = (1/4) Σ_i p_i |v_i|² = (1/4) tr(S2)
               and when M1=0, tr(C)=tr(S2), giving k2=1/4.
"""

import mpmath as mp

K1 = 0.5   # coherent prefactor
K2 = 0.25  # incoherent prefactor


def compute_moments_mp(vs_mp, p_f64, dps=60):
    """
    Compute M1, S2, C, tr(C), floor2 in mpmath.

    vs_mp  : list of [vx, vy, vz] (each component an mpmath mpf)
    p_f64  : (N,) float64 or list of mpmath mpf weights. mpf weights are used
             as-is (no float64 roundtrip), enabling exact M1 cancellation at
             high T where float64 weight precision limits the achievable M1.
    dps    : working decimal places

    Returns dict with mpmath values:
      M1      : [3] list of mpf — first moment
      S2      : 3×3 list of lists of mpf — second-moment matrix
      C       : 3×3 list of lists of mpf — centred covariance
      tr_C    : mpf — tr(C)
      floor2  : mpf = K2 * tr(C) — achievable incoherent floor
      M1_norm : mpf = |M1|
    """
    N = len(vs_mp)
    with mp.workdps(dps + 10):
        # M1 = Σ p_i v_i
        M1 = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        for i in range(N):
            pi = p_f64[i]
            w = pi if isinstance(pi, mp.mpf) else mp.mpf(float(pi))
            for k in range(3):
                M1[k] += w * vs_mp[i][k]

        # S2 = Σ p_i v_i v_iᵀ  (3×3)
        S2 = [[mp.mpf(0)] * 3 for _ in range(3)]
        for i in range(N):
            pi = p_f64[i]
            w = pi if isinstance(pi, mp.mpf) else mp.mpf(float(pi))
            for j in range(3):
                for k in range(3):
                    S2[j][k] += w * vs_mp[i][j] * vs_mp[i][k]

        # C = S2 - M1 M1ᵀ
        C = [[S2[j][k] - M1[j] * M1[k] for k in range(3)] for j in range(3)]

        tr_C = C[0][0] + C[1][1] + C[2][2]
        M1_norm = mp.sqrt(M1[0] ** 2 + M1[1] ** 2 + M1[2] ** 2)
        floor2 = mp.mpf(K2) * tr_C

    return {
        "M1": M1,
        "S2": S2,
        "C": C,
        "tr_C": tr_C,
        "floor2": floor2,
        "M1_norm": M1_norm,
    }


def dn2_moment_mp(M1_norm, floor2):
    """
    dn2_moment = K1 * |M1| + K2 * tr(C) = |M1|/2 + floor2.
    All inputs mpmath mpf; returns mpf.
    """
    return mp.mpf(K1) * M1_norm + floor2
