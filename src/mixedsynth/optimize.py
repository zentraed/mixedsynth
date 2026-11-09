"""
Two-stage weight optimization in the generator picture.

Stage 1 — Frank-Wolfe min-norm-point on conv({v_i}) ⊆ R³:
  min_p  |Bp|²  s.t. p ∈ Δ_N
  where B = [v_1 … v_N] (3×N), Δ_N = N-simplex.
  Optimal value = squared distance from origin to conv({v_i}).
  Carathéodory (d=3) ⇒ n_nz ≤ 4 at any optimal solution.

Stage 2 — Linear program on the M1=0 face:
  min_p  Σ p_i |v_i|²  s.t.  Bp = 0,  Σp = 1,  p ≥ 0
  Objective tr(C) = tr(S2) when M1=0 — minimizes the incoherent floor.
  LP at a vertex of the polytope ⇒ still n_nz ≤ 4.

Both stages run in float64. The final moment computation uses mpmath.
"""

import numpy as np
from scipy.optimize import linprog


def frank_wolfe_f64(vs_f64, n_iter=2000, tol=1e-24):
    """
    Frank-Wolfe (conditional gradient) for min-norm-point in conv(vs).

    vs_f64 : (N, 3) float64 array of rotation vectors
    Returns : (p, hull_dist, M1)
      p          : (N,) weight vector (non-negative, sums to 1)
      hull_dist  : distance from origin to conv({v_i})
      M1         : (3,) first moment at optimal weights
    """
    N = len(vs_f64)
    if N == 0:
        return np.array([]), 0.0, np.zeros(3)

    # Initialise at the member closest to the origin (smallest |v_i|)
    norms_sq = np.einsum("ij,ij->i", vs_f64, vs_f64)
    k0 = int(np.argmin(norms_sq))
    p = np.zeros(N)
    p[k0] = 1.0
    M1 = vs_f64[k0].copy()

    best_sq = float(norms_sq[k0])
    best_p = p.copy()
    best_M1 = M1.copy()

    for it in range(n_iter):
        # Recompute M1 every 200 steps to avoid float accumulation drift
        if it > 0 and it % 200 == 0:
            M1 = vs_f64.T @ p

        dots = vs_f64 @ M1               # v_i · M1,  shape (N,)
        k_star = int(np.argmin(dots))

        m1_sq = float(np.dot(M1, M1))
        fw_gap = m1_sq - float(dots[k_star])   # = (|M1|² - v_{k*}·M1) ≥ 0

        if fw_gap <= tol:
            break

        d = vs_f64[k_star] - M1          # direction from current point to FW vertex
        dd = float(np.dot(d, d))
        if dd == 0.0:
            break
        dM = float(np.dot(d, M1))
        gamma = max(0.0, min(1.0, -dM / dd))

        p = (1.0 - gamma) * p
        p[k_star] += gamma
        M1 = M1 + gamma * d

        sq = float(np.dot(M1, M1))
        if sq < best_sq:
            best_sq = sq
            best_p = p.copy()
            best_M1 = M1.copy()

    # Final recompute to remove accumulated drift
    M1_final = vs_f64.T @ best_p
    hull_dist = float(np.sqrt(float(np.dot(M1_final, M1_final))))
    return best_p, hull_dist, M1_final


def stage2_lp(vs_f64, max_m1_ratio=0.01):
    """
    LP to minimise the incoherent floor tr(C) subject to M1=0.

    Solves:  min  Σ p_i |v_i|²
             s.t. Bp = 0  (3 equality constraints)
                  Σp = 1  (normalisation)
                  p ≥ 0

    Scales vectors so that min|v_i|=1 before passing to HiGHS — this ensures
    the LP feasibility tolerance (default ~1e-9) is meaningful relative to the
    actual vector magnitudes (which can be as small as 1e-15 at T≥150).

    Returns weight vector (N,) or None if infeasible or if |M1_result| > max_m1_ratio × |v_best|.
    max_m1_ratio: reject LP solution if residual first moment is too large.
    """
    N = len(vs_f64)
    if N == 0:
        return None

    norms_sq = np.einsum("ij,ij->i", vs_f64, vs_f64)
    best_norm = float(np.sqrt(np.min(norms_sq)))
    if best_norm == 0.0:
        return None

    # Scale so min |v_i| = 1 — keeps LP tolerance meaningful
    scale = 1.0 / best_norm
    vs_s = vs_f64 * scale

    c = np.einsum("ij,ij->i", vs_s, vs_s)          # |v_i_scaled|² — linear in p
    A_eq = np.vstack([vs_s.T, np.ones((1, N))])     # (4, N)
    b_eq = np.zeros(4)
    b_eq[3] = 1.0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, None)] * N,
                  method="highs",
                  options={"disp": False})

    if not (res.success and res.status == 0):
        return None

    p = np.maximum(res.x, 0.0)
    p_sum = p.sum()
    if p_sum == 0.0:
        return None
    p /= p_sum

    # Sanity check: verify |M1| is actually small (LP feasibility held meaningfully)
    M1_check = vs_f64.T @ p   # in original units
    m1_check = float(np.sqrt(float(np.dot(M1_check, M1_check))))
    if m1_check > max_m1_ratio * best_norm:
        return None            # LP satisfied the constraint only nominally; discard

    return p


def optimize_ms(vs_f64, enclosed_tol_ratio=0.1, fw_iter=2000, top_k=40):
    """
    Two-stage optimization in the generator picture.

    vs_f64            : (N, 3) float64 rotation vectors
    enclosed_tol_ratio: origin is 'enclosed' when hull_dist < ratio * |v_best|
    fw_iter           : Frank-Wolfe iteration cap
    top_k             : restrict FW + LP to the top-K smallest-|v_i| members.
                        Large-|v_i| members (bad approximations at low T) have dot
                        products that dwarf the good members, hijacking the FW
                        direction search. Only the top-K near-optimal members can
                        participate in useful first-moment cancellation.
                        None = use all members (only safe when the pool is already
                        homogeneous in magnitude).

    Returns dict with keys:
      p           : (N,) optimal weights (full pool, zeros for non-top-k members)
      hull_dist   : origin-to-hull distance (float64, computed on top-k subset)
      M1_f64      : (3,) first moment at p (float64, for diagnostics)
      enclosed    : bool — whether origin is inside conv({top-k v_i})
      stage2_used : bool
    """
    N = len(vs_f64)
    if N == 0:
        return {"p": np.array([]), "hull_dist": 0.0, "M1_f64": np.zeros(3),
                "enclosed": False, "stage2_used": False}

    norms_sq = np.einsum("ij,ij->i", vs_f64, vs_f64)

    # Select top-k members by smallest |v_i| (best approximation quality)
    if top_k is not None and top_k < N:
        idx = np.argsort(norms_sq)[:top_k]
    else:
        idx = np.arange(N)
    vs_sub = vs_f64[idx]

    norms_sub = np.sqrt(norms_sq[idx])
    best_norm_sub = float(norms_sub[0])  # idx is argsorted, so [0] is smallest |v_i|

    # FW tolerance must be below (best_norm_sub)² so fw_gap = |M1|²−v_{k*}·M1
    # does not fire on the first iteration at high T (where |v_best|² ≪ 1e-24).
    fw_tol = min(1e-24, 1e-8 * float(norms_sq[idx[0]]))
    p_sub, hull_dist, M1 = frank_wolfe_f64(vs_sub, n_iter=fw_iter, tol=fw_tol)

    enclosed = hull_dist < enclosed_tol_ratio * best_norm_sub
    stage2_used = False

    if enclosed:
        p2_sub = stage2_lp(vs_sub)
        if p2_sub is not None:
            p_sub = p2_sub
            M1 = vs_sub.T @ p_sub
            hull_dist = float(np.sqrt(float(np.dot(M1, M1))))
            stage2_used = True

    # Map sub-weights back to full vector
    p = np.zeros(N)
    p[idx] = p_sub
    support = list(np.where(p > 1e-6)[0])  # global indices of non-zero weights

    return {
        "p": p,
        "hull_dist": hull_dist,   # post-LP when stage2 ran, FW value otherwise
        "M1_f64": M1,
        "enclosed": enclosed,
        "stage2_used": stage2_used,
        "support": support,
    }
