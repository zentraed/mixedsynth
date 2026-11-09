"""
U3 mixed-channel composition from three independent Rz mixed-channels.

U3(θ,φ,λ) = Rz(φ+5π/2) · H · Rz(θ) · H · Rz(λ−π/2)

Two optimization modes:

optimize_u3 (equal-split)
  Each Rz is optimized independently within t_total//3 T-gates.  The per-shot
  three Rz members are drawn independently.  This cancels the U3 first moment
  when each Rz first moment is zero (because the intervening H Cliffords act as
  invertible SO(3) rotations on su(2)).  E[T_U3] = E[T_rz1]+E[T_rz2]+E[T_rz3].
  The U3 product ensemble has ≤ 4³=64 composite members.

optimize_u3_hardcap (hard constraint)
  Enforces T_1+T_2+T_3 ≤ t_total exactly.  One composite U3 is drawn per shot
  (one composite member is sampled, not three independent per-Rz members).
  Carathéodory in R³ still gives n_nz ≤ 4 composite members in the mixture.
  Allows flexible per-channel allocation (e.g., T1=60, T2=50, T3=40) rather
  than the equal-split t_total//3 per Rz.  E[T] ≤ t_total by construction.
  T=0 (Clifford) and T=1 allocations are supported via t_min=0.
"""

import heapq
import time
import numpy as np
import mpmath as mp

from .harvest import harvest_rz, rz_angles_for_u3, rz_exact_mp
from .generators import rotation_vectors_mp
from .optimize import optimize_ms
from .moments import compute_moments_mp, dn2_moment_mp


def _mp_to_np(U_mp):
    """Convert mpmath 2×2 matrix to numpy complex128 array."""
    return np.array([[complex(U_mp[i, j]) for j in range(2)]
                     for i in range(2)], dtype=complex)


def _channel_members(p, w_mp, t_counts, seqs, tol=1e-12):
    """
    The synthesized mixed channel as a list of its active members, sorted by
    descending weight: [{"weight", "t_count", "sequence"}, ...].  Weights are
    mpmath mpf when the high-precision refinement ran, otherwise float.
    """
    out = []
    for i in np.flatnonzero(p > tol):
        w = w_mp[i] if w_mp is not None else float(p[i])
        out.append({"weight": w, "t_count": t_counts[i], "sequence": seqs[i]})
    out.sort(key=lambda m: -float(m["weight"]))
    return out


def _refine_weights_mp(vs_mp_s, dps=60):
    """
    Given the LP/FW support members (≤4 in R³), solve
        [V_S; 1^T] w = [0, 0, 0, 1]
    in mpmath and return the weight list, or None if infeasible/non-positive.

    For n=4 this is a square solve; for n<4 we use the minimum-norm least-squares
    solution (A^T (A A^T)^{-1} b).
    """
    n = len(vs_mp_s)
    if n < 2:
        return None
    with mp.workdps(dps):
        A = mp.matrix(4, n)
        for j, v in enumerate(vs_mp_s):
            for i in range(3):
                A[i, j] = v[i]
            A[3, j] = mp.mpf(1)
        b = mp.zeros(4, 1)
        b[3] = mp.mpf(1)

        try:
            if n == 4:
                x = mp.lu_solve(A, b)
            else:
                # min-norm least-squares: w = A^T (A A^T)^{-1} b
                AAt = A * A.T
                z = mp.lu_solve(AAt, b)
                x = A.T * z
        except Exception:
            return None

        w = [x[j] for j in range(n)]
        if any(float(wi) < -1e-8 for wi in w):
            return None
        s = sum(w)
        if float(s) <= 0:
            return None
        return [wi / s for wi in w]


def optimize_u3(theta, phi, lam, t_budget=None, t_total=None,
                n_keep=5, stride=None, dps=60,
                enclosed_tol_ratio=0.1, fw_iter=2000,
                t_min=0):
    """
    Harvest and optimize the U3 mixed channel.

    Budget specification (one of):
      t_total  : total T-gate budget for the full U3 shot. Each of the three
                 Rz channels receives t_total // 3 T-gates (equal split, which
                 is optimal for random angles since all channels have the same
                 error scaling).
      t_budget : T-gate budget per Rz channel directly. Use this to override
                 the equal-split allocation.

    Returns a dict with all per-gate metrics, plus:
      t_budget_rz : the per-Rz T budget actually used
    Timing split:
      t_sweep : seconds for all three gridsynth harvests
      t_opt   : seconds for v_i extraction + Frank-Wolfe + LP + moments
    """
    if t_total is not None:
        t_budget = t_total // 3
    elif t_budget is None:
        raise ValueError("Specify either t_budget (per Rz) or t_total (for full U3 shot)")

    angles = rz_angles_for_u3(theta, phi, lam)
    if stride is None:
        stride = max(1, t_budget // 30)

    # ---- harvest ----
    t0 = time.time()
    channels = []
    channel_seqs = []
    for angle in angles:
        members = harvest_rz(angle, t_min=t_min, t_budget=t_budget,
                             n_keep=n_keep, stride=stride, dps=dps,
                             with_sequences=True)
        channels.append([(t, U) for t, _s, U in members])
        channel_seqs.append([sq for _t, sq, _U in members])
    t_sweep = time.time() - t0

    # ---- optimize each Rz independently ----
    t0 = time.time()
    per_rz = []
    for k, (angle, members) in enumerate(zip(angles, channels)):
        if not members:
            per_rz.append(None)
            continue

        t_counts = [t for t, _U in members]
        U_list   = [U for _t, U in members]

        # Exact target for this Rz angle
        V_mp = rz_exact_mp(angle, dps=dps)

        # Rotation vectors in mpmath
        vecs_mp_mags = rotation_vectors_mp(U_list, V_mp, dps=dps)
        vs_mp  = [v for v, _m in vecs_mp_mags]
        vs_f64 = np.array([[float(c) for c in v] for v in vs_mp], dtype=np.float64)

        # Stage 1 + Stage 2 (float64)
        opt = optimize_ms(vs_f64, enclosed_tol_ratio=enclosed_tol_ratio, fw_iter=fw_iter)
        p = opt["p"]

        # Best single member (min |v_i| ≈ min dn2)
        mags_f64 = np.array([float(m) for _v, m in vecs_mp_mags])
        best_i = int(np.argmin(mags_f64))
        eps = float(mags_f64[best_i]) / 2   # dn2 of best single = |v_best|/2

        # mpmath weight refinement: re-solve exact weights on the LP support.
        # Float64 LP selects the support reliably; only weights need precision.
        # Restrict to the clean Carathéodory support (≤4 members, threshold 1e-4)
        # to avoid giving the solver a spurious overdetermined system.
        # Keep weights as mpmath mpf (no float64 roundtrip) so that the moment
        # computation achieves M1≈0 to full mpmath precision at all T.
        w_mp_full = None   # mpmath weights indexed by support; None = use float64 p
        if opt["stage2_used"]:
            support = [i for i in opt["support"] if p[i] > 1e-4]
            if 2 <= len(support) <= 4:
                vs_mp_s = [vs_mp[i] for i in support]
                w_mp = _refine_weights_mp(vs_mp_s, dps=dps)
                if w_mp is not None:
                    # Build a full-length list with mpmath weights at support indices
                    w_mp_full = [mp.mpf(0)] * len(vs_mp)
                    for j, gi in enumerate(support):
                        w_mp_full[gi] = w_mp[j]
                    # Also update float64 p for n_nz / E_T accounting
                    p = np.zeros(len(vs_mp))
                    for j, gi in enumerate(support):
                        p[gi] = float(w_mp[j])

        # Exact moments in mpmath (use mpmath weights if available to avoid float64 roundtrip)
        p_for_moments = w_mp_full if w_mp_full is not None else p
        mom = compute_moments_mp(vs_mp, p_for_moments, dps=dps)
        dn2 = dn2_moment_mp(mom["M1_norm"], mom["floor2"])

        # Dominance guard: never return worse than the best single member.
        # The best single member is always a feasible point of the simplex,
        # so any optimizer that truly minimises dn2 must have dn2 ≤ eps.
        if float(dn2) > eps:
            p = np.zeros(len(vs_mp))
            p[best_i] = 1.0
            mom = compute_moments_mp(vs_mp, p, dps=dps)
            dn2 = dn2_moment_mp(mom["M1_norm"], mom["floor2"])

        n_nz = int(np.sum(p > 1e-4))
        E_T  = float(np.dot(p, t_counts))

        per_rz.append({
            "n_mem"      : len(members),
            "n_nz"       : n_nz,
            "dn2"        : dn2,
            "M1_norm"    : mom["M1_norm"],
            "hull_dist"  : opt["hull_dist"],
            "floor2"     : mom["floor2"],
            "tr_C"       : mom["tr_C"],
            "eps"        : eps,
            "E_T"        : E_T,
            "enclosed"   : opt["enclosed"],
            "stage2_used": opt["stage2_used"],
            "p"          : p,
            "t_counts"   : t_counts,
            "U_list"     : U_list,
            "sequences"  : channel_seqs[k],  # gate strings (matrix order), index-matched to U_list
            "mags_f64"   : list(mags_f64),  # rotation vector magnitudes (index-matched to U_list)
            "w_mp"       : w_mp_full,        # mpmath mpf weights (None if mpmath refinement skipped)
        })

    t_opt = time.time() - t0

    if any(r is None for r in per_rz):
        return None

    # ---- aggregate U3 metrics ----
    # dn2_U3 ≈ dn2_rz1 + dn2_rz2 + dn2_rz3  (independent-channel leading order).
    # This holds because M1_U3 = R3·M1_3 + R2·M1_2 + R1·M1_1 = 0 when each Rz
    # first moment is zero (Rk are invertible SO(3) adjoint actions of Cliffords),
    # and because the per-shot Rz draws are independent so the cross-covariances
    # factor as E[e_i⊗e_j] = M1_i⊗M1_j ≈ 0 — the U3 covariance is block-diagonal.
    dn2_u3   = sum(float(r["dn2"])    for r in per_rz)
    floor2_u3= sum(float(r["floor2"]) for r in per_rz)
    m1_u3    = sum(float(r["M1_norm"])for r in per_rz)   # ∑|M1_k| (upper bound on |M1_U3|)
    eps_u3   = sum(r["eps"]           for r in per_rz)   # sum of best-single-member dn2 per Rz
    n_mem_u3 = sum(r["n_mem"]         for r in per_rz)
    # n_nz: max across channels (each ≤ 4 by Carathéodory — reports the per-Rz bound)
    n_nz_u3  = max(r["n_nz"]          for r in per_rz)
    # n_prod: product ensemble size — the U3 channel draws one member from each Rz
    # independently, so the composed ensemble has ≤ n_nz1 × n_nz2 × n_nz3 ≤ 64 elements.
    n_prod_u3 = 1
    for r in per_rz:
        n_prod_u3 *= r["n_nz"]
    E_T_u3   = sum(r["E_T"]           for r in per_rz)
    hull_u3  = max(r["hull_dist"]     for r in per_rz)
    enclosed_u3 = all(r["enclosed"]   for r in per_rz)

    gain = (eps_u3 / dn2_u3) if dn2_u3 > 0 else float("inf")
    hdroom = (dn2_u3 / floor2_u3) if floor2_u3 > 0 else float("inf")

    return {
        # per-Rz data (for cross-checks and matched-cost baseline)
        "per_rz"   : per_rz,
        # U3-level aggregates
        "n_mem"    : n_mem_u3,
        "n_nz"     : n_nz_u3,
        "n_prod"   : n_prod_u3,
        "dn2"      : dn2_u3,
        "m1"       : m1_u3,
        "hull_dist": hull_u3,
        "floor2"   : floor2_u3,
        "eps"      : eps_u3,
        "gain"     : gain,
        "headroom" : hdroom,
        "E_T"      : E_T_u3,
        "enclosed"    : enclosed_u3,
        "t_sweep"     : t_sweep,
        "t_opt"       : t_opt,
        "t_budget_rz" : t_budget,   # per-Rz T budget used (= t_total//3 when t_total given)
    }


# ---------------------------------------------------------------------------
# Hard-constraint optimizer: T_1 + T_2 + T_3 ≤ t_total
# ---------------------------------------------------------------------------

def optimize_u3_hardcap(theta, phi, lam, t_total,
                         n_keep=5, n_keep_total=10, dps=60,
                         enclosed_tol_ratio=0.1, fw_iter=2000,
                         t_min=0, stride=None):
    """
    Optimize the U3 mixed channel under the hard constraint T_1+T_2+T_3 ≤ t_total.

    Unlike optimize_u3 (equal-split t_total//3 per Rz), this allows flexible
    allocation across channels — one channel may use T=0 (Clifford) while the
    others use the saved budget.  Mirrors build_u3_channel_reduced from genlist.py
    but uses the generator-picture optimizer (Frank-Wolfe + LP on composite
    rotation vectors) instead of SLSQP.

    The final mixture has n_nz ≤ 4 composite U3 members (Carathéodory in R³).
    E[T] ≤ t_total by construction.

    Parameters
    ----------
    t_total        : hard total T-gate budget (T_1+T_2+T_3 ≤ t_total per shot)
    n_keep         : candidates per T-level per Rz (harvest diversity per channel)
    n_keep_total   : composite U3 members to retain per total-T level (FW pool size)
    t_min          : minimum T per Rz channel (0 = allow Clifford approximations)
    stride         : T-level step for harvest (default: max(1, t_total//30))
    dps            : mpmath decimal places

    Returns
    -------
    dict with keys:
      n_mem      : composite pool size (total selected composites)
      n_nz       : active-weight composites (≤ 4 by Carathéodory)
      dn2        : dn2_moment for the composite channel (mpmath float)
      m1         : |M1| for the composite channel
      hull_dist  : origin-to-hull distance for the composite pool
      floor2     : tr(C)/4 — incoherent floor
      eps        : dn2 of best single composite at T_total (the deterministic baseline)
      gain       : eps / dn2
      headroom   : dn2 / floor2
      E_T        : expected total T per shot
      enclosed   : True if origin is inside conv({composite v_i})
      t_sweep    : harvest time (s)
      t_opt      : optimization + moment time (s)
      U_list     : list of mpmath 2×2 composite U3 matrices (for cross-check)
      p          : float64 weight array (len = n_mem, indexed into U_list)
      w_mp       : mpmath weight list or None (indexed into U_list)
      t_counts   : list of total T-counts per composite member
      sequences  : Clifford+T gate string per composite (matrix order),
                   index-matched to U_list
      members    : the synthesized channel — active members only, as
                   [{"weight", "t_count", "sequence"}, ...] sorted by weight
      mags_f64   : list of composite rotation vector magnitudes
      angles     : Rz angles (for cross-check)
    """
    angles = rz_angles_for_u3(theta, phi, lam)
    if stride is None:
        stride = max(1, t_total // 30)

    # ---- harvest each Rz channel up to t_total ----
    # Each channel may receive any T ∈ [t_min, t_total]; the joint constraint
    # T_1+T_2+T_3 ≤ t_total is enforced in the composite pool step below.
    t0 = time.time()
    channels = []
    channel_seqs = []
    for angle in angles:
        members = harvest_rz(angle, t_min=t_min, t_budget=t_total,
                             n_keep=n_keep, stride=stride, dps=dps,
                             with_sequences=True)
        channels.append([(t, U) for t, _s, U in members])   # list of (t_count, U_mp)
        channel_seqs.append([sq for _t, sq, _U in members])
    t_sweep = time.time() - t0

    if any(not ch for ch in channels):
        return None

    # ---- build composite pool: T_1+T_2+T_3 ≤ t_total ----
    # For each total-T level, keep the n_keep_total best composites by process
    # fidelity |Tr(V†U)|²/4 (float64 selection, then mpmath composition).
    t0 = time.time()

    with mp.workdps(dps + 10):
        H_mp = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)
        rz_targets = [rz_exact_mp(a, dps=dps) for a in angles]
        target_u3_mp = rz_targets[2] * H_mp * rz_targets[1] * H_mp * rz_targets[0]

    target_u3_np = _mp_to_np(target_u3_mp)
    H_np = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    tH = target_u3_np.conj().T

    # Float64 versions of per-Rz members (for fast triple-loop selection)
    m_np = [[(t, _mp_to_np(U)) for t, U in ch] for ch in channels]

    # pool[t_sum] = min-heap of (process_fidelity, counter, i1, i2, i3)
    # Initial pass: keep n_preselect = 3*n_keep_total candidates per level by
    # process fidelity.  A second pass after mpmath rotation-vector computation
    # re-selects n_keep_total per level by actual |v| — this correctly handles
    # composites where per-Rz errors cancel in SO(3) (small |v| but only
    # moderate process fidelity), which would otherwise be evicted by higher-
    # fidelity balanced composites when n_keep_total is large.
    n_preselect = 3 * n_keep_total
    pool = {}
    ctr = 0
    for i1, (t1, U1_np) in enumerate(m_np[0]):
        HU1 = H_np @ U1_np
        for i2, (t2, U2_np) in enumerate(m_np[1]):
            if t1 + t2 > t_total:
                break
            U21 = U2_np @ HU1
            for i3, (t3, U3_np) in enumerate(m_np[2]):
                t_sum = t1 + t2 + t3
                if t_sum > t_total:
                    break
                U_comp = U3_np @ H_np @ U21
                fe = abs(np.trace(tH @ U_comp)) ** 2 / 4
                entry = (fe, ctr, i1, i2, i3)
                ctr += 1
                if t_sum not in pool:
                    pool[t_sum] = []
                h = pool[t_sum]
                if len(h) < n_preselect:
                    heapq.heappush(h, entry)
                elif fe > h[0][0]:
                    heapq.heapreplace(h, entry)

    if not pool:
        return None

    # Build mpmath composites for selected combinations
    composite = []   # list of (t_sum, U_comp_mp)
    composite_seqs = []   # matrix-order gate strings, index-matched to composite
    for t_level in sorted(pool.keys()):
        for fe, _, i1, i2, i3 in pool[t_level]:
            _t1, U1_mp = channels[0][i1]
            _t2, U2_mp = channels[1][i2]
            _t3, U3_mp = channels[2][i3]
            with mp.workdps(dps + 10):
                U_comp = U3_mp * H_mp * U2_mp * H_mp * U1_mp
            composite.append((t_level, U_comp))
            composite_seqs.append(channel_seqs[2][i3] + "H" + channel_seqs[1][i2]
                                  + "H" + channel_seqs[0][i1])

    t_counts_raw = [t for t, _ in composite]
    U_list_raw   = [U for _, U in composite]

    # ---- compute composite rotation vectors in mpmath ----
    # v_i = rotation_vector(V_U3† · U_composite_i) — error in the U3 frame
    vecs_mp_mags = rotation_vectors_mp(U_list_raw, target_u3_mp, dps=dps)
    vs_mp_raw   = [v for v, _ in vecs_mp_mags]
    mags_f64_raw = np.array([float(m) for _, m in vecs_mp_mags])

    # Second pass: re-select n_keep_total per level by |v| (not process fidelity).
    # Process fidelity is monotone in |v| only for small errors; composites where
    # per-Rz errors cancel in SO(3) can have small |v| but lower fidelity than
    # balanced composites, causing them to be evicted in the first pass.
    level_to_idxs = {}
    for idx, t_lev in enumerate(t_counts_raw):
        level_to_idxs.setdefault(t_lev, []).append(idx)

    keep = []
    for t_lev in sorted(level_to_idxs):
        idxs = level_to_idxs[t_lev]
        best = sorted(idxs, key=lambda i: mags_f64_raw[i])[:n_keep_total]
        keep.extend(best)

    composite = [composite[i]   for i in keep]
    vs_mp     = [vs_mp_raw[i]   for i in keep]
    U_list    = [U_list_raw[i]  for i in keep]
    t_counts  = [t_counts_raw[i] for i in keep]
    seqs      = [composite_seqs[i] for i in keep]
    mags_f64  = mags_f64_raw[np.array(keep, dtype=int)]

    # Sort composite pool by |v| ascending so that idx[0] in optimize_ms
    # (which uses np.arange(N) when top_k=None) correctly resolves to the
    # best-quality member for fw_tol and enclosed-threshold calibration.
    sort_idx = np.argsort(mags_f64)
    vs_mp     = [vs_mp[i]    for i in sort_idx]
    mags_f64  = mags_f64[sort_idx]
    U_list    = [U_list[i]   for i in sort_idx]
    t_counts  = [t_counts[i] for i in sort_idx]
    seqs      = [seqs[i]     for i in sort_idx]

    vs_f64 = np.array([[float(c) for c in v] for v in vs_mp], dtype=np.float64)

    best_i = 0   # mags_f64[0] is smallest after sort
    eps = float(mags_f64[0]) / 2

    # ---- optimize weights over composite pool ----
    # top_k=None: use all composites. Safe here because the pool is pre-sorted
    # by |v|, so idx[0] in optimize_ms gives the correct best-norm calibration.
    # The FW adaptive tol (min(1e-24, 1e-8·|v_best|²)) handles high T correctly.
    opt = optimize_ms(vs_f64, enclosed_tol_ratio=enclosed_tol_ratio,
                      fw_iter=fw_iter, top_k=None)
    p = opt["p"]

    # ---- mpmath weight refinement on the Carathéodory support ----
    w_mp_full = None
    if opt["stage2_used"]:
        support = [i for i in opt["support"] if p[i] > 1e-4]
        if 2 <= len(support) <= 4:
            vs_mp_s = [vs_mp[i] for i in support]
            w_mp = _refine_weights_mp(vs_mp_s, dps=dps)
            if w_mp is not None:
                w_mp_full = [mp.mpf(0)] * len(vs_mp)
                for j, gi in enumerate(support):
                    w_mp_full[gi] = w_mp[j]
                p = np.zeros(len(vs_mp))
                for j, gi in enumerate(support):
                    p[gi] = float(w_mp[j])

    # ---- moments and dn2 ----
    p_for_moments = w_mp_full if w_mp_full is not None else p
    mom = compute_moments_mp(vs_mp, p_for_moments, dps=dps)
    dn2 = dn2_moment_mp(mom["M1_norm"], mom["floor2"])

    # Dominance guard: mixture can never exceed the best single composite.
    if float(dn2) > eps:
        p = np.zeros(len(vs_mp))
        p[best_i] = 1.0
        w_mp_full = None
        mom = compute_moments_mp(vs_mp, p, dps=dps)
        dn2 = dn2_moment_mp(mom["M1_norm"], mom["floor2"])

    n_nz = int(np.sum(p > 1e-4))
    E_T  = float(np.dot(p, t_counts))

    t_opt = time.time() - t0

    gain   = (eps / float(dn2))     if float(dn2)         > 0 else float("inf")
    hdroom = (float(dn2) / float(mom["floor2"])) if float(mom["floor2"]) > 0 else float("inf")

    return {
        "n_mem"     : len(composite),
        "n_nz"      : n_nz,
        "dn2"       : float(dn2),
        "m1"        : float(mom["M1_norm"]),
        "hull_dist" : opt["hull_dist"],
        "floor2"    : float(mom["floor2"]),
        "eps"       : eps,
        "t_eps"     : t_counts[best_i],   # total T-count of the best single composite
        "gain"      : gain,
        "headroom"  : hdroom,
        "E_T"       : E_T,
        "enclosed"  : opt["enclosed"],
        "t_sweep"   : t_sweep,
        "t_opt"     : t_opt,
        # For cross-check and matched-cost baseline
        "U_list"    : U_list,
        "p"         : p,
        "w_mp"      : w_mp_full,
        "t_counts"  : t_counts,
        "sequences" : seqs,       # matrix-order gate strings, index-matched to U_list
        "members"   : _channel_members(p, w_mp_full, t_counts, seqs),
        "mags_f64"  : list(mags_f64),
        "angles"    : angles,
    }
