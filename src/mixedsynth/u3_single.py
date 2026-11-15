"""
Deterministic (single-sequence) U3 synthesis from three harvested Rz pieces.

U3(θ,φ,λ) = Rz3 · H · Rz2 · H · Rz1   (up to global phase), with
    rz1 = Rz(λ−π/2)  (rightmost, applied first),
    rz2 = Rz(θ),
    rz3 = Rz(φ+5π/2) (leftmost)
from :func:`mixedsynth.harvest.rz_angles_for_u3`.  A composite sequence is the
matrix-order string ``rz3 + "H" + rz2 + "H" + rz1``.

Functions
---------
harvest_u3            : the three Rz harvests, exactly as optimize_u3_hardcap
                        runs them (same calls, order and defaults).
synthesize_u3_single  : the single best composite with T1+T2+T3 ≤ t_total,
                        by exhaustive search over all harvested combinations.
enumerate_u3_splits   : one entry (per_split="best") or every combination
                        (per_split="all") for every split t1+t2+t3 == t_total,
                        with the splits that have no harvested candidates
                        reported in "missing".

No mixed channel, no Frank–Wolfe/LP.  Candidates are never filtered by
quality: low-T pieces (Cliffords, C·T) can be very poor approximations of
their Rz angle, and they are ranked like any other.

Error metric (Trasyn's, phase-insensitive)
------------------------------------------
    err = sqrt(max(0, 1 − (|tr(V†U)|/2)²))
with V the exact U3 target and U the composite.  Ranking is vectorised in
complex128; every reported err (and the final ranking) is evaluated in mpmath
at ``dps`` from the exact matrices of the gate strings
(``sequences.sequence_to_matrix(seq, dps)``), not from harvest_rz's stored
unitaries, whose T=0/T=1 entries are float64-accurate only.

Rotation vector "v"
-------------------
``mixedsynth.rotation_vector(U, V)`` with U the composite (exact matrix of
the sequence) and V the U3 target: the vector of strip_phase(V†U) =
exp(i(|v|/2) n·σ).  So |v| = 2·asin(err), and it is the composite vector that
optimize_u3_hardcap optimises over (hardcap builds it from the stored
harvested unitaries; the two agree to ~1e-15).

Ordering and ties
-----------------
Results are ranked by err; errors equal to within 10^-(dps//2) are ties,
broken by lower total T, then by the lexicographically smaller composite
string.  So output is deterministic for a given harvest (seed ``random``:
the norm-equation solver in the harvest uses it).
"""

import itertools
import time
from functools import lru_cache

import numpy as np
import mpmath as mp

from .harvest import harvest_rz, rz_angles_for_u3, rz_exact_mp
from .generators import rotation_vector
from .sequences import sequence_to_matrix

_H_NP = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_NEAR_TIE = 1e-10   # float64 err² window re-ranked in mpmath


def _mp_to_np(U_mp):
    return np.array([[complex(U_mp[i, j]) for j in range(2)]
                     for i in range(2)], dtype=complex)


# ---------------------------------------------------------------------------
# Harvest and shared helpers
# ---------------------------------------------------------------------------

def harvest_u3(theta, phi, lam, t_budget, n_keep=5, t_min=0, stride=None,
               dps=60, odd_t=False):
    """
    Harvest Rz candidates for the three U3 angles, in the same order and with
    the same arguments as :func:`optimize_u3_hardcap` (rz1, rz2, rz3; each
    ``harvest_rz(angle, t_min, t_budget, n_keep, stride, dps,
    with_sequences=True)``; stride default ``max(1, t_budget // 30)``).  With
    the same ``random`` seed it reproduces hardcap's harvest for
    ``t_total = t_budget``.  odd_t is passed to harvest_rz (odd T ≥ 3
    levels; False reproduces earlier harvests exactly, and True leaves the
    odd_t=False candidates unchanged, in the same order, as a subsequence).

    Returns a list of three lists of ``(t_count, seq, U_mp)`` triples, ordered
    rz1, rz2, rz3, each sorted by T ascending.  t_count is the actual T-count
    of seq.  harvest_rz may return candidates with T > t_budget (the gridsynth
    sweep overshoots); they are kept here and excluded by the budget later.

    Pass the result as ``candidates=`` to :func:`synthesize_u3_single` or
    :func:`enumerate_u3_splits` to reuse one harvest for several budgets.
    """
    if stride is None:
        stride = max(1, t_budget // 30)
    return [list(harvest_rz(a, t_min=t_min, t_budget=t_budget, n_keep=n_keep,
                            stride=stride, dps=dps, with_sequences=True,
                            odd_t=odd_t))
            for a in rz_angles_for_u3(theta, phi, lam)]


def _u3_target_mp(angles, dps):
    """Rz3·H·Rz2·H·Rz1 in mpmath — the same target optimize_u3_hardcap uses."""
    with mp.workdps(dps + 10):
        H = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)
        r = [rz_exact_mp(a, dps=dps) for a in angles]
        return r[2] * H * r[1] * H * r[0]


@lru_cache(maxsize=1 << 14)
def _piece_matrices(seq, dps):
    """
    (U_mp, U_np) of a candidate string at dps, cached: enumerate_u3_splits is
    called once per budget with the same harvest, so the same strings recur.
    The matrices are shared between callers and must not be modified.
    """
    Um = sequence_to_matrix(seq, dps=dps)
    return Um, _mp_to_np(Um)


def _pieces(candidates, t_max, dps):
    """
    Per channel, the candidates with T ≤ t_max as (t, seq, U_mp, U_np), with
    U_mp the exact matrix of seq at dps.  Duplicate strings are dropped.
    """
    out = []
    for ch in candidates:
        seen, lst = set(), []
        for t, s, _U in ch:
            if t <= t_max and s not in seen:
                seen.add(s)
                lst.append((t, s) + _piece_matrices(s, dps))
        out.append(lst)
    return out


@lru_cache(maxsize=1 << 14)
def _left_product(seq3, seq2, dps):
    """
    U3·H·U2·H in mpmath for candidate strings seq3, seq2, evaluated left to
    right exactly as in ``_Scorer.score`` (so the composite U3·H·U2·H·U1 is
    the same to the last bit); cached because the same (Rz3, Rz2) pair recurs
    over Rz1 choices and budgets. Shared result: do not modify.
    """
    with mp.workdps(dps + 10):
        H = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)
        return _piece_matrices(seq3, dps)[0] * H * _piece_matrices(seq2, dps)[0] * H


class _Scorer:
    """Exact (mpmath) error, rank key and entry construction."""

    def __init__(self, angles, dps):
        self.angles = angles
        self.dps = dps
        self.V = _u3_target_mp(angles, dps)
        self.tH = _mp_to_np(self.V).conj().T
        self.tol = mp.mpf(10) ** (-(dps // 2))
        with mp.workdps(dps + 10):
            self.H = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)

    def score(self, picks):
        """picks = (p1, p2, p3) pieces; returns (key, err_mp, U_mp, picks)."""
        with mp.workdps(self.dps + 10):
            U = _left_product(picks[2][1], picks[1][1], self.dps) * picks[0][2]
            V = self.V
            tr = sum(V[i, j].conjugate() * U[i, j]
                     for i in range(2) for j in range(2))
            err = mp.sqrt(max(mp.mpf(0), 1 - (abs(tr) / 2) ** 2))
            seq = picks[2][1] + "H" + picks[1][1] + "H" + picks[0][1]
            tsum = picks[0][0] + picks[1][0] + picks[2][0]
            return ((int(mp.nint(err / self.tol)), tsum, seq), err, U, picks)

    def entry(self, scored, with_v=True):
        (_k, tsum, seq), err, U, picks = scored
        rz = [{"seq": p[1], "t_count": p[0], "angle": float(self.angles[k])}
              for k, p in enumerate(picks)]
        e = {"sequence": seq, "t_count": tsum,
             "t_split": [c["t_count"] for c in rz], "rz": rz,
             "err": float(err)}
        if with_v:
            with mp.workdps(self.dps + 10):
                e["v"] = [float(c) for c in rotation_vector(U, self.V)]
        return e


def _float_err2(sc, P1, P2, P3):
    """
    complex128 err² for every (i1, i2, i3) in P1×P2×P3 (lists of pieces),
    shape (n1, n2, n3).
    """
    U1 = np.array([p[3] for p in P1])
    U2 = np.array([p[3] for p in P2])
    U3 = np.array([p[3] for p in P3])
    A3 = np.einsum("ij,kjl,lm->kim", sc.tH, U3, _H_NP)        # tH·U3·H
    M12 = np.einsum("bij,jk,akl->abil", U2, _H_NP, U1)        # U2·H·U1
    tr = np.einsum("cij,abji->abc", A3, M12)
    return 1.0 - (np.abs(tr) / 2) ** 2


# ---------------------------------------------------------------------------
# Best single sequence, T1+T2+T3 ≤ t_total
# ---------------------------------------------------------------------------

def _float_search(sc, chans, t_total, shortlist):
    """
    Score every combination with t1+t2+t3 ≤ t_total in complex128.
    Returns (n_combos, [(i1, i2, i3), ...]) for the ``shortlist`` best (float
    order: err², T, indices) plus all float near-ties of the best.
    """
    t1, t2, t3 = (np.array([p[0] for p in ch], dtype=int) for ch in chans)
    U1, U2, U3 = (np.array([p[3] for p in ch]) for ch in chans)
    A3 = np.einsum("ij,kjl,lm->kim", sc.tH, U3, _H_NP)
    HU1 = np.einsum("ij,kjl->kil", _H_NP, U1)

    e2s, tss, idxs = [], [], []
    for i1 in range(len(U1)):
        rem = t_total - t1[i1]
        ok2 = np.flatnonzero(t2 <= rem)
        if ok2.size == 0:
            continue
        ok3 = np.flatnonzero(t3 <= rem - t2[ok2].min())
        if ok3.size == 0:
            continue
        M12 = np.einsum("kij,jl->kil", U2[ok2], HU1[i1])
        tr = np.einsum("aij,bji->ab", A3[ok3], M12)            # (n3, n2)
        tsum = t1[i1] + t2[ok2][None, :] + t3[ok3][:, None]
        a3, a2 = np.nonzero(tsum <= t_total)
        if a3.size == 0:
            continue
        e2s.append(1.0 - (np.abs(tr[a3, a2]) / 2) ** 2)
        tss.append(tsum[a3, a2])
        idxs.append(np.stack([np.full(a3.size, i1), ok2[a2], ok3[a3]], 1))

    if not e2s:
        return 0, []
    err2 = np.concatenate(e2s)
    tsum = np.concatenate(tss)
    idx = np.concatenate(idxs)
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0], tsum, err2))
    n_near = int(np.sum(err2 <= err2[order[0]] + _NEAR_TIE))
    take = order[:max(shortlist, n_near)]
    return err2.size, [tuple(int(x) for x in idx[i]) for i in take]


def synthesize_u3_single(theta, phi, lam, t_total, n_keep=5, t_min=0,
                         stride=None, dps=60, pool=False, pool_size=128,
                         candidates=None, odd_t=False):
    """
    Best single Clifford+T sequence Rz3·H·Rz2·H·Rz1 for U3(θ,φ,λ) with
    T1+T2+T3 ≤ t_total.

    Harvest: identical to :func:`optimize_u3_hardcap` (see
    :func:`harvest_u3`; same defaults n_keep=5, t_min=0,
    stride=max(1, t_total // 30), dps=60), so with the same ``random`` seed
    and arguments both functions see the same Rz candidates.

    Search: every combination (i1, i2, i3) of harvested candidates, one per
    channel, with t1+t2+t3 ≤ t_total, is scored in complex128 by err (module
    docstring).  The best max(1, pool_size if pool else 1) combinations plus
    every float near-tie of the best (err² within 1e-10) are re-scored in
    mpmath and ranked by (err, total T, sequence string), ties being errors
    equal to within 10^-(dps//2).  Nothing is filtered by error.

    Parameters
    ----------
    theta, phi, lam : U3 angles (Trasyn convention,
        U3 = [[cos(θ/2), −e^{iλ} sin(θ/2)], [e^{iφ} sin(θ/2), e^{i(φ+λ)} cos(θ/2)]]).
    t_total    : hard T budget for the whole sequence.
    n_keep, t_min, stride, dps, odd_t : harvest arguments (see harvest_u3).
    pool       : if True, also return the ranked "pool".
    pool_size  : number of best combinations in "pool" (winner included).
    candidates : optional output of harvest_u3; no harvest is run
                 (n_keep/t_min/stride/odd_t ignored), and only its
                 candidates with T ≤ t_total are used.

    Returns
    -------
    None if no combination fits the budget (e.g. 3·t_min > t_total), else
      sequence     : composite string rz3 + "H" + rz2 + "H" + rz1 (matrix
                     order, pyLIQTR tokens, correct up to global phase)
      t_count      : total T-count (= sum(t_split) ≤ t_total)
      t_split      : [t1, t2, t3]
      rz           : [rz1, rz2, rz3], each {"seq", "t_count", "angle"} (same
                     shape as optimize_u3_hardcap's "components" entries;
                     angle raw from rz_angles_for_u3, not reduced)
      err          : float (mpmath-evaluated)
      v            : [vx, vy, vz] floats, rotation_vector(U, V); |v| = 2 asin(err)
      n_candidates : number of combinations searched (within budget)
      n_harvested  : [n1, n2, n3] distinct candidates per channel with T ≤ t_total
      t_harvest    : seconds harvesting (0.0 if candidates were given)
      t_search     : seconds searching and refining
      pool         : (pool=True only) best min(pool_size, n_candidates)
                     combinations, sorted by the same key, each
                     {"sequence", "t_count", "t_split", "rz", "err"};
                     pool[0] is the returned result.
    """
    angles = rz_angles_for_u3(theta, phi, lam)
    t0 = time.time()
    if candidates is None:
        candidates = harvest_u3(theta, phi, lam, t_total, n_keep=n_keep,
                                t_min=t_min, stride=stride, dps=dps,
                                odd_t=odd_t)
        t_harvest = time.time() - t0
    else:
        t_harvest = 0.0

    t0 = time.time()
    chans = _pieces(candidates, t_total, dps)
    if any(not ch for ch in chans):
        return None
    sc = _Scorer(angles, dps)
    n_combos, short = _float_search(sc, chans, t_total,
                                    max(1, pool_size if pool else 1))
    if n_combos == 0:
        return None
    ranked = sorted((sc.score((chans[0][a], chans[1][b], chans[2][c]))
                     for a, b, c in short), key=lambda r: r[0])

    out = sc.entry(ranked[0])
    out.update({
        "n_candidates": n_combos,
        "n_harvested": [len(ch) for ch in chans],
        "t_harvest": t_harvest,
        "t_search": time.time() - t0,
    })
    if pool:
        out["pool"] = [sc.entry(r, with_v=False) for r in ranked[:pool_size]]
    return out


# ---------------------------------------------------------------------------
# Every split t1+t2+t3 == t_total
# ---------------------------------------------------------------------------

def enumerate_u3_splits(theta, phi, lam, t_total, per_split="best",
                        piece="all", n_keep=5, t_min=0, stride=None, dps=60,
                        candidates=None, odd_t=False):
    """
    Composite sequences Rz3·H·Rz2·H·Rz1 for U3(θ,φ,λ) for every split
    (t1, t2, t3) of nonnegative integers with t1+t2+t3 == t_total and each
    t_k ≥ t_min, from one harvest (harvest_u3 with t_budget = t_total, i.e.
    the same harvest as optimize_u3_hardcap / synthesize_u3_single for the
    same seed and arguments).

    Splits are visited in lexicographic order of (t1, t2, t3).  A piece with
    T-count t_k must be a harvested candidate for channel k whose actual
    T-count is exactly t_k.  By default the gridsynth sweep only yields even
    T ≥ 2 (plus the explicit T=0 Cliffords and T=1 C·T circuits), so splits
    with an odd t_k ≥ 3 normally have no candidate; odd_t=True harvests odd
    levels T ≥ 3 too (see harvest_rz).  Splits without a candidate are not
    dropped silently but listed in "missing".

    per_split : "best" — one entry per available split: the combination of
                that split's candidates with the smallest err (ties as in the
                module docstring).
                "all"  — every combination of that split's candidates, each
                its own entry, sorted within the split by (err, sequence).
    piece     : "all"  — use every harvested candidate at each (channel, T).
                "best" — first reduce each (channel, T) to its single
                candidate closest to Rz(angle) (min phase-insensitive error,
                ties by string), then combine.  Note the per-split best
                composite need not use the per-piece best candidates, because
                piece errors can partly cancel.
    candidates: optional harvest_u3 output to reuse; n_keep/t_min/stride/
                odd_t are then ignored for the harvest (t_min still
                restricts splits).
    odd_t     : harvest odd T ≥ 3 levels too (see harvest_u3).

    Returns a dict
      entries   : list of {"sequence", "t_count" (== t_total), "t_split",
                  "rz", "err", "v"} (fields as in synthesize_u3_single)
      missing   : list of {"t_split": [t1, t2, t3], "channels": [k, ...]}
                  for splits skipped because channel k (0 = rz1, 1 = rz2,
                  2 = rz3) has no candidate with exactly T = t_k
      n_splits  : number of splits considered ((t_total+2 choose 2) when
                  t_min = 0)
      available_t : per channel, sorted list of T-counts ≤ t_total that have
                  at least one candidate
      n_per_t   : per channel, {T: number of candidates used at that T}
      t_harvest, t_search : seconds
    """
    if per_split not in ("best", "all"):
        raise ValueError("per_split must be 'best' or 'all'")
    if piece not in ("best", "all"):
        raise ValueError("piece must be 'best' or 'all'")

    angles = rz_angles_for_u3(theta, phi, lam)
    t0 = time.time()
    if candidates is None:
        candidates = harvest_u3(theta, phi, lam, t_total, n_keep=n_keep,
                                t_min=t_min, stride=stride, dps=dps,
                                odd_t=odd_t)
        t_harvest = time.time() - t0
    else:
        t_harvest = 0.0

    t0 = time.time()
    chans = _pieces(candidates, t_total, dps)
    by_t = []
    for k, ch in enumerate(chans):
        g = {}
        for p in ch:
            g.setdefault(p[0], []).append(p)
        if piece == "best":
            V = rz_exact_mp(angles[k], dps=dps)
            for t, lst in g.items():
                def key(p):
                    with mp.workdps(dps + 10):
                        tr = sum(V[i, j].conjugate() * p[2][i, j]
                                 for i in range(2) for j in range(2))
                        return (float(1 - (abs(tr) / 2) ** 2), p[1])
                g[t] = [min(lst, key=key)]
        by_t.append(g)

    sc = _Scorer(angles, dps)
    entries, missing, n_splits = [], [], 0
    for t1 in range(t_min, t_total + 1):
        for t2 in range(t_min, t_total - t1 + 1):
            t3 = t_total - t1 - t2
            if t3 < t_min:
                continue
            n_splits += 1
            split = (t1, t2, t3)
            lack = [k for k in range(3) if split[k] not in by_t[k]]
            if lack:
                missing.append({"t_split": list(split), "channels": lack})
                continue
            P = [by_t[k][split[k]] for k in range(3)]
            if per_split == "all":
                combos = list(itertools.product(*P))
            else:
                e2 = _float_err2(sc, *P).ravel()
                order = np.argsort(e2, kind="stable")
                n_near = int(np.sum(e2 <= e2[order[0]] + _NEAR_TIE))
                shape = (len(P[0]), len(P[1]), len(P[2]))
                combos = [tuple(P[k][i] for k, i in
                                enumerate(np.unravel_index(j, shape)))
                          for j in order[:n_near]]
            ranked = sorted((sc.score(c) for c in combos), key=lambda r: r[0])
            if per_split == "best":
                ranked = ranked[:1]
            entries.extend(sc.entry(r) for r in ranked)

    return {
        "entries": entries,
        "missing": missing,
        "n_splits": n_splits,
        "available_t": [sorted(g) for g in by_t],
        "n_per_t": [{t: len(v) for t, v in sorted(g.items())} for g in by_t],
        "t_harvest": t_harvest,
        "t_search": time.time() - t0,
    }
