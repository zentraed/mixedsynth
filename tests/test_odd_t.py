"""
Odd T-counts (T >= 3) in the Rz harvest (harvest_rz(..., odd_t=True)).

The grid solver only solves the det = 1 problem, so it yields even T.  Odd
candidates for Rz(θ) are T·V with V an even-T candidate for Rz(θ − π/4)
(T†·V' with V' ≈ Rz(θ + π/4) as a fallback for empty levels).  These tests
check that the odd candidates are exact, odd, T-optimal and deduplicated,
that odd_t=True leaves the odd_t=False harvest (and the global random
stream) unchanged, that odd splits are filled, and compare with Haskell
gridsynth -p on the fixture tests/data/gridsynth_rz_phase.tsv.
"""

import math
import os
import random

import mpmath as mp
import numpy as np
import pytest

from mixedsynth import (enumerate_u3_splits, harvest_rz, harvest_u3, hst,
                        synthesize_u3_single)
from mixedsynth.harvest import (_phase_free_key, _prepend_t, rz_exact_mp)
from mixedsynth.sequences import sequence_to_matrix, t_count

DPS = 40
ANGLES = [0.7, 1.3, -0.5, 2.1, 0.3, 9.30314937273944, -6.1]


def _harvest(angle, t_budget, odd_t, seed=1234, **kw):
    random.seed(seed)
    return harvest_rz(angle, t_budget=t_budget, dps=DPS, with_sequences=True,
                      odd_t=odd_t, **kw)


def _err(U, angle):
    """Phase-insensitive error sqrt(1 - (|tr(Rz†U)|/2)^2) in mpmath."""
    with mp.workdps(DPS + 10):
        W = rz_exact_mp(angle, DPS).H * U
        tr = W[0, 0] + W[1, 1]
        return mp.sqrt(max(mp.mpf(0), 1 - (abs(tr) / 2) ** 2))


def _det_omega_exponent(U):
    """k with det(U) = ω^k exactly (to 1e-40), U a Clifford+T matrix."""
    with mp.workdps(DPS + 10):
        d = mp.det(U)
        for k in range(8):
            if abs(d - mp.expjpi(mp.mpf(k) / 4)) < mp.mpf(10) ** -40:
                return k
    raise AssertionError("det is not a power of omega: %s" % d)


def _is_odd_level(t):
    return t >= 3 and t % 2 == 1


# ---------------------------------------------------------------------------
# Backward compatibility and RNG isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", ANGLES[:4])
def test_odd_t_false_is_default(angle):
    a = [(t, s) for t, s, _ in _harvest(angle, 10, odd_t=False)]
    random.seed(1234)
    b = [(t, s) for t, s, _ in harvest_rz(angle, t_budget=10, dps=DPS,
                                          with_sequences=True)]
    assert a == b
    assert not any(_is_odd_level(t) for t, _ in a)


@pytest.mark.parametrize("angle", ANGLES)
def test_even_candidates_unchanged_in_order(angle):
    """odd_t=True adds candidates only at odd T >= 3; dropping them gives the
    odd_t=False harvest exactly (strings, order and matrices)."""
    off = _harvest(angle, 12, odd_t=False)
    on = _harvest(angle, 12, odd_t=True)
    rest = [c for c in on if not _is_odd_level(c[0])]
    assert [(t, s) for t, s, _ in rest] == [(t, s) for t, s, _ in off]
    for (_, _, U1), (_, _, U2) in zip(rest, off):
        assert U1 == U2
    assert len(on) > len(off)


def test_global_random_stream_unchanged():
    random.seed(99)
    harvest_rz(0.7, t_budget=9, dps=DPS, odd_t=False)
    s_off = random.getstate()
    random.seed(99)
    harvest_rz(0.7, t_budget=9, dps=DPS, odd_t=True)
    assert random.getstate() == s_off


def test_harvest_u3_even_subset_and_later_channels_unchanged():
    """Channels are harvested in sequence from one global random stream; the
    odd levels of rz1 must not perturb rz2/rz3's even candidates."""
    random.seed(1234)
    off = harvest_u3(1.1, 2.3, 0.4, 8, dps=DPS)
    random.seed(1234)
    on = harvest_u3(1.1, 2.3, 0.4, 8, dps=DPS, odd_t=True)
    for ch_off, ch_on in zip(off, on):
        assert ([(t, s) for t, s, _ in ch_on if not _is_odd_level(t)]
                == [(t, s) for t, s, _ in ch_off])
        assert any(_is_odd_level(t) for t, _, _ in ch_on)


# harvest_rz(0.7, t_budget=7, n_keep=3, dps=40, odd_t=True) after
# random.seed(1234): its odd candidates, identical in the gridsynth and
# ttrain environments.
PINNED_ODD = [(3, 'STHTSHTYS'), (3, 'SHTSHTHTZH'), (5, 'THTHTHTSHTX'),
              (7, 'HTHTHTHTSHTHTHTHZ'), (7, 'SHTHTHTSHTHTSHTHTHSd'),
              (9, 'THTHTHTHTSHTHTHTHTS'), (9, 'THTSHTHTSHTHTHTHTSHTSHS'),
              (9, 'STHTHTHTSHTHTSHTHTHTZ')]


def test_odd_candidates_pinned():
    got = [(t, s) for t, s, _ in _harvest(0.7, 7, odd_t=True, n_keep=3)
           if _is_odd_level(t)]
    assert got == PINNED_ODD


# ---------------------------------------------------------------------------
# Exactness of the odd candidates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", ANGLES)
def test_odd_candidates_exact(angle):
    """Each odd candidate: odd ω-exponent of det, t_count = actual minimal
    T-count of its string, stored matrix = string matrix up to phase, and
    error vs Rz(θ) = error of T†U vs Rz(θ − π/4)."""
    odd = [c for c in _harvest(angle, 12, odd_t=True) if _is_odd_level(c[0])]
    assert odd
    with mp.workdps(DPS + 10):
        Tinv = mp.matrix([[1, 0], [0, mp.expjpi(mp.mpf(-1) / 4)]])
    for t, seq, U in odd:
        assert t_count(seq) == t
        assert hst.t_count(hst.normal_form(hst.from_pyliqtr(seq))) == t
        M = sequence_to_matrix(seq, dps=DPS)
        assert _det_omega_exponent(M) % 2 == 1
        assert _det_omega_exponent(U) % 2 == 1
        with mp.workdps(DPS + 10):
            W = M.H * U
            assert 2 - abs(W[0, 0] + W[1, 1]) < mp.mpf(10) ** -35, seq
            V = Tinv * U
            assert _det_omega_exponent(V) % 2 == 0
            assert abs(_err(U, angle) - _err(V, angle - mp.pi / 4)) \
                < mp.mpf(10) ** -35


@pytest.mark.parametrize("angle", ANGLES)
def test_no_duplicate_unitaries(angle):
    cands = _harvest(angle, 12, odd_t=True)
    with mp.workdps(DPS + 10):
        scale = mp.mpf(10) ** ((DPS + 10) // 2)
        odd_keys = [_phase_free_key(U, scale) for t, _s, U in cands
                    if _is_odd_level(t)]
        all_keys = {_phase_free_key(U, scale) for t, _s, U in cands
                    if not _is_odd_level(t)}
    assert len(set(odd_keys)) == len(odd_keys)
    assert not set(odd_keys) & all_keys


def test_prepend_t_merges_diagonals():
    cases = {"HT": "THT", "THT": "SHT", "TSH": "ZH",
             "STH": "ZH", "SdH": "SdTH", "ZTH": "SdH", "": "T"}
    for seq, want in cases.items():
        got = _prepend_t(seq)
        assert got == want, (seq, got)
        with mp.workdps(DPS + 10):
            d = sequence_to_matrix("T" + seq, dps=DPS) - \
                sequence_to_matrix(got, dps=DPS)
            assert mp.mnorm(d, 1) < mp.mpf(10) ** -40
    assert _prepend_t("H", 7) == "SdTH"


def test_odd_levels_respect_t_min_and_stride():
    # (the even sweep may return T < t_min; odd candidates never do)
    for t_min in (3, 4, 5):
        for t, _s, _U in _harvest(0.7, 12, odd_t=True, t_min=t_min):
            assert t >= t_min or t % 2 == 0
    # stride 2 from t_min=3: nominal odd-level targets 2, 4, ..., 10
    odd = {t for t, _s, _U in _harvest(1.3, 11, odd_t=True, stride=2)
           if _is_odd_level(t)}
    assert {3, 5, 7, 9, 11} <= odd


def test_without_sequences_matches():
    a = _harvest(2.1, 9, odd_t=True)
    random.seed(1234)
    b = harvest_rz(2.1, t_budget=9, dps=DPS, odd_t=True)
    assert [(t, U) for t, _s, U in a] == b


# ---------------------------------------------------------------------------
# U3: odd splits filled
# ---------------------------------------------------------------------------

U3_TARGETS = [(1.1, 2.3, 0.4), (6.16277, 4.30739, 4.08696),
              (3.05259, 5.58882, 5.86877), (5.00345, 1.44917, 0.32686)]


@pytest.mark.parametrize("target", U3_TARGETS)
def test_enumerate_splits_no_missing(target):
    random.seed(11)
    off = enumerate_u3_splits(*target, 10, dps=DPS)
    random.seed(11)
    on = enumerate_u3_splits(*target, 10, dps=DPS, odd_t=True)
    assert off["n_splits"] == on["n_splits"] == 66
    assert len(off["missing"]) == 42
    assert on["missing"] == []
    assert all(e["t_count"] == 10 for e in on["entries"])
    # every split available without odd_t gives the same best entry
    best_on = {tuple(e["t_split"]): e for e in on["entries"]}
    for e in off["entries"]:
        assert best_on[tuple(e["t_split"])]["sequence"] == e["sequence"]


def test_synthesize_single_odd_t_not_worse():
    for target in U3_TARGETS[:2]:
        random.seed(5)
        off = synthesize_u3_single(*target, 7, dps=DPS)
        random.seed(5)
        on = synthesize_u3_single(*target, 7, dps=DPS, odd_t=True)
        assert on["err"] <= off["err"] + 1e-12
        assert on["t_count"] <= 7


# ---------------------------------------------------------------------------
# Haskell gridsynth -p fixture
# ---------------------------------------------------------------------------

def _load_fixture():
    path = os.path.join(os.path.dirname(__file__), "data",
                        "gridsynth_rz_phase.tsv")
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, e, mode, seq = line.rstrip("\n").split("\t")
            rows.append((float(a), float(e), mode, seq))
    return rows


FIXTURE = _load_fixture()

# Minimal T-count of each gridsynth -p output (angle, eps) -> T.
EXPECTED_T_P = {
    (0.7, 0.3): 1, (0.7, 0.1): 1, (0.7, 0.03): 12, (0.7, 0.01): 17,
    (0.7, 0.003): 22,
    (1.3, 0.3): 0, (1.3, 0.1): 7, (1.3, 0.03): 15, (1.3, 0.01): 19,
    (1.3, 0.003): 25,
    (-0.5, 0.3): 0, (-0.5, 0.1): 9, (-0.5, 0.03): 16, (-0.5, 0.01): 20,
    (-0.5, 0.003): 22,
    (2.1, 0.3): 0, (2.1, 0.1): 7, (2.1, 0.03): 11, (2.1, 0.01): 20,
    (2.1, 0.003): 23,
    (0.3, 0.3): 0, (0.3, 0.1): 9, (0.3, 0.03): 14, (0.3, 0.01): 20,
    (0.3, 0.003): 20,
}


def test_fixture_complete():
    assert len(FIXTURE) == 50
    assert {(a, e) for a, e, m, _ in FIXTURE if m == "p"} == set(EXPECTED_T_P)


@pytest.mark.parametrize("row", FIXTURE,
                         ids=["%s_%s_%s" % r[:3] for r in FIXTURE])
def test_gridsynth_fixture_parity(row):
    """gridsynth without -p is phase-exact (det 1, even T); with -p, the
    normal-form T-count parity is the det ω-exponent parity, and odd counts
    occur where expected.  Errors are within epsilon."""
    angle, eps, mode, seq = row
    h = hst.from_gridsynth(seq)
    nf = hst.normal_form(h)
    t = hst.t_count(nf)
    assert t == hst.t_count(h)          # gridsynth output is T-optimal
    M = hst.hst_to_matrix(h, dps=DPS)
    k = _det_omega_exponent(M)
    assert k % 2 == t % 2
    if mode == "exact":
        assert k == 0 and t % 2 == 0
    else:
        assert t == EXPECTED_T_P[(angle, eps)]
    assert _err(M, angle) <= eps


ODD_P_ROWS = [(a, e) for (a, e), t in EXPECTED_T_P.items() if _is_odd_level(t)]


@pytest.mark.parametrize("ae", ODD_P_ROWS, ids=["%s_%s" % x for x in ODD_P_ROWS])
def test_mixedsynth_has_odd_candidate_at_gridsynth_level(ae):
    """For every odd T >= 3 that gridsynth -p returns, the odd_t harvest with
    that budget has a candidate at exactly that T (quality is not asserted:
    the searches differ)."""
    angle, eps = ae
    T = EXPECTED_T_P[ae]
    cands = _harvest(angle, T, odd_t=True)
    assert any(t == T for t, _s, _U in cands)
    assert not any(t == T for t, _s, _U in _harvest(angle, T, odd_t=False))


def test_hardcap_odd_t_threads_through():
    from mixedsynth import optimize_u3_hardcap
    random.seed(3)
    r = optimize_u3_hardcap(1.1, 2.3, 0.4, 6, dps=DPS, odd_t=True)
    assert r is not None
    for seq, t, comp in zip(r["sequences"], r["t_counts"], r["components"]):
        assert seq == comp[2]["seq"] + "H" + comp[1]["seq"] + "H" + comp[0]["seq"]
        assert t == sum(c["t_count"] for c in comp) == t_count(seq) <= 6
    assert any(_is_odd_level(c["t_count"]) for comp in r["components"]
               for c in comp)
