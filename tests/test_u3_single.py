"""
Tests for synthesize_u3_single / enumerate_u3_splits / harvest_u3.

A composite is the matrix-order string rz3 + "H" + rz2 + "H" + rz1
(Rz3 · H · Rz2 · H · Rz1).  The norm-equation solver uses Python's `random`,
so every harvest is seeded.
"""

import itertools
import math
import random
from math import comb

import numpy as np
import pytest

from mixedsynth import (
    enumerate_u3_splits, harvest_u3, optimize_u3_hardcap, rz_angles_for_u3,
    synthesize_u3_single,
)
from mixedsynth.hst import from_pyliqtr, hst_equal
from mixedsynth.sequences import sequence_to_matrix, t_count

SEED = 1234
TARGETS = [(1.1, 2.3, 0.4), (0.3, -1.2, 2.9), (2.7, 0.5, -0.8)]
BUDGETS = (0, 1, 2, 3, 5, 10)


def u3(theta, phi, lam):
    """Trasyn's U3 convention."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def err_of(seq, target):
    U = sequence_to_matrix(seq)
    ov = abs(np.trace(u3(*target).conj().T @ U)) / 2
    return math.sqrt(max(0.0, 1 - ov ** 2))


def harvest(target, T):
    random.seed(SEED)
    return harvest_u3(*target, T)


@pytest.fixture(scope="module")
def harvests():
    return {tg: harvest(tg, 10) for tg in TARGETS}


@pytest.fixture(scope="module", params=[(tg, T) for tg in TARGETS for T in BUDGETS],
                ids=lambda p: "%s-T%d" % (p[0], p[1]))
def single(request):
    tg, T = request.param
    random.seed(SEED)
    r = synthesize_u3_single(*tg, T, pool=True, pool_size=32)
    assert r is not None
    return tg, T, r


def check_entry(e, target, t_total=None, exact_total=False):
    rz1, rz2, rz3 = e["rz"]
    assert e["sequence"] == rz3["seq"] + "H" + rz2["seq"] + "H" + rz1["seq"]
    assert e["t_split"] == [rz1["t_count"], rz2["t_count"], rz3["t_count"]]
    assert sum(e["t_split"]) == e["t_count"] == t_count(e["sequence"])
    if t_total is not None:
        assert (e["t_count"] == t_total) if exact_total else (e["t_count"] <= t_total)
    for c in e["rz"]:
        assert t_count(c["seq"]) == c["t_count"]
    assert [c["angle"] for c in e["rz"]] == [float(a) for a in rz_angles_for_u3(*target)]
    assert e["err"] == pytest.approx(err_of(e["sequence"], target), abs=1e-12)
    if "v" in e:
        assert math.hypot(*e["v"]) == pytest.approx(2 * math.asin(e["err"]), abs=1e-12)
    parts = [from_pyliqtr(c["seq"]) for c in e["rz"]]
    assert hst_equal(from_pyliqtr(e["sequence"]),
                     parts[2] + "H" + parts[1] + "H" + parts[0])


# ---------------------------------------------------------------------------
# synthesize_u3_single
# ---------------------------------------------------------------------------

def test_single_consistency(single):
    tg, T, r = single
    check_entry(r, tg, T)
    assert set(r) >= {"sequence", "t_count", "t_split", "rz", "err", "v",
                      "n_candidates", "n_harvested", "t_harvest", "t_search", "pool"}
    for e in r["pool"]:
        check_entry(e, tg, T)


def test_single_pool_sorted_and_first_is_result(single):
    _tg, _T, r = single
    p = r["pool"]
    assert 1 <= len(p) == min(32, r["n_candidates"])
    errs = [e["err"] for e in p]
    assert all(a <= b + 1e-15 for a, b in zip(errs, errs[1:]))
    assert {k: p[0][k] for k in p[0]} == {k: r[k] for k in p[0]}
    # distinct piece combinations (different pieces can concatenate to the
    # same composite string, e.g. rz2 = "" with rz1 = "S" vs rz2 = "S", rz1 = "")
    assert len({tuple(c["seq"] for c in e["rz"]) for e in p}) == len(p)


def test_budget_zero_is_clifford(harvests):
    for tg in TARGETS:
        random.seed(SEED)
        r = synthesize_u3_single(*tg, 0)
        assert r["t_count"] == 0 and r["t_split"] == [0, 0, 0]
        assert "T" not in r["sequence"]
        assert r["n_candidates"] == 5 ** 3


@pytest.mark.parametrize("T", [3, 4])
@pytest.mark.parametrize("tg", TARGETS)
def test_single_is_brute_force_minimum(tg, T):
    cands = harvest(tg, T)
    r = synthesize_u3_single(*tg, T, candidates=cands)
    best, n = None, 0
    chans = [[(t, s) for t, s, _U in ch if t <= T] for ch in cands]
    for (t1, s1), (t2, s2), (t3, s3) in itertools.product(*chans):
        if t1 + t2 + t3 > T:
            continue
        n += 1
        e = err_of(s3 + "H" + s2 + "H" + s1, tg)
        best = e if best is None else min(best, e)
    assert r["n_candidates"] == n
    assert r["err"] == pytest.approx(best, abs=1e-12)


@pytest.mark.parametrize("tg", TARGETS)
def test_single_err_nonincreasing_on_fixed_harvest(tg, harvests):
    """
    With one fixed harvest the feasible set only grows with t_total, so err is
    non-increasing.  (Separate calls harvest separately: the random stream
    and the T levels swept differ, so that is not guaranteed across calls.)
    """
    errs = [synthesize_u3_single(*tg, T, candidates=harvests[tg])["err"]
            for T in range(11)]
    assert all(b <= a for a, b in zip(errs, errs[1:]))


@pytest.mark.parametrize("T", [2, 6])
def test_single_matches_hardcap_harvest_and_eps(T):
    tg = TARGETS[0]
    random.seed(SEED)
    h = optimize_u3_hardcap(*tg, t_total=T)
    random.seed(SEED)
    r = synthesize_u3_single(*tg, T)
    cands = harvest(tg, T)
    seqs = [{s for _t, s, _U in ch} for ch in cands]
    for comps in h["components"]:
        for k, c in enumerate(comps):
            assert c["seq"] in seqs[k]
    # hardcap's eps = |v_best|/2 = asin(err_best): the same best composite.
    assert h["eps"] == pytest.approx(math.asin(r["err"]), rel=1e-9)


def test_single_no_combination_returns_none():
    random.seed(SEED)
    assert synthesize_u3_single(*TARGETS[0], 3, t_min=2) is None


# ---------------------------------------------------------------------------
# enumerate_u3_splits
# ---------------------------------------------------------------------------

def _available(cands, T):
    return [{t for t, _s, _U in ch if t <= T} for ch in cands]


@pytest.mark.parametrize("T", [0, 2, 5, 10])
@pytest.mark.parametrize("tg", TARGETS)
def test_enumerate_best(tg, T, harvests):
    cands = harvests[tg] if T == 10 else harvest(tg, T)
    r = enumerate_u3_splits(*tg, T, candidates=cands)
    assert r["n_splits"] == comb(T + 2, 2)
    av = _available(cands, T)
    assert r["available_t"] == [sorted(a) for a in av]
    expect_splits = [(a, b, T - a - b) for a in range(T + 1) for b in range(T - a + 1)]
    got = [tuple(e["t_split"]) for e in r["entries"]]
    miss = [tuple(m["t_split"]) for m in r["missing"]]
    # every split is either an entry or missing, in lexicographic order
    assert sorted(got + miss) == expect_splits
    assert got == sorted(got) and len(set(got)) == len(got)
    for m in r["missing"]:
        assert m["channels"] == [k for k in range(3) if m["t_split"][k] not in av[k]]
        assert m["channels"]
    for s in got:
        assert all(s[k] in av[k] for k in range(3))
    for e in r["entries"]:
        check_entry(e, tg, T, exact_total=True)
    if all(set(range(T + 1)) <= a for a in av):
        assert not r["missing"] and len(r["entries"]) == comb(T + 2, 2)


def test_enumerate_t2_has_no_missing():
    tg = TARGETS[0]
    r = enumerate_u3_splits(*tg, 2, candidates=harvest(tg, 2))
    assert not r["missing"] and len(r["entries"]) == 6


def test_odd_t_above_one_is_never_harvested(harvests):
    """The gridsynth sweep yields even T ≥ 2 only (plus explicit T=0, T=1)."""
    for cands in harvests.values():
        for ch in cands:
            assert not any(t >= 3 and t % 2 for t, _s, _U in ch)


@pytest.mark.parametrize("T", [5, 10])
@pytest.mark.parametrize("tg", TARGETS[:2])
def test_enumerate_all_vs_best(tg, T, harvests):
    cands = harvests[tg] if T == 10 else harvest(tg, T)
    best = enumerate_u3_splits(*tg, T, candidates=cands)
    allr = enumerate_u3_splits(*tg, T, per_split="all", candidates=cands)
    assert allr["missing"] == best["missing"]
    groups = {}
    for i, e in enumerate(allr["entries"]):
        if i % 10 == 0:   # full checks on a sample; all are cheap-checked below
            check_entry(e, tg, T, exact_total=True)
        assert sum(e["t_split"]) == T
        groups.setdefault(tuple(e["t_split"]), []).append(e)
    assert list(groups) == [tuple(e["t_split"]) for e in best["entries"]]
    for e in best["entries"]:
        g = groups[tuple(e["t_split"])]
        n = 1
        for k in range(3):
            n *= allr["n_per_t"][k][e["t_split"][k]]
        assert len(g) == n == len({tuple(c["seq"] for c in x["rz"]) for x in g})
        assert g[0]["sequence"] == e["sequence"]
        assert e["err"] == min(x["err"] for x in g)
        errs = [x["err"] for x in g]
        assert errs == sorted(errs)


@pytest.mark.parametrize("tg", TARGETS)
def test_single_is_min_over_splits(tg, harvests):
    T = 10
    cands = harvests[tg]
    s = synthesize_u3_single(*tg, T, candidates=cands)
    entries = [e for t in range(T + 1)
               for e in enumerate_u3_splits(*tg, t, candidates=cands)["entries"]]
    m = min(e["err"] for e in entries)
    assert s["err"] == pytest.approx(m, abs=1e-15)
    winners = [e for e in entries if abs(e["err"] - m) < 1e-15]
    w = min(winners, key=lambda e: (e["t_count"], e["sequence"]))
    assert (s["sequence"], s["t_split"]) == (w["sequence"], w["t_split"])


def test_enumerate_piece_best_uses_one_candidate_per_t(harvests):
    tg = TARGETS[0]
    r = enumerate_u3_splits(*tg, 10, piece="best", per_split="all",
                            candidates=harvests[tg])
    assert all(n == 1 for d in r["n_per_t"] for n in d.values())
    assert len(r["entries"]) == r["n_splits"] - len(r["missing"])
    full = enumerate_u3_splits(*tg, 10, candidates=harvests[tg])
    for a, b in zip(full["entries"], r["entries"]):
        assert a["t_split"] == b["t_split"] and a["err"] <= b["err"]


def test_enumerate_t_min():
    tg = TARGETS[0]
    r = enumerate_u3_splits(*tg, 6, t_min=1, candidates=harvest(tg, 6))
    assert r["n_splits"] == comb(6 - 3 + 2, 2)
    for e in r["entries"]:
        assert min(e["t_split"]) >= 1


def test_enumerate_harvests_once_with_seed():
    tg = TARGETS[1]
    random.seed(SEED)
    a = enumerate_u3_splits(*tg, 4)
    b = enumerate_u3_splits(*tg, 4, candidates=harvest(tg, 4))
    assert [e["sequence"] for e in a["entries"]] == [e["sequence"] for e in b["entries"]]
    assert a["t_harvest"] > 0 and b["t_harvest"] == 0.0


def test_enumerate_caches_do_not_change_results():
    """Cached piece matrices and (Rz3, Rz2) products give the cold-cache output."""
    from mixedsynth import u3_single
    th, ph, la = 0.7, 1.9, -0.4
    random.seed(5)
    cands = u3_single.harvest_u3(th, ph, la, 6, n_keep=3, dps=40)

    def run():
        return [u3_single.enumerate_u3_splits(th, ph, la, b, candidates=cands, dps=40)["entries"]
                for b in range(7)]

    u3_single._piece_matrices.cache_clear()
    u3_single._left_product.cache_clear()
    cold = run()
    warm = run()
    assert warm == cold
    # the cached left product is U3·H·U2·H evaluated left to right
    s3, s2 = cands[2][-1][1], cands[1][-1][1]
    mp = u3_single.mp
    with mp.workdps(50):
        H = mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2)
        direct = sequence_to_matrix(s3, dps=40) * H * sequence_to_matrix(s2, dps=40) * H
    assert u3_single._left_product(s3, s2, 40) == direct
