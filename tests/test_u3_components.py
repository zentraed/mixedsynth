"""
Tests for the per-member Rz components returned by optimize_u3_hardcap.

Each composite U3 member is the matrix-order string
    rz3.seq + "H" + rz2.seq + "H" + rz1.seq     (Rz3 · H · Rz2 · H · Rz1)
and optimize_u3_hardcap reports the three pieces ("components", and per
member "rz" / "t_split").  The norm-equation solver uses Python's `random`,
so every run is seeded.
"""

import random

import numpy as np
import pytest

from mixedsynth import harvest_rz, optimize_u3_hardcap, rz_angles_for_u3
from mixedsynth.hst import from_pyliqtr, hst_equal
from mixedsynth.sequences import sequence_to_matrix, t_count

TARGET = (1.1, 2.3, 0.4)
SEED = 1234
BUDGETS = (6, 10)


def _run(t_total):
    random.seed(SEED)
    return optimize_u3_hardcap(*TARGET, t_total=t_total)


@pytest.fixture(scope="module", params=BUDGETS)
def result(request):
    r = _run(request.param)
    assert r is not None
    return request.param, r


def _phase_overlap(A, B):
    """|tr(A†B)|/2: 1 iff A and B agree up to global phase."""
    return abs(np.trace(A.conj().T @ B)) / 2


def _np(U_mp):
    return np.array([[complex(U_mp[i, j]) for j in range(2)] for i in range(2)])


def _rz(a):
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


# (a) sequence == rz3 + "H" + rz2 + "H" + rz1, for pool and members ----------

def test_sequence_is_concatenation_of_components(result):
    _T, r = result
    assert len(r["components"]) == len(r["sequences"]) == len(r["U_list"])
    for comps, seq in zip(r["components"], r["sequences"]):
        assert len(comps) == 3
        assert seq == comps[2]["seq"] + "H" + comps[1]["seq"] + "H" + comps[0]["seq"]
    for m in r["members"]:
        rz1, rz2, rz3 = m["rz"]
        assert m["sequence"] == rz3["seq"] + "H" + rz2["seq"] + "H" + rz1["seq"]


def test_members_match_pool_components(result):
    _T, r = result
    by_seq = {s: c for s, c in zip(r["sequences"], r["components"])}
    for m in r["members"]:
        assert m["rz"] == by_seq[m["sequence"]]


def test_component_angles_follow_rz_angles_for_u3(result):
    _T, r = result
    expected = [float(a) for a in rz_angles_for_u3(*TARGET)]
    theta, phi, lam = TARGET
    assert expected == pytest.approx([lam - np.pi / 2, theta, phi + 5 * np.pi / 2])
    for comps in r["components"]:
        assert [c["angle"] for c in comps] == expected


# (b) T accounting -----------------------------------------------------------

def test_t_split_and_component_t_counts(result):
    T, r = result
    for comps, tc in zip(r["components"], r["t_counts"]):
        assert sum(c["t_count"] for c in comps) == tc <= T
        for c in comps:
            assert t_count(c["seq"]) == c["t_count"]
    for m in r["members"]:
        assert m["t_split"] == [c["t_count"] for c in m["rz"]]
        assert sum(m["t_split"]) == m["t_count"]
        for c in m["rz"]:
            assert t_count(c["seq"]) == c["t_count"]


# (c) components are genuine harvested approximations of Rz(angle) -----------

def test_components_are_harvested_rz_candidates(result):
    """
    Low-T approximations of Rz can be poor (a Clifford is far from a generic
    Rz), and the harvest deliberately keeps several non-optimal candidates per
    T level, so a fixed fidelity threshold is not meaningful.  Instead check
    that every component is exactly one of the candidates harvest_rz produced
    for its angle, with the same T-count, and that its matrix reproduces that
    candidate's unitary up to global phase.  The harvest is re-run with the
    same seed and arguments, which reproduces optimize_u3_hardcap's harvest.
    """
    T, r = result
    random.seed(SEED)
    stride = max(1, T // 30)
    harvested = []
    for a in rz_angles_for_u3(*TARGET):
        cands = harvest_rz(a, t_min=0, t_budget=T, n_keep=5, stride=stride,
                           dps=60, with_sequences=True)
        harvested.append({s: (t, _np(U)) for t, s, U in cands})

    for comps in r["components"]:
        for k, c in enumerate(comps):
            assert c["seq"] in harvested[k]
            t_h, U_h = harvested[k][c["seq"]]
            assert c["t_count"] == t_h
            V = sequence_to_matrix(c["seq"])
            assert _phase_overlap(V, U_h) == pytest.approx(1, abs=1e-12)
            # Its fidelity to Rz(angle) is exactly that of the harvested U.
            assert _phase_overlap(V, _rz(c["angle"])) == pytest.approx(
                _phase_overlap(U_h, _rz(c["angle"])), abs=1e-12)


def test_components_compose_to_member_unitary(result):
    _T, r = result
    H = sequence_to_matrix("H")
    for comps, U_mp, seq in zip(r["components"], r["U_list"], r["sequences"]):
        V1, V2, V3 = (sequence_to_matrix(c["seq"]) for c in comps)
        prod = V3 @ H @ V2 @ H @ V1
        assert _phase_overlap(prod, _np(U_mp)) == pytest.approx(1, abs=1e-12)
        # and the hst conversion agrees exactly with the concatenation
        hst_parts = [from_pyliqtr(c["seq"]) for c in comps]
        assert hst_equal(from_pyliqtr(seq),
                         hst_parts[2] + "H" + hst_parts[1] + "H" + hst_parts[0])


# (d) results unchanged by the change ----------------------------------------

# Recorded from optimize_u3_hardcap before components were added
# (random.seed(1234), U3(1.1, 2.3, 0.4), default arguments).
BASELINE = {
    6: {
        "dn2": 0.013295014633567658,
        "eps": 0.09838872356688146,
        "E_T": 3.2635084806882597,
        "n_pool": 70,
        "members": [
            ("0.446390902209854", 2, "SSSHTHSSSHTH"),
            ("0.316975970157442", 3, "HTHSTHSST"),
            ("0.233055224711204", 6, "THTHTHTSdHZHSHSHTSHTYS"),
            ("0.00357790292150058", 6, "SHSSSHHHTSHTHTSHTHTHTSd"),
        ],
    },
    10: {
        "dn2": 0.011603200312383927,
        "eps": 0.08920053380912611,
        "E_T": 9.352838949837084,
        "n_pool": 110,
        "members": [
            ("0.431391878588021", 9, "SHTHTHTSHTHTHTHTHTZHSHSSSHTHSSH"),
            ("0.342836956153318", 10, "HTHTHTSHTYHHTHTSHTHTSHTHTSH"),
            ("0.117886579471213", 10, "HTHTHTSHTYHTSHTSdHSdHHTHTHTSHTX"),
            ("0.107884585787448", 8, "TSHTSdHSHTHTHTSHTHTHTHZH"),
        ],
    },
}


def test_results_match_pre_change_baseline(result):
    T, r = result
    b = BASELINE[T]
    assert r["dn2"] == pytest.approx(b["dn2"], rel=1e-12)
    assert r["eps"] == pytest.approx(b["eps"], rel=1e-12)
    assert r["E_T"] == pytest.approx(b["E_T"], rel=1e-12)
    assert len(r["sequences"]) == b["n_pool"]
    assert len(r["members"]) == len(b["members"])
    for m, (w, tc, seq) in zip(r["members"], b["members"]):
        assert float(m["weight"]) == pytest.approx(float(w), rel=1e-12)
        assert m["t_count"] == tc
        assert m["sequence"] == seq
    assert set(r["members"][0]) == {"weight", "t_count", "sequence", "t_split", "rz"}


def test_member_rz_is_a_copy(result):
    """Mutating a member's rz dicts must not alter the pool components."""
    _T, r = result
    m = r["members"][0]
    by_seq = {s: c for s, c in zip(r["sequences"], r["components"])}
    assert m["rz"] is not by_seq[m["sequence"]]
    assert all(a is not b for a, b in zip(m["rz"], by_seq[m["sequence"]]))
