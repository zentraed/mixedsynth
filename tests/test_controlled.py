"""
Tests for mixedsynth.controlled: C-V for V = e^{iγ}U3(θ,φ,λ) from
unconditional single-qubit pieces, as a single circuit and as a mixed channel.

The norm-equation solver uses Python's `random`, so every synthesis is seeded.
"""

import itertools
import math
import random

import numpy as np
import pytest

from mixedsynth import (
    circuit_matrix, circuit_t_count, controlled_u3_matrix, controlled_u3_pieces,
    optimize_controlled_u3, sample_controlled, split_budget,
    synthesize_controlled_u3_single, u3_params,
)
from mixedsynth.controlled import PIECES, _unitary_dn2, controlled_circuit
from mixedsynth.sequences import sequence_to_matrix, t_count

SEED = 1234
TARGETS = [(1.1, 2.3, 0.4, 0.0), (0.3, -1.2, 2.9, 0.7), (2.7, 0.5, -0.8, -2.0)]
EDGE = [(0.0, 0.9, -0.3, 0.2), (math.pi, 0.4, 1.3, 0.0), (0.0, 0.0, 0.0, 0.0),
        (math.pi / 2, math.pi / 4, -math.pi / 2, math.pi / 8)]
BUDGETS = (0, 6, 20)

I2 = np.eye(2, dtype=complex)
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def same_up_to_phase(A, B, tol=1e-10):
    return abs(abs(np.trace(A.conj().T @ B)) - A.shape[0]) < tol


def compose(mats):
    """Two-qubit matrix of the C-V circuit from 2×2 piece matrices."""
    return (np.kron(mats["P"], I2) @ np.kron(I2, mats["A"]) @ CX
            @ np.kron(I2, mats["B"]) @ CX @ np.kron(I2, mats["C"]))


def pairwise_t_ok(r, t_total):
    """Every circuit of the product ensemble respects t_total."""
    for combo in itertools.product(*(r["pieces"][k]["members"] for k in PIECES)):
        assert sum(m["t_count"] for m in combo) <= t_total


# ---------------------------------------------------------------------------
# Exact decomposition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tg", TARGETS + EDGE)
def test_pieces_compose_to_controlled_v_exactly(tg):
    pieces = controlled_u3_pieces(*tg)
    target = controlled_u3_matrix(*tg)
    mats = {k: u3(*pieces[k]["u3"]) for k in PIECES}
    # P is exact; A, B, C are exact only up to phase, so the composite is too.
    assert same_up_to_phase(compose(mats), target)
    assert pieces["P"]["qubit"] == 0 and all(pieces[k]["qubit"] == 1 for k in "ABC")


@pytest.mark.parametrize("tg", TARGETS + EDGE)
def test_any_phase_on_any_piece_is_global(tg):
    """The property the whole construction relies on."""
    rng = np.random.default_rng(7)
    pieces = controlled_u3_pieces(*tg)
    target = controlled_u3_matrix(*tg)
    for _ in range(20):
        mats = {k: np.exp(1j * rng.uniform(0, 2 * np.pi)) * u3(*pieces[k]["u3"])
                for k in PIECES}
        assert same_up_to_phase(compose(mats), target)


def test_relative_phase_gamma_matters():
    a = controlled_u3_matrix(1.1, 2.3, 0.4, 0.0)
    b = controlled_u3_matrix(1.1, 2.3, 0.4, 0.5)
    assert not same_up_to_phase(a, b)
    assert _unitary_dn2(a, b) == pytest.approx(math.sin(0.25), abs=1e-12)


def test_u3_params_roundtrip():
    rng = np.random.default_rng(3)
    mats = []
    for _ in range(50):
        Z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        Q, R = np.linalg.qr(Z)
        mats.append(Q * (np.diag(R) / abs(np.diag(R))))
    mats += [np.diag([np.exp(0.3j), np.exp(-1.1j)]),
             np.array([[0, np.exp(0.4j)], [np.exp(2.0j), 0]]), I2]
    for V in mats:
        th, ph, la, ga = u3_params(V)
        assert 0 <= th <= math.pi + 1e-12
        assert np.allclose(np.exp(1j * ga) * u3(th, ph, la), V, atol=1e-12)


def test_u3_params_rejects_non_unitary():
    with pytest.raises(ValueError):
        u3_params(np.array([[1, 1], [0, 1]]))


# ---------------------------------------------------------------------------
# Circuits and metrics
# ---------------------------------------------------------------------------

def test_circuit_matrix_conventions():
    assert np.allclose(circuit_matrix([("cx", 0, 1)]), CX)
    SW = np.eye(4)[[0, 2, 1, 3]]
    assert np.allclose(circuit_matrix([("cx", 1, 0)]), SW @ CX @ SW)
    X = sequence_to_matrix("X")
    assert np.allclose(circuit_matrix([("u", 0, "X")]), np.kron(X, I2))
    # application order: the first operation is applied first
    ops = [("u", 0, "H"), ("cx", 0, 1)]
    assert np.allclose(circuit_matrix(ops), CX @ np.kron(sequence_to_matrix("H"), I2))
    assert circuit_t_count([("u", 0, "TST"), ("cx", 0, 1), ("u", 1, "HT")]) == 3
    with pytest.raises(ValueError):
        circuit_matrix([("cz", 0, 1)])


def test_unitary_dn2_known_values():
    I4 = np.eye(4, dtype=complex)
    for phi in (0.0, 0.3, 1.7, math.pi - 1e-3):
        U = np.diag([1, 1, 1, np.exp(1j * phi)])
        assert _unitary_dn2(U, I4) == pytest.approx(math.sin(phi / 2), abs=1e-12)
        assert _unitary_dn2(np.exp(0.9j) * U, I4) == pytest.approx(math.sin(phi / 2), abs=1e-12)
    # eigenvalues surround the origin: perfectly distinguishable
    assert _unitary_dn2(np.diag([1, 1j, -1, -1j]), I4) == 1.0


# ---------------------------------------------------------------------------
# Budget split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tg", TARGETS + EDGE)
@pytest.mark.parametrize("T", (0, 1, 5, 12, 31))
def test_split_budget_within_total(tg, T):
    s = split_budget(controlled_u3_pieces(*tg), T)
    assert set(s) == set(PIECES) and all(v >= 0 for v in s.values())
    assert sum(s.values()) <= T


def test_split_budget_generic_ratio():
    # generic angles: A and B have two non-trivial Rz each, C and P one
    s = split_budget(controlled_u3_pieces(*TARGETS[0]), 60)
    assert s == {"A": 20, "B": 20, "C": 10, "P": 10}


def test_split_validation():
    with pytest.raises(ValueError):
        synthesize_controlled_u3_single(*TARGETS[0][:3])
    with pytest.raises(ValueError):
        synthesize_controlled_u3_single(*TARGETS[0][:3], t_total=4, t_split=[2, 2, 2, 2])
    with pytest.raises(ValueError):
        synthesize_controlled_u3_single(*TARGETS[0][:3], t_split={"A": 1, "B": 1})


# ---------------------------------------------------------------------------
# Single circuit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=[(tg, T) for tg in TARGETS + EDGE[:2] for T in BUDGETS],
                ids=lambda p: "%s-T%d" % (p[0], p[1]))
def single(request):
    tg, T = request.param
    random.seed(SEED)
    r = synthesize_controlled_u3_single(*tg[:3], t_total=T, gamma=tg[3])
    assert r is not None
    return tg, T, r


def test_single_budget_and_structure(single):
    tg, T, r = single
    ops = r["circuit"]
    assert [op[0] for op in ops] == ["u", "cx", "u", "cx", "u", "u"]
    assert r["t_count"] == circuit_t_count(ops) <= T
    assert sum(r["t_split"].values()) <= T
    for k in PIECES:
        p = r["pieces"][k]
        assert p["t_count"] == t_count(p["sequence"]) <= p["t_budget"] == r["t_split"][k]
    assert ops == controlled_circuit({k: r["pieces"][k]["sequence"] for k in PIECES})


def test_single_error_is_exact_and_bounded(single):
    tg, T, r = single
    target = controlled_u3_matrix(*tg)
    U = circuit_matrix(r["circuit"])
    assert r["dn2"] == pytest.approx(_unitary_dn2(U, target), abs=1e-12)
    # subadditivity over the pieces
    assert r["dn2"] <= sum(r["pieces"][k]["dn2"] for k in PIECES) + 1e-9
    # each piece error is its own phase-insensitive qubit error
    for k in PIECES:
        p = r["pieces"][k]
        ov = abs(np.trace(u3(*p["u3"]).conj().T @ sequence_to_matrix(p["sequence"]))) / 2
        assert p["dn2"] == pytest.approx(math.sqrt(max(0.0, 1 - ov ** 2)), abs=1e-9)


def test_single_respects_gamma():
    """The circuit implements the requested relative phase, not just V up to phase."""
    random.seed(SEED)
    tg = TARGETS[0][:3]
    r = synthesize_controlled_u3_single(*tg, t_total=40, gamma=0.6)
    U = circuit_matrix(r["circuit"])
    assert _unitary_dn2(U, controlled_u3_matrix(*tg, 0.6)) == pytest.approx(r["dn2"])
    assert _unitary_dn2(U, controlled_u3_matrix(*tg, 0.0)) > r["dn2"] + 0.1


def test_exact_targets_synthesize_exactly():
    """Clifford+T-exact C-V (controlled-H, -S, -X, -Y, -Sd) come out exact."""
    for g in ("H", "S", "X", "Y", "Sd", "HS"):
        V = sequence_to_matrix(g)
        th, ph, la, ga = u3_params(V)
        random.seed(SEED)
        r = synthesize_controlled_u3_single(th, ph, la, t_total=12, gamma=ga)
        assert r["dn2"] < 1e-7
        assert r["t_count"] <= 6   # ≤ one T per odd-π/4 rotation


def test_controlled_t_is_approximate():
    """C-T has det ω: not exact without an ancilla, so it is approximated."""
    th, ph, la, ga = u3_params(sequence_to_matrix("T"))
    random.seed(SEED)
    lo = synthesize_controlled_u3_single(th, ph, la, t_total=12, gamma=ga)["dn2"]
    random.seed(SEED)
    hi = synthesize_controlled_u3_single(th, ph, la, t_total=40, gamma=ga)["dn2"]
    assert 1e-6 < hi < lo


def test_pieces_are_simplified():
    random.seed(SEED)
    r = synthesize_controlled_u3_single(0.0, 0.0, 0.0, t_total=4)   # C-I
    assert all(r["pieces"][k]["sequence"] == "" for k in PIECES)
    assert r["dn2"] < 1e-12
    for k in PIECES:
        p = r["pieces"][k]
        A, B = sequence_to_matrix(p["sequence"]), sequence_to_matrix(p["u3_sequence"])
        assert same_up_to_phase(A, B) and p["t_count"] <= t_count(p["u3_sequence"])


# ---------------------------------------------------------------------------
# Mixed channel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=[(tg, T) for tg in TARGETS[:2] + EDGE[:1] for T in (6, 24)],
                ids=lambda p: "%s-T%d" % (p[0], p[1]))
def mixed(request):
    tg, T = request.param
    random.seed(SEED)
    r = optimize_controlled_u3(*tg[:3], t_total=T, gamma=tg[3])
    assert r is not None
    return tg, T, r


def test_mixed_hard_budget(mixed):
    tg, T, r = mixed
    assert sum(r["t_split"].values()) <= T
    assert r["t_max"] <= T
    assert r["E_T"] <= r["t_max"] + 1e-9
    pairwise_t_ok(r, T)
    for k in PIECES:
        p = r["pieces"][k]
        assert 1 <= len(p["members"]) <= 4
        assert sum(float(m["weight"]) for m in p["members"]) == pytest.approx(1.0)
        assert all(m["t_count"] <= p["t_budget"] for m in p["members"])
        assert len({m["sequence"] for m in p["members"]}) == len(p["members"])
        for m in p["members"]:
            assert m["t_count"] == t_count(m["sequence"]) <= t_count(m["u3_sequence"])
            assert same_up_to_phase(sequence_to_matrix(m["sequence"]),
                                    sequence_to_matrix(m["u3_sequence"]))


def test_mixed_bounds(mixed):
    tg, T, r = mixed
    assert 0 <= r["dn2_lower"] <= r["dn2_upper"] + 1e-12
    assert r["dn2_upper"] == pytest.approx(sum(r["pieces"][k]["dn2"] for k in PIECES))
    assert r["n_prod"] == np.prod([len(r["pieces"][k]["members"]) for k in PIECES])


def test_mixed_beats_single_circuit():
    random.seed(SEED)
    tg = TARGETS[0][:3]
    s = synthesize_controlled_u3_single(*tg, t_total=36)
    random.seed(SEED)
    r = optimize_controlled_u3(*tg, t_total=36)
    assert r["dn2_upper"] < 0.5 * s["dn2"]


def test_mixed_lower_bound_matches_brute_force(mixed):
    """dn2_lower recomputed independently from the product ensemble."""
    tg, T, r = mixed
    def choi(U):
        v = np.kron(np.eye(4), U) @ (np.eye(4).reshape(-1) / 2)
        return np.outer(v, v.conj())
    J = -choi(controlled_u3_matrix(*tg))
    for combo in itertools.product(*(r["pieces"][k]["members"] for k in PIECES)):
        mats = {k: sequence_to_matrix(m["sequence"]) for k, m in zip(PIECES, combo)}
        J += np.prod([float(m["weight"]) for m in combo]) * choi(compose(mats))
    assert r["dn2_lower"] == pytest.approx(0.5 * np.abs(np.linalg.eigvalsh(J)).sum(), abs=1e-12)


def test_sample_controlled(mixed):
    tg, T, r = mixed
    rng = np.random.default_rng(0)
    circuits = sample_controlled(r, rng=rng, size=50)
    assert len(circuits) == 50
    members = {k: {m["sequence"] for m in r["pieces"][k]["members"]} for k in PIECES}
    for ops in circuits:
        assert circuit_t_count(ops) <= T
        seqs = {"C": ops[0][2], "B": ops[2][2], "A": ops[4][2], "P": ops[5][2]}
        assert all(seqs[k] in members[k] for k in PIECES)
        assert ops == controlled_circuit(seqs)
    one = sample_controlled(r, rng=rng)
    assert isinstance(one, list) and one[1] == ("cx", 0, 1)


def test_mixed_channel_matches_sampling_average():
    """The averaged sampled circuits converge to the reported channel."""
    random.seed(SEED)
    tg = TARGETS[1]
    r = optimize_controlled_u3(*tg[:3], t_total=12, gamma=tg[3])
    def choi(U):
        v = np.kron(np.eye(4), U) @ (np.eye(4).reshape(-1) / 2)
        return np.outer(v, v.conj())
    exact = sum(np.prod([float(m["weight"]) for m in combo])
                * choi(compose({k: sequence_to_matrix(m["sequence"]) for k, m in zip(PIECES, combo)}))
                for combo in itertools.product(*(r["pieces"][k]["members"] for k in PIECES)))
    n = 4000
    emp = sum(choi(circuit_matrix(c)) for c in
              sample_controlled(r, rng=np.random.default_rng(1), size=n)) / n
    assert 0.5 * np.abs(np.linalg.eigvalsh(emp - exact)).sum() < 0.05
