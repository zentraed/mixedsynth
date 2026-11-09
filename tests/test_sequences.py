"""
Sequence generation + channel synthesis end-to-end checks.

Every gate string mixedsynth returns must reproduce its unitary (up to global
phase) and carry the advertised T-count; the synthesized channel must respect
the hard T budget and beat the best single sequence.
"""

import random

import numpy as np
import pytest

from mixedsynth import harvest_rz, optimize_u3_hardcap
from mixedsynth.sequences import (
    tokenize, t_count, circuit_order, sequence_to_matrix, sample,
)


def _np(U_mp):
    return np.array([[complex(U_mp[i, j]) for j in range(2)] for i in range(2)])


def _phase_overlap(A, B):
    return abs(np.trace(A.conj().T @ B)) / 2


def test_tokenize_roundtrip():
    assert tokenize("SdHTS") == ["Sd", "H", "T", "S"]
    assert circuit_order("SdHTS") == ["S", "T", "H", "Sd"]
    assert t_count("HTSHTSdT") == 3
    assert tokenize("") == []
    with pytest.raises(ValueError):
        tokenize("HQT")


def test_sequence_to_matrix_mp_matches_np():
    seq = "HTSHTHTSdHTXYZ"
    M_np = sequence_to_matrix(seq)
    M_mp = _np(sequence_to_matrix(seq, dps=40))
    assert np.allclose(M_np, M_mp, atol=1e-14)


@pytest.mark.parametrize("angle", [0.3, 1.7, 4.2])
def test_harvest_sequences_match_unitaries(angle):
    random.seed(0)
    members = harvest_rz(angle, t_min=0, t_budget=16, n_keep=3,
                         dps=30, with_sequences=True)
    assert members, "harvest returned nothing"
    assert {t for t, _s, _U in members} >= {0, 1}
    for t, seq, U_mp in members:
        assert t_count(seq) == t
        assert _phase_overlap(sequence_to_matrix(seq), _np(U_mp)) == pytest.approx(1, abs=1e-12)


def test_harvest_default_shape_unchanged():
    random.seed(0)
    members = harvest_rz(0.9, t_min=0, t_budget=6, n_keep=2, dps=30)
    assert all(len(m) == 2 for m in members)


def test_hardcap_channel_sequences():
    random.seed(0)
    T = 18
    r = optimize_u3_hardcap(1.1, 2.3, 0.4, t_total=T, n_keep=3,
                            n_keep_total=6, dps=40)
    assert r is not None

    # every pool sequence reproduces its composite unitary
    for U_mp, seq, t in zip(r["U_list"], r["sequences"], r["t_counts"]):
        assert t_count(seq) == t <= T
        assert _phase_overlap(sequence_to_matrix(seq), _np(U_mp)) == pytest.approx(1, abs=1e-12)

    members = r["members"]
    assert 1 <= len(members) <= 4                       # Carathéodory in R³
    assert sum(float(m["weight"]) for m in members) == pytest.approx(1, abs=1e-9)
    assert all(m["t_count"] <= T for m in members)      # hard per-shot budget
    assert r["dn2"] <= r["eps"]                         # never worse than best single

    rng = np.random.default_rng(0)
    draws = sample(members, rng=rng, size=50)
    allowed = {m["sequence"] for m in members}
    assert set(draws) <= allowed
    assert sample(members, rng=rng) in allowed
