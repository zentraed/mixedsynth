"""
H/S/T/W conversion and normal-form checks (mixedsynth.hst).

Every rewrite must be exact including global phase; the normal form must be
unique per unitary (phase included), reproduce its input exactly, agree with
gridsynth's own normal form, and never raise the T-count.
"""

import itertools
import math
import os
import random

import numpy as np
import pytest

from mixedsynth import hst
from mixedsynth.sequences import sequence_to_matrix

W = np.exp(1j * np.pi / 4)
# Independent reference gates (not taken from mixedsynth).
REF = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]),
    "Sd": np.diag([1, -1j]),
    "T": np.diag([1, W]),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
    "W": W * np.eye(2),
    "I": np.eye(2),
}
REF["E"] = REF["H"] @ REF["S"] @ REF["S"] @ REF["S"] * W**3


def ref_product(tokens):
    M = np.eye(2, dtype=complex)
    for g in tokens:
        M = M @ REF[g]
    return M


def rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def load_gridsynth():
    path = os.path.join(os.path.dirname(__file__), "data", "gridsynth_rz.tsv")
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            angle, digits, seq = line.rstrip("\n").split("\t")
            theta = eval(angle, {"pi": math.pi})
            rows.append((theta, int(digits), seq))
    return rows


GRIDSYNTH = load_gridsynth()

# Trasyn 0.3.0 output, trasyn.synthesize(..., gpu=False, rng=7).
TRASYN = [
    "shththz",
    "shxththyththththythththtz",
    "shththxthththxthththx",
    "hthythththythththxththths",
    "thssthththsthst",
    "hsththshththsththsht",
]


# ---------------------------------------------------------------------------
# Rewrites
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tok", ["H", "S", "T", "X", "Y", "Z", "E", "W", "I"])
def test_gridsynth_tokens_exact(tok):
    assert np.allclose(hst.hst_to_matrix(hst.from_gridsynth(tok)), REF[tok], atol=1e-12)


def test_gridsynth_phase_tokens():
    assert hst.from_gridsynth("-") == "WWWW"
    assert hst.from_gridsynth("i") == "WW"
    assert hst.from_gridsynth("I") == ""
    assert hst.from_gridsynth("WWWWWWWWW") == "W"


@pytest.mark.parametrize("tok", ["H", "S", "Sd", "T", "X", "Y", "Z"])
def test_pyliqtr_tokens_exact(tok):
    assert np.allclose(hst.hst_to_matrix(hst.from_pyliqtr(tok)), REF[tok], atol=1e-12)


def test_pyliqtr_string_matches_sequences_module():
    seq = "HTSHTHTSdHTXYZSdTH"
    assert np.allclose(hst.hst_to_matrix(hst.from_pyliqtr(seq)),
                       sequence_to_matrix(seq), atol=1e-12)


@pytest.mark.parametrize("seq", TRASYN)
def test_trasyn_strings_exact(seq):
    expected = ref_product([c.upper() for c in seq])
    assert np.allclose(hst.hst_to_matrix(hst.from_trasyn(seq)), expected, atol=1e-12)


def test_trasyn_matches_trasyn_seq2mat():
    trasyn = pytest.importorskip("trasyn")
    for seq in TRASYN:
        assert np.allclose(hst.hst_to_matrix(hst.from_trasyn(seq)),
                           trasyn.utils.seq2mat(seq), atol=1e-12)


def test_phase_collected_at_end():
    s = hst.from_gridsynth("WHYWT")
    assert s == "HHSSHSST" + "W" * 4
    assert hst.is_hst(s)


@pytest.mark.parametrize("fn,bad", [
    (hst.from_gridsynth, "HTQ"), (hst.from_gridsynth, "ht"),
    (hst.from_pyliqtr, "HTW"), (hst.from_pyliqtr, "Td"),
    (hst.from_trasyn, "HT"), (hst.from_trasyn, "htq"),
])
def test_unknown_tokens_rejected(fn, bad):
    with pytest.raises(ValueError):
        fn(bad)


def test_hst_functions_reject_other_letters():
    for fn in (hst.hst_to_ring, hst.t_count, hst.normal_form):
        with pytest.raises(ValueError):
            fn("HXT")


# ---------------------------------------------------------------------------
# gridsynth: matrix order, exact phase, and its normal form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta,digits,seq", GRIDSYNTH)
def test_gridsynth_is_phase_exact_rz(theta, digits, seq):
    V = hst.hst_to_matrix(hst.from_gridsynth(seq))
    assert np.linalg.norm(V - rz(theta), 2) <= 10.0 ** (-digits)


@pytest.mark.parametrize("theta,digits,seq", GRIDSYNTH)
def test_gridsynth_output_is_normal_form(theta, digits, seq):
    s = hst.from_gridsynth(seq)
    assert hst.is_normal_form(s)
    assert hst.normal_form(s) == s


# ---------------------------------------------------------------------------
# Normal form
# ---------------------------------------------------------------------------

def test_192_distinct_cliffords():
    assert len(hst._clifford_table()) == 192


def test_known_identities():
    assert hst.normal_form("") == ""
    assert hst.normal_form("HH") == ""
    assert hst.normal_form("SSSS") == ""
    assert hst.normal_form("TTTTTTTT") == ""
    assert hst.normal_form("WWWWWWWW") == ""
    assert hst.normal_form("SHSHSH") == "W"        # (SH)^3 = ω
    assert hst.normal_form("TT") == "S"


def _syllable_prefixes(n):
    """All (T)?(HT|SHT)* words with T-count <= n."""
    out = [""]
    frontier = [""]
    for _ in range(n):
        frontier = [p + s for p in frontier for s in ("HT", "SHT")]
        out += frontier
    return out + ["T" + p for p in out if p.count("T") < n]


def test_enumeration_unique_and_fixed():
    n = 3
    prefixes = _syllable_prefixes(n)
    assert len(prefixes) == 3 * 2 ** n - 2
    keys = set()
    words = [p + c for p in prefixes for c in hst._clifford_table().values()]
    for w in words:
        keys.add(hst.ring_key(w))
    assert len(keys) == len(words)                  # unique, phase included
    # Matsumoto–Amano count of distinct unitaries mod phase with T <= n.
    assert len(words) // 8 == 24 * (3 * 2 ** n - 2)
    rng = random.Random(0)
    for w in rng.sample(words, 400):
        assert hst.normal_form(w) == w


def _random_word(rng, n):
    return "".join(rng.choice("HSTW") for _ in range(n))


def _pad_identities(rng, seq, k):
    ids = ["HH", "SSSS", "TTTTTTTT", "WWWWWWWW", "SHSHSHWWWWWWW", "STTTTTT"]
    for _ in range(k):
        i = rng.randrange(len(seq) + 1)
        seq = seq[:i] + rng.choice(ids) + seq[i:]
    return seq


def test_random_words():
    rng = random.Random(1)
    for _ in range(300):
        s = _random_word(rng, rng.randrange(0, 40))
        nf = hst.normal_form(s)
        assert hst.is_normal_form(nf)
        assert hst.hst_equal(nf, s)
        assert hst.normal_form(nf) == nf
        assert hst.t_count(nf) <= hst.t_count(s)
        assert hst.normal_form(_pad_identities(rng, s, 4)) == nf


def test_normal_form_of_tool_output():
    for seq in TRASYN:
        s = hst.from_trasyn(seq)
        nf = hst.normal_form(s)
        assert hst.hst_equal(nf, s) and hst.t_count(nf) <= seq.count("t")


def test_ring_input():
    s = "HTSHTW"
    assert hst.normal_form(hst.hst_to_ring(s)) == hst.normal_form(s)


def test_split_normal_form():
    parts = hst.split_normal_form("THTSHTSHHSSHSSWWW")
    assert parts == {"syllables": ["T", "HT", "SHT"], "K": "SH", "b": 1, "c": 2, "d": 3}
    with pytest.raises(ValueError):
        hst.split_normal_form("HH")


@pytest.mark.parametrize("K,b,c,d", itertools.product(["", "H", "SH"], [0, 1], range(4), range(8)))
def test_clifford_word_matches_coordinates(K, b, c, d):
    expected = ref_product(list(K)) @ np.linalg.matrix_power(REF["X"], b) \
        @ np.linalg.matrix_power(REF["S"], c) * W**d
    assert np.allclose(hst.hst_to_matrix(hst.clifford_word(K, b, c, d)), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Matrices, phase alignment, helpers
# ---------------------------------------------------------------------------

def test_matrix_mp_matches_np():
    s = "HTSHTHTSHTWWW"
    M_mp = hst.hst_to_matrix(s, dps=40)
    M = np.array([[complex(M_mp[i, j]) for j in range(2)] for i in range(2)])
    assert np.allclose(M, hst.hst_to_matrix(s), atol=1e-14)


@pytest.mark.parametrize("d,r", [(0, 0.0), (3, 0.1), (7, -0.3), (5, math.pi / 8 - 1e-6)])
def test_align_phase(d, r):
    s = "HTSHTHTSHT"
    U = hst.hst_to_matrix(s) * W**d * np.exp(1j * r)
    aligned, residual = hst.align_phase(s + "WW", U)
    assert aligned == s + "W" * d
    assert abs(residual - r) < 1e-9
    assert abs(residual) <= math.pi / 8 + 1e-12


def test_align_phase_on_trasyn():
    th, ph, la = 1.1, 2.3, 0.4
    U = np.array([[np.cos(th / 2), -np.exp(1j * la) * np.sin(th / 2)],
                  [np.exp(1j * ph) * np.sin(th / 2), np.exp(1j * (ph + la)) * np.cos(th / 2)]])
    aligned, r = hst.align_phase(hst.from_trasyn(TRASYN[1]), U)
    V = hst.hst_to_matrix(aligned)
    ov = np.trace(V.conj().T @ U)
    assert abs(np.angle(ov) - r) < 1e-12 and abs(r) <= math.pi / 8


def test_helpers():
    assert hst.t_count("HTSHTTW") == 3
    assert hst.to_circuit_order("HTSWWW") == ["S", "T", "H", "WWW"]
    assert hst.to_circuit_order("HT") == ["T", "H"]


# ---------------------------------------------------------------------------
# Compact normal form
# ---------------------------------------------------------------------------

def test_shortest_clifford_words():
    best = hst._shortest_clifford()
    assert len(best) == 24
    words = [w for w, _ in best.values()]
    assert len(set(words)) == 24 and max(map(len, words)) == 6
    # No shorter H/S word reaches any Clifford (independent brute force).
    reach = {}
    for n in range(7):
        for w in map("".join, itertools.product("HS", repeat=n)):
            cls = hst.normal_form(w).rstrip("W")
            reach.setdefault(cls, n)
    for cls, (w, _) in best.items():
        assert len(w) == reach[cls]


def test_compact_form_on_random_words():
    rng = random.Random(2)
    for _ in range(300):
        s = _random_word(rng, rng.randrange(0, 40))
        nf, cf = hst.normal_form(s), hst.compact_form(s)
        assert hst.hst_equal(cf, s)
        assert hst.t_count(cf) == hst.t_count(nf)
        # Fewer or equal gates; the W count can differ (it encodes the phase).
        assert len(cf.rstrip("W")) <= len(nf.rstrip("W"))
        assert hst.compact_form(cf) == cf and hst.is_compact_form(cf)
        assert hst.normal_form(cf) == nf
        assert hst.compact_form(_pad_identities(rng, s, 4)) == cf


@pytest.mark.parametrize("theta,digits,seq", GRIDSYNTH)
def test_compact_form_of_gridsynth(theta, digits, seq):
    s = hst.from_gridsynth(seq)
    cf = hst.compact_form(s)
    assert hst.hst_equal(cf, s) and len(cf.rstrip("W")) <= len(s.rstrip("W"))
    assert hst.t_count(cf) == hst.t_count(s)


def test_is_compact_form_rejects_gridsynth_layout():
    assert not hst.is_compact_form("SHHSSH")      # gridsynth layout of HSHS
    assert hst.is_compact_form(hst.compact_form("SHHSSH"))


# ---------------------------------------------------------------------------
# Fast paths: rewriting normal form, blocked and per-factor ring products
# ---------------------------------------------------------------------------

def _ring_letter_by_letter(seq):
    """Reference exact matrix: one gate at a time, reduced at the end."""
    m = hst.MAT_D_OMEGA.I()
    for c in seq:
        m = m @ hst._GATE[c]
    return hst._canonical(m)


def _key(m):
    return (m.k,) + tuple((e.a1, e.a2, e.a3, e.a4) for e in m.elements())


def test_hst_to_ring_blocked_matches_letter_by_letter():
    rng = random.Random(10)
    words = ["", "H", "T", "W", "HH", "TTTTTTTT"] + [
        _random_word(rng, rng.randrange(0, 90)) for _ in range(200)]
    for s in words:
        assert _key(hst.hst_to_ring(s)) == _key(_ring_letter_by_letter(s)), s


def test_hst_product_to_ring():
    rng = random.Random(11)
    for _ in range(200):
        parts = [_random_word(rng, rng.randrange(0, 25)) for _ in range(rng.randrange(1, 6))]
        assert _key(hst.hst_product_to_ring(*parts)) == _key(hst.hst_to_ring("".join(parts)))
    assert _key(hst.hst_product_to_ring()) == _key(hst.hst_to_ring(""))


def test_normal_form_rewriting_matches_decompose_all_short_words():
    # every word of length <= 5 over H, S, T, W
    for n in range(6):
        for w in map("".join, itertools.product("HSTW", repeat=n)):
            assert hst.normal_form(w) == hst._normal_form_by_decompose(w), w


def test_normal_form_rewriting_matches_decompose_random_words():
    rng = random.Random(12)
    for _ in range(400):
        s = _random_word(rng, rng.randrange(0, 90))
        if rng.random() < 0.3:
            s = _pad_identities(rng, s, 3)
        nf = hst.normal_form(s)
        assert nf == hst._normal_form_by_decompose(s)
        assert hst.normal_form(hst.hst_to_ring(s)) == nf


def test_normal_form_rewriting_on_tool_output():
    for seq in TRASYN:
        s = hst.from_trasyn(seq)
        assert hst.normal_form(s) == hst._normal_form_by_decompose(s)
    for _theta, _digits, seq in GRIDSYNTH:
        s = hst.from_gridsynth(seq)
        assert hst.normal_form(s) == hst._normal_form_by_decompose(s) == s


def test_normal_form_product():
    rng = random.Random(13)
    for _ in range(300):
        parts = [_random_word(rng, rng.randrange(0, 25)) for _ in range(rng.randrange(1, 6))]
        joined = "".join(parts)
        assert hst.normal_form_product(*parts) == hst.normal_form(joined)
        assert hst.compact_form_product(*parts) == hst.compact_form(joined)
        # factors already in normal form
        nfs = [hst.normal_form(p) for p in parts]
        assert hst.normal_form_product(*nfs) == hst.normal_form(joined)
    assert hst.normal_form_product() == ""
    with pytest.raises(ValueError):
        hst.normal_form_product("HT", "X")


def test_compact_tails_cover_all_cliffords():
    tails = hst._compact_tails()
    assert len(tails) == 192
    for tail, comp in tails.items():
        assert hst.hst_equal(tail, comp) and len(comp.rstrip("W")) <= 6


def test_align_phase_product_matches_align_phase():
    rng = random.Random(14)
    for _ in range(100):
        parts = [_random_word(rng, rng.randrange(0, 25)) for _ in range(rng.randrange(1, 6))]
        th, ph, la = (rng.uniform(-math.pi, math.pi) for _ in range(3))
        c, s_ = math.cos(th / 2), math.sin(th / 2)
        U = np.array([[c, -np.exp(1j * la) * s_],
                      [np.exp(1j * ph) * s_, np.exp(1j * (ph + la)) * c]])
        try:
            want = hst.align_phase("".join(parts), U)
        except ValueError:
            with pytest.raises(ValueError):
                hst.align_phase_product(parts, U)
            continue
        assert hst.align_phase_product(parts, U) == want   # residual bit for bit
