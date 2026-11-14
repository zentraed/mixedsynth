"""
H/S/T/W sequences: one alphabet for Clifford+T words from any synthesis tool.

Alphabet
--------
An *HST string* uses four letters, in **matrix order** ("HTS" is the matrix
H·T·S, so S is applied first):

    H, S, T   the gates
    W         the global phase ω = exp(iπ/4)

Phase letters are collected at the end of the string, at most seven of them
(ω⁸ = 1). The identity is the empty string. Every rewrite below is exact,
phase included.

Converting tool output
----------------------
    from_gridsynth(s)   newsynth/gridsynth strings: H S T X Y Z E W I - i
    from_pyliqtr(s)     pyLIQTR / mixedsynth strings: H S Sd X Y Z T
    from_trasyn(s)      Trasyn strings: lowercase h s t x y z w i

    X -> HSSH      Y -> HSSHSS·W²     Z -> SS       Sd -> SSS
    E -> HSSS·W³   I -> (empty)       - -> W⁴       i  -> W²

Normal form
-----------
``normal_form`` returns the Matsumoto–Amano normal form in gridsynth's layout,
written in H, S, T, W:

    (T)? (HT | SHT)*  K  (HSSH)^b  S^c  W^d
    K ∈ {"", "H", "SH"},  b ∈ {0, 1},  c ∈ {0..3},  d ∈ {0..7}

K, b, c pick one of the 24 Cliffords (newsynth's ``clifford_decompose_coset``)
and d is the exact global phase. The form is unique per unitary, phase
included, and has the minimal T-count.

Exact matrices
--------------
``hst_to_ring`` returns a ``MAT_D_OMEGA`` over D[ω] = Z[ω, 1/√2], reduced to the
smallest denominator exponent k, so that two strings are the same unitary
exactly when ``ring_key`` agrees.

Phase alignment
---------------
Trasyn and pyLIQTR return sequences only up to global phase. ``align_phase``
picks the W^d that brings a string closest in phase to a numeric target and
reports the leftover continuous phase.
"""

import math
import re
from functools import lru_cache

import numpy as np

from ._pyliqtr.matrices import MAT_D_OMEGA
from ._pyliqtr.rings import Z_OMEGA, is_reducible, reduce as _reduce_sqrt2
from ._pyliqtr import exact_decomp as _ed

__all__ = [
    "from_gridsynth", "from_pyliqtr", "from_trasyn",
    "hst_to_ring", "hst_to_matrix", "ring_key", "hst_equal",
    "normal_form", "is_normal_form", "split_normal_form", "clifford_word",
    "align_phase", "t_count", "to_circuit_order", "is_hst",
]

# ---------------------------------------------------------------------------
# Rewriting tool output into H, S, T, W
# ---------------------------------------------------------------------------

# token -> (gates in H/S/T, number of ω phases)
_REWRITE = {
    "H": ("H", 0), "S": ("S", 0), "T": ("T", 0), "W": ("", 1),
    "X": ("HSSH", 0),
    "Y": ("HSSHSS", 2),          # Y = i·X·Z
    "Z": ("SS", 0),
    "Sd": ("SSS", 0),
    "E": ("HSSS", 3),            # E = H·S³·ω³ (newsynth)
    "I": ("", 0),
    "-": ("", 4),                # -I
    "i": ("", 2),                # iI
}

_GRIDSYNTH_RE = re.compile(r"[HSTXYZEWI\-i]")
_PYLIQTR_RE = re.compile(r"Sd|[HSXYZT]")
_TRASYN_RE = re.compile(r"[hstxyzwi]")
_HST_RE = re.compile(r"^[HST]*W{0,7}$")
_NF_RE = re.compile(r"^(T)?((?:HT|SHT)*)(SH|H)?(HSSH)?(S{0,3})(W{0,7})$")


def _tokenize(seq, pattern, source):
    tokens = pattern.findall(seq)
    if "".join(tokens) != seq:
        bad = pattern.sub("", seq)
        raise ValueError("unrecognised %s token(s) %r in %r" % (source, bad, seq))
    return tokens


def _rewrite(tokens):
    gates = []
    phase = 0
    for tok in tokens:
        g, w = _REWRITE[tok]
        gates.append(g)
        phase += w
    return "".join(gates) + "W" * (phase % 8)


def from_gridsynth(seq: str) -> str:
    """
    Rewrite a newsynth/gridsynth gate string into an HST string.

    Accepts gridsynth's whole interchange alphabet: ``H S T X Y Z E W``, ``I``
    (identity, which is also how gridsynth prints the empty word), ``-`` (−I)
    and ``i`` (i·I). The string is in matrix order, as gridsynth prints it.
    """
    return _rewrite(_tokenize(seq.strip(), _GRIDSYNTH_RE, "gridsynth"))


def from_pyliqtr(seq: str) -> str:
    """
    Rewrite a pyLIQTR / mixedsynth gate string into an HST string.

    Accepts the tokens ``H S Sd X Y Z T`` produced by
    ``exact_decomp_to_matrix_string`` and returned by ``harvest_rz`` and
    ``optimize_u3_hardcap``. These strings are correct only up to global
    phase; use :func:`align_phase` to fix the phase against a target.
    """
    return _rewrite(_tokenize(seq, _PYLIQTR_RE, "pyLIQTR"))


def from_trasyn(seq: str) -> str:
    """
    Rewrite a Trasyn gate string into an HST string.

    Accepts Trasyn's lowercase gate letters ``h s t x y z`` plus ``w`` and
    ``i``. Trasyn strings are in matrix order (``trasyn.seq2mat`` multiplies
    left to right). Trasyn optimises fidelity, which ignores global phase, so
    the phase of the result is arbitrary; use :func:`align_phase`.
    """
    toks = _tokenize(seq, _TRASYN_RE, "Trasyn")
    return _rewrite([t.upper() if t != "i" else "I" for t in toks])


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def is_hst(seq: str) -> bool:
    """True if ``seq`` is H/S/T gates followed by at most seven ``W``."""
    return bool(_HST_RE.match(seq))


def _check_hst(seq):
    if not isinstance(seq, str) or not re.fullmatch(r"[HSTW]*", seq):
        raise ValueError("not an HST string (letters H, S, T, W only): %r" % (seq,))


def t_count(seq: str) -> int:
    """Number of T gates in an HST string."""
    _check_hst(seq)
    return seq.count("T")


def to_circuit_order(seq: str) -> list:
    """
    Gates of an HST string in application order (first gate applied first).

    ``W`` letters are dropped and the total phase is returned as one ``"W"*d``
    entry at the end, if nonzero.
    """
    _check_hst(seq)
    gates = [c for c in reversed(seq) if c != "W"]
    d = seq.count("W") % 8
    return gates + (["W" * d] if d else [])


# ---------------------------------------------------------------------------
# Exact matrices over D[ω]
# ---------------------------------------------------------------------------

_W = MAT_D_OMEGA(Z_OMEGA(0, 0, 1, 0), Z_OMEGA(0, 0, 0, 0),
                 Z_OMEGA(0, 0, 0, 0), Z_OMEGA(0, 0, 1, 0), 0)
_GATE = {"H": MAT_D_OMEGA.H(), "S": MAT_D_OMEGA.S(), "T": MAT_D_OMEGA.T(), "W": _W}


def _canonical(m: MAT_D_OMEGA) -> MAT_D_OMEGA:
    """Divide out factors of √2 until the denominator exponent k is minimal."""
    entries = m.elements()
    k = m.k
    while all(is_reducible(e) for e in entries) and any(e != 0 for e in entries):
        entries = [_reduce_sqrt2(e) for e in entries]
        k -= 1
    return MAT_D_OMEGA(*entries, k)


def hst_to_ring(seq: str) -> MAT_D_OMEGA:
    """
    Exact matrix of an HST string over D[ω], with minimal denominator exponent.

    ``W`` letters may appear anywhere (they are scalars).
    """
    _check_hst(seq)
    m = MAT_D_OMEGA.I()
    for c in seq:
        m = m @ _GATE[c]
    return _canonical(m)


def ring_key(m) -> tuple:
    """
    Hashable exact identity of a unitary: an HST string or a ``MAT_D_OMEGA``.

    Two arguments have equal keys exactly when they are the same matrix,
    global phase included.
    """
    if isinstance(m, str):
        m = hst_to_ring(m)
    else:
        m = _canonical(m)
    return (m.k,) + tuple((e.a1, e.a2, e.a3, e.a4) for e in m.elements())


def hst_equal(a: str, b: str) -> bool:
    """True if two HST strings are exactly the same matrix, phase included."""
    return ring_key(a) == ring_key(b)


def hst_to_matrix(seq: str, dps=None):
    """
    Unitary of an HST string, from the exact ring matrix.

    ``dps=None`` returns a complex128 numpy array; an integer returns a 2×2
    mpmath matrix evaluated at that many decimal places.
    """
    m = hst_to_ring(seq)
    if dps is None:
        return m.to_array()

    import mpmath as mp
    with mp.workdps(dps + 10):
        w = mp.expjpi(mp.mpf(1) / 4)
        s2k = mp.sqrt(2) ** m.k

        def val(z):
            return (z.a1 * w**3 + z.a2 * w**2 + z.a3 * w + z.a4) / s2k

        out = mp.matrix([[val(m.a00), val(m.a01)], [val(m.a10), val(m.a11)]])
    return out


# ---------------------------------------------------------------------------
# Normal form
# ---------------------------------------------------------------------------

def clifford_word(K: str, b: int, c: int, d: int) -> str:
    """
    The HST word for the Clifford K·X^b·S^c·ω^d in gridsynth's layout.

    K is "", "H" or "SH"; b in {0, 1}; c in {0..3}; d in {0..7}.
    """
    if K not in ("", "H", "SH") or b not in (0, 1) or not 0 <= c <= 3 or not 0 <= d <= 7:
        raise ValueError("bad Clifford coordinates %r" % ((K, b, c, d),))
    return K + "HSSH" * b + "S" * c + "W" * d


@lru_cache(maxsize=1)
def _clifford_table():
    """ring_key -> HST word, for all 192 Cliffords with phase."""
    table = {}
    for K in ("", "H", "SH"):
        for b in (0, 1):
            for c in range(4):
                for d in range(8):
                    word = clifford_word(K, b, c, d)
                    key = ring_key(word)
                    if key in table:
                        raise AssertionError("Clifford words not unique")
                    table[key] = word
    return table


_DECOMP_TOKENS = ["S", "Sd", "H", "X", "Y", "Z", "T"]


def _syllables(m: MAT_D_OMEGA) -> str:
    """The (T)?(HT|SHT)* part of the normal form, via pyLIQTR's decompose."""
    try:
        toks = _ed.decompose(m.convert_to_so3(), _DECOMP_TOKENS)
    except SystemExit as exc:  # the vendored code exits instead of raising
        raise ValueError("matrix is not an exact Clifford+T unitary") from exc
    word = "".join(reversed(toks))  # decompose returns circuit order
    last_t = word.rfind("T")
    prefix = word[: last_t + 1]
    if not re.fullmatch(r"T?(?:HT|SHT)*", prefix):
        raise AssertionError("unexpected syllable layout from decompose: %r" % word)
    return prefix


def _inverse_word(seq: str) -> str:
    inv = {"H": "H", "S": "SSS", "T": "TTTTTTT", "W": "WWWWWWW"}
    return "".join(inv[c] for c in reversed(seq))


def normal_form(seq) -> str:
    """
    Matsumoto–Amano normal form of an HST string (or a ``MAT_D_OMEGA``).

    Returns ``(T)?(HT|SHT)* K (HSSH)^b S^c W^d`` (see the module docstring).
    Global phase is kept exactly: the result is the same matrix as the input.
    Its T-count is minimal over all Clifford+T words for that unitary.
    """
    m = hst_to_ring(seq) if isinstance(seq, str) else _canonical(seq)
    prefix = _syllables(m)
    rest = _canonical(hst_to_ring(_inverse_word(prefix)) @ m)
    try:
        tail = _clifford_table()[ring_key(rest)]
    except KeyError:
        raise AssertionError("remainder after the syllables is not a Clifford")
    nf = prefix + tail
    if ring_key(nf) != ring_key(m):
        raise AssertionError("normal form does not reproduce the input")
    return nf


def is_normal_form(seq: str) -> bool:
    """True if ``seq`` is written in the normal-form layout."""
    return bool(_NF_RE.match(seq))


def split_normal_form(nf: str) -> dict:
    """
    Parts of a normal-form string.

    Returns ``{"syllables": [...], "K": str, "b": int, "c": int, "d": int}``;
    syllables are "T", "HT" or "SHT" in matrix order.
    """
    mo = _NF_RE.match(nf)
    if not mo:
        raise ValueError("not in normal form: %r" % nf)
    t0, body, K, x, s, w = mo.groups()
    sylls = (["T"] if t0 else []) + re.findall(r"HT|SHT", body)
    return {"syllables": sylls, "K": K or "", "b": 1 if x else 0,
            "c": len(s), "d": len(w)}


# ---------------------------------------------------------------------------
# Phase alignment against a numeric target
# ---------------------------------------------------------------------------

def align_phase(seq: str, target) -> tuple:
    """
    Choose the global phase W^d that best matches a target unitary.

    Parameters
    ----------
    seq    : HST string; any W letters in it are replaced.
    target : 2×2 array-like (numpy or mpmath) with a definite phase.

    Returns
    -------
    (aligned, residual) where ``aligned`` is ``seq`` without its W letters
    followed by the chosen ``"W"*d``, and ``residual`` is the leftover phase
    r in (−π/8, π/8] with  tr(V†·U) ∝ exp(i·r),  V = matrix(aligned), U = target.
    """
    _check_hst(seq)
    gates = seq.replace("W", "")
    V = hst_to_matrix(gates)
    U = np.array([[complex(target[i, j]) for j in range(2)] for i in range(2)])
    ov = np.trace(V.conj().T @ U)
    if abs(ov) < 1e-12:
        raise ValueError("sequence and target are orthogonal; phase is undefined")
    alpha = math.atan2(ov.imag, ov.real)
    d = int(round(alpha / (math.pi / 4))) % 8
    r = math.remainder(alpha - d * math.pi / 4, 2 * math.pi)
    return gates + "W" * d, r
