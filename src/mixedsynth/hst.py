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

``compact_form`` is the same normal form with each Clifford tail written as
its shortest H/S word (at most 6 letters instead of 9). It is the form for
training labels; ``normal_form`` is the form that matches gridsynth.

For strings both are computed by rewriting, letter by letter, with tables of
the 192 Cliffords (no matrix arithmetic; linear in the length; see the
comment above ``_nf_tables``). ``normal_form_product(*seqs)`` and
``compact_form_product(*seqs)`` give the form of a product of strings and
reuse the cached state of the first factor. A ``MAT_D_OMEGA`` input goes
through pyLIQTR's exact decomposition, which the tests use as the reference.

Exact matrices
--------------
``hst_to_ring`` returns a ``MAT_D_OMEGA`` over D[ω] = Z[ω, 1/√2], reduced to the
smallest denominator exponent k, so that two strings are the same unitary
exactly when ``ring_key`` agrees. ``hst_product_to_ring(*seqs)`` does the same
for a product from per-factor cached matrices, and
``align_phase_product(seqs, target)`` aligns the phase of a product.

Phase alignment
---------------
Trasyn and pyLIQTR return sequences only up to global phase. ``align_phase``
picks the W^d that brings a string closest in phase to a numeric target and
reports the leftover continuous phase.
"""

import itertools
import math
import re
from functools import lru_cache

import numpy as np

from ._pyliqtr.matrices import MAT_D_OMEGA
from ._pyliqtr.rings import Z_OMEGA, is_reducible, reduce as _reduce_sqrt2
from ._pyliqtr import exact_decomp as _ed

__all__ = [
    "from_gridsynth", "from_pyliqtr", "from_trasyn",
    "hst_to_ring", "hst_product_to_ring", "hst_to_matrix", "ring_key", "hst_equal",
    "normal_form", "normal_form_product", "is_normal_form", "split_normal_form",
    "clifford_word", "compact_form", "compact_form_product", "is_compact_form",
    "align_phase", "align_phase_product", "t_count", "to_circuit_order", "is_hst",
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
    # Same steps as rings.is_reducible / rings.reduce, on plain integers.
    a = [(e.a1, e.a2, e.a3, e.a4) for e in m.elements()]
    k = m.k
    if any(any(t) for t in a):
        while all((a1 + a3) % 2 == 0 and (a2 + a4) % 2 == 0 for a1, a2, a3, a4 in a):
            a = [((a2 - a4) // 2, (a1 + a3) // 2, (a2 + a4) // 2, (a3 - a1) // 2)
                 for a1, a2, a3, a4 in a]
            k -= 1
    return MAT_D_OMEGA(*(Z_OMEGA(*t) for t in a), k)


_RING_BLOCK = 6   # hst_to_ring multiplies precomputed products of this many letters


@lru_cache(maxsize=1 << 14)
def _block_ring(block: str) -> MAT_D_OMEGA:
    """Exact reduced product of a short block of H/S/T/W letters (cached)."""
    m = _GATE[block[0]]
    for c in block[1:]:
        m = m @ _GATE[c]
    return _canonical(m)


def hst_to_ring(seq: str) -> MAT_D_OMEGA:
    """
    Exact matrix of an HST string over D[ω], with minimal denominator exponent.

    ``W`` letters may appear anywhere (they are scalars). The string is
    multiplied in blocks of ``_RING_BLOCK`` letters whose products are cached;
    the result is exact and reduced, so it does not depend on the blocking.
    """
    _check_hst(seq)
    m = MAT_D_OMEGA.I()
    for i in range(0, len(seq), _RING_BLOCK):
        m = m @ _block_ring(seq[i:i + _RING_BLOCK])
    return _canonical(m)


@lru_cache(maxsize=1 << 14)
def _factor_ring(seq: str) -> MAT_D_OMEGA:
    return hst_to_ring(seq)


def hst_product_to_ring(*seqs) -> MAT_D_OMEGA:
    """
    Exact matrix of the product ``seqs[0]·seqs[1]·…`` of HST strings (matrix
    order), equal to ``hst_to_ring("".join(seqs))``.

    Each factor's matrix is cached per string, so products of recurring
    factors (for example Rz3·H·Rz2·H·Rz1 over a few dozen Rz candidates) cost
    one matrix product per factor.
    """
    m = MAT_D_OMEGA.I()
    for s in seqs:
        m = m @ _factor_ring(s)
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


# ---------------------------------------------------------------------------
# Normal form by rewriting (strings): no matrix arithmetic per call
# ---------------------------------------------------------------------------
#
# A word in normal form is a syllable prefix P = (T)?(HT|SHT)* followed by a
# Clifford C = K·X^b·S^c·ω^d (gridsynth layout, 192 elements with phase).
# Multiplying on the right by a letter keeps that shape:
#
#   H, S, W :  P·C·g  = P·(C·g)                        (C·g is a Clifford)
#   T       :  P·C·T  = P·K·T·C',  C' = T†·X^b·S^c·ω^d·T  (a Clifford, since
#              X^b·S^c·ω^d normalises T up to a Clifford: T†·S·T = S,
#              T†·X·T = ω·S†·X)
#              K = H or SH : append the syllable K·T;
#              K = ""      : if P is empty, P = "T"; otherwise the last
#                            syllable σ'·T of P meets the new T, σ'·T·T =
#                            σ'·S, and P loses that syllable while the
#                            Clifford becomes σ'·S·C'.
#
# The result is always in the (T)?(HT|SHT)* C layout and equals the input
# exactly, so by uniqueness of the Matsumoto–Amano form it is the normal form.
# The tables below are built once from exact ring matrices.

def _clifford_index(K, b, c, d):
    return ((("", "H", "SH").index(K) * 2 + b) * 4 + c) * 8 + d


@lru_cache(maxsize=1)
def _nf_tables():
    """
    (words, right, t_step):
      words[i]   gridsynth-layout word of Clifford i (0..191)
      right[g][i] index of Clifford_i · g for g in "H", "S", "W"
      t_step[i]  (K, j) with Clifford_i · T = K · T · Clifford_j
    """
    table = _clifford_table()
    words = [None] * 192
    for K in ("", "H", "SH"):
        for b in (0, 1):
            for c in range(4):
                for d in range(8):
                    words[_clifford_index(K, b, c, d)] = clifford_word(K, b, c, d)
    index = {ring_key(w): i for i, w in enumerate(words)}
    if len(index) != 192 or any(table[k] != words[i] for k, i in index.items()):
        raise AssertionError("Clifford index does not match the Clifford table")
    right = {g: [index[ring_key(w + g)] for w in words] for g in ("H", "S", "W")}
    t_step = []
    for i, w in enumerate(words):
        K = ("", "H", "SH")[i // 64]
        rest = w[len(K):]                       # X^b·S^c·ω^d
        j = index[ring_key("T" * 7 + rest + "T")]
        if ring_key(w + "T") != ring_key(K + "T" + words[j]):
            raise AssertionError("bad T rewrite for Clifford %r" % w)
        t_step.append((K, j))
    return words, right, tuple(t_step)


def _nf_feed(sylls, cl, seq, tables):
    """
    Right-multiply the normal-form state (syllable list ``sylls``, modified in
    place, and Clifford index ``cl``) by the letters of ``seq``; returns the
    new Clifford index.
    """
    words, right, t_step = tables
    rH, rS, rW = right["H"], right["S"], right["W"]
    for g in seq:
        if g == "H":
            cl = rH[cl]
        elif g == "S":
            cl = rS[cl]
        elif g == "W":
            cl = rW[cl]
        else:  # "T"
            K, cl = t_step[cl]
            if K:
                sylls.append(K + "T")
            elif not sylls:
                sylls.append("T")
            else:
                # σ'·T·T·C' = σ'·S·C'
                c2 = rS[_clifford_index(sylls.pop()[:-1], 0, 0, 0)]
                for h in words[cl]:
                    c2 = right[h][c2]
                cl = c2
    return cl


@lru_cache(maxsize=1 << 16)
def _nf_state(seq: str) -> tuple:
    """(syllables, Clifford index) of the normal form of an HST string (cached)."""
    sylls = []
    cl = _nf_feed(sylls, 0, seq, _nf_tables())
    return tuple(sylls), cl


def _nf_string(seq: str) -> str:
    """Gridsynth-layout normal form of an HST string by rewriting."""
    sylls, cl = _nf_state(seq)
    return "".join(sylls) + _nf_tables()[0][cl]


def normal_form_product(*seqs) -> str:
    """
    Normal form (gridsynth layout) of the product ``seqs[0]·seqs[1]·…`` of HST
    strings, in matrix order; the same string as ``normal_form("".join(seqs))``.

    The rewriting state after ``seqs[0]`` is cached per string, so products
    that share a left factor (for example Rz3·H·Rz2·H·Rz1 over many Rz2, Rz1)
    only pay for the later factors, letter by letter, without matrix
    arithmetic.
    """
    for s in seqs:
        _check_hst(s)
    if not seqs:
        return ""
    tables = _nf_tables()
    sylls, cl = _nf_state(seqs[0])
    sylls = list(sylls)
    for s in seqs[1:]:
        cl = _nf_feed(sylls, cl, s, tables)
    return "".join(sylls) + tables[0][cl]


def compact_form_product(*seqs) -> str:
    """Compact normal form of the product ``seqs[0]·seqs[1]·…`` (see
    :func:`normal_form_product` and :func:`compact_form`)."""
    return _compact_from_nf(normal_form_product(*seqs))


def normal_form(seq) -> str:
    """
    Matsumoto–Amano normal form of an HST string (or a ``MAT_D_OMEGA``).

    Returns ``(T)?(HT|SHT)* K (HSSH)^b S^c W^d`` (see the module docstring).
    Global phase is kept exactly: the result is the same matrix as the input.
    Its T-count is minimal over all Clifford+T words for that unitary.

    A string is rewritten letter by letter (see the comment above
    ``_nf_tables``; linear in its length, results cached per string); a
    ``MAT_D_OMEGA`` goes through pyLIQTR's exact decomposition
    (``_normal_form_by_decompose``, the reference the tests compare with).
    """
    if isinstance(seq, str):
        _check_hst(seq)
        return _nf_string(seq)
    return _normal_form_by_decompose(seq)


def _normal_form_by_decompose(seq) -> str:
    """
    Reference implementation of :func:`normal_form` through exact matrices and
    pyLIQTR's SO(3) ``decompose`` (used for ``MAT_D_OMEGA`` input and in tests).
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
# Compact normal form: shortest Clifford words
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shortest_clifford():
    """
    Gridsynth-layout Clifford (phase stripped) -> (shortest H/S word, its phase d).

    Words are searched by length, then alphabetically (H before S), so the
    choice is deterministic. The longest is 6 letters (gridsynth layout: 9).
    """
    table = _clifford_table()
    best = {}
    n = 0
    while len(best) < 24:
        for w in map("".join, itertools.product("HS", repeat=n)):
            g = table[ring_key(w)]
            cls = g.rstrip("W")
            if cls not in best:
                best[cls] = (w, len(g) - len(cls))
        n += 1
    return best


def compact_form(seq) -> str:
    """
    Compact normal form: the normal form with each Clifford tail written as its
    shortest H/S word.

    ``(T)?(HT | SHT)*  C  W^d`` where C is the shortest word for the Clifford
    (ties broken alphabetically) and d the exact phase. Like
    :func:`normal_form` it is unique per unitary, phase included, and has the
    minimal T-count; it differs only in how the Clifford is spelled. Use this
    form for training labels; use :func:`normal_form` to compare with gridsynth.
    """
    return _compact_from_nf(normal_form(seq))


@lru_cache(maxsize=1)
def _compact_tails():
    """Gridsynth-layout Clifford tail (with W^d) -> compact tail, all 192, checked exactly."""
    out = {}
    for tail in _clifford_table().values():
        cls = tail.rstrip("W")
        word, d_word = _shortest_clifford()[cls]
        comp = word + "W" * ((len(tail) - len(cls) - d_word) % 8)
        if ring_key(comp) != ring_key(tail):
            raise AssertionError("compact Clifford word %r is not %r" % (comp, tail))
        out[tail] = comp
    return out


def _compact_from_nf(nf: str) -> str:
    """Compact form of a string already in gridsynth-layout normal form."""
    mo = _NF_RE.match(nf)
    prefix = (mo.group(1) or "") + mo.group(2)
    return prefix + _compact_tails()[nf[len(prefix):]]


def is_compact_form(seq: str) -> bool:
    """True if ``seq`` is in compact normal form."""
    mo = re.match(r"^(T?(?:HT|SHT)*)([HS]*?)(W{0,7})$", seq)
    if not mo:
        return False
    words = {w for w, _ in _shortest_clifford().values()}
    return mo.group(2) in words and compact_form(seq) == seq


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
    return _align_gates(gates, hst_to_ring(gates).to_array(), target)


def align_phase_product(seqs, target) -> tuple:
    """
    :func:`align_phase` of the product ``seqs[0]·seqs[1]·…`` of HST strings:
    the same ``(aligned, residual)`` as ``align_phase("".join(seqs), target)``,
    with the exact matrix assembled from per-factor cached matrices
    (:func:`hst_product_to_ring`).
    """
    gates = [s.replace("W", "") for s in seqs]
    for g in gates:
        _check_hst(g)
    return _align_gates("".join(gates), hst_product_to_ring(*gates).to_array(), target)


def _align_gates(gates, V, target):
    """align_phase for a W-free string ``gates`` with numeric matrix ``V``."""
    U = np.array([[complex(target[i, j]) for j in range(2)] for i in range(2)])
    ov = np.trace(V.conj().T @ U)
    if abs(ov) < 1e-12:
        raise ValueError("sequence and target are orthogonal; phase is undefined")
    alpha = math.atan2(ov.imag, ov.real)
    d = int(round(alpha / (math.pi / 4))) % 8
    r = math.remainder(alpha - d * math.pi / 4, 2 * math.pi)
    return gates + "W" * d, r
