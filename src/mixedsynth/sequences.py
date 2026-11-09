"""
Helpers for working with Clifford+T gate sequences produced by mixedsynth.

Sequence convention
-------------------
Every sequence returned by mixedsynth (``harvest_rz(..., with_sequences=True)``,
``optimize_u3_hardcap(...)["sequences"]`` / ``["members"]``) is a string of
gate tokens in **matrix order**, the same convention as pyLIQTR's
``exact_decomp_to_matrix_string``:

    "HTS"  means the matrix  H @ T @ S
           i.e. S is applied to the state first, H last.

Tokens: ``H``, ``S``, ``Sd`` (S†), ``X``, ``Y``, ``Z``, ``T``.  The empty string
is the identity.  Matrices agree with the synthesized unitaries up to a global
phase.

Use :func:`circuit_order` to get the gates in the order they are applied.
"""

import re

import numpy as np

_TOKEN_RE = re.compile(r"Sd|[HSXYZT]")

_SQ2 = 1 / np.sqrt(2)
_GATES = {
    "H":  np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex),
    "S":  np.array([[1, 0], [0, 1j]], dtype=complex),
    "Sd": np.array([[1, 0], [0, -1j]], dtype=complex),
    "X":  np.array([[0, 1], [1, 0]], dtype=complex),
    "Y":  np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z":  np.array([[1, 0], [0, -1]], dtype=complex),
    "T":  np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}


def tokenize(seq):
    """Split a matrix-order sequence string into gate tokens."""
    tokens = _TOKEN_RE.findall(seq)
    if "".join(tokens) != seq:
        raise ValueError("Unrecognised gate token in sequence %r" % seq)
    return tokens


def t_count(seq):
    """Number of T gates in a sequence."""
    return tokenize(seq).count("T")


def circuit_order(seq):
    """Gate tokens in application (circuit) order: first gate applied first."""
    return tokenize(seq)[::-1]


def sequence_to_matrix(seq, dps=None):
    """
    Unitary implemented by a matrix-order sequence.

    dps=None returns a complex128 numpy array; an integer returns an mpmath
    2×2 matrix evaluated at that many decimal places.
    """
    tokens = tokenize(seq)
    if dps is None:
        M = np.eye(2, dtype=complex)
        for g in tokens:
            M = M @ _GATES[g]
        return M

    import mpmath as mp
    with mp.workdps(dps + 10):
        s2 = 1 / mp.sqrt(2)
        w = mp.expjpi(mp.mpf(1) / 4)
        gates = {
            "H":  mp.matrix([[s2, s2], [s2, -s2]]),
            "S":  mp.matrix([[1, 0], [0, mp.j]]),
            "Sd": mp.matrix([[1, 0], [0, -mp.j]]),
            "X":  mp.matrix([[0, 1], [1, 0]]),
            "Y":  mp.matrix([[0, -mp.j], [mp.j, 0]]),
            "Z":  mp.matrix([[1, 0], [0, -1]]),
            "T":  mp.matrix([[1, 0], [0, w]]),
        }
        M = mp.eye(2)
        for g in tokens:
            M = M * gates[g]
        return M


def sample(members, rng=None, size=None):
    """
    Draw sequence(s) from a synthesized mixed channel.

    members : the ``"members"`` list returned by ``optimize_u3_hardcap``
              ([{"weight", "t_count", "sequence"}, ...])
    rng     : numpy Generator (default: fresh ``np.random.default_rng()``)
    size    : None for a single sequence string, or an int for a list

    Running a freshly sampled sequence on every shot implements the mixed
    channel Σ_i w_i U_i(·)U_i†.
    """
    rng = np.random.default_rng() if rng is None else rng
    w = np.array([float(m["weight"]) for m in members])
    w = w / w.sum()
    idx = rng.choice(len(members), p=w, size=size)
    if size is None:
        return members[int(idx)]["sequence"]
    return [members[int(i)]["sequence"] for i in idx]
