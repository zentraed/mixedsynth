"""
Clifford+T ring utilities for mixedsynth.

The central function is seq_to_ring_mat, which converts a gate sequence string
(in Trasyn or gridsynth format) to an exact MAT_D_OMEGA matrix over the ring
D[ω] = Z[ω, 1/√2].  This is the inverse direction of exact_decomp: instead of
decomposing a ring matrix into a gate sequence, it builds the ring matrix from
an already-known sequence.

Convention (verified against trasyn.seq2mat):
    seq = "abc..." → result = G_a @ G_b @ G_c @ ...
    i.e., characters are composed left-to-right via right-multiplication.
    The first character corresponds to the first gate applied to the state.

Supported characters
--------------------
    h  Hadamard
    t  T gate  (π/8 rotation, T = diag(1, ω) where ω = e^{iπ/4})
    s  S gate  (T² = diag(1, i))
    x  Pauli X
    y  Pauli Y
    z  Pauli Z
    i  Identity (optional; silently skipped)

All of these are elements of the Clifford+T group and have exact representations
over Z[ω].  A global-phase gate 'w' (ω·I) is also supported for completeness.
"""

import numpy as np


from mixedsynth._pyliqtr.matrices import MAT_D_OMEGA
from mixedsynth._pyliqtr.rings import Z_OMEGA

# ---------------------------------------------------------------------------
# Gate map: single character → MAT_D_OMEGA
# ---------------------------------------------------------------------------

_W = MAT_D_OMEGA(
    Z_OMEGA(0, 0, 1, 0), Z_OMEGA(0, 0, 0, 0),
    Z_OMEGA(0, 0, 0, 0), Z_OMEGA(0, 0, 1, 0),
    0,
)

_GATE_MAP = {
    'h': MAT_D_OMEGA.H(),
    't': MAT_D_OMEGA.T(),
    's': MAT_D_OMEGA.S(),
    'x': MAT_D_OMEGA.X(),
    'y': MAT_D_OMEGA.Y(),
    'z': MAT_D_OMEGA.Z(),
    'i': MAT_D_OMEGA.I(),
    'w': _W,
}


def seq_to_ring_mat(seq: str) -> MAT_D_OMEGA:
    """
    Convert a Clifford+T gate sequence string to an exact MAT_D_OMEGA.

    Parameters
    ----------
    seq : str
        Gate sequence using the character alphabet {h, t, s, x, y, z, i, w}.
        Characters compose left-to-right: seq="ht" → H @ T (H applied first).

    Returns
    -------
    MAT_D_OMEGA
        Exact 2×2 matrix over D[ω] = Z[ω, 1/√2].
        Call .to_array() for a complex128 numpy array.
        Call .convert_to_so3() for the SO(3) representation.
    """
    result = MAT_D_OMEGA.I()
    for c in seq:
        g = _GATE_MAP.get(c)
        if g is None:
            raise ValueError("Unknown gate character %r in sequence %r" % (c, seq))
        result = result @ g
    return result


def seq_to_np(seq: str) -> np.ndarray:
    """
    Convert a gate sequence to a complex128 numpy array via exact ring arithmetic.

    Equivalent to trasyn.seq2mat but computed over Z[ω] then cast to float64.
    """
    return seq_to_ring_mat(seq).to_array()


def seq_to_mp(seq: str, dps: int = 60):
    """
    Convert a gate sequence to an mpmath matrix at arbitrary precision.

    The ring elements (integers) are exact; only the division by √2^k requires
    mpmath.  The result is a 2×2 mpmath.matrix suitable for use with
    rotation_vectors_mp and other mixedsynth mpmath functions.

    Parameters
    ----------
    seq : str
        Gate sequence string (characters h, t, s, x, y, z, i, w).
    dps : int
        mpmath working precision in decimal digits.

    Returns
    -------
    mpmath.matrix (2×2, complex entries)
    """
    import mpmath as mp

    mat = seq_to_ring_mat(seq)
    with mp.workdps(dps + 10):
        s2k = mp.sqrt(mp.mpf(2)) ** mat.k
        def _to_mp(z):
            w = mp.exp(1j * mp.pi / 4)
            return (mp.mpf(z.a1) * w**3 + mp.mpf(z.a2) * w**2
                    + mp.mpf(z.a3) * w + mp.mpf(z.a4)) / s2k
        return mp.matrix([[_to_mp(mat.a00), _to_mp(mat.a01)],
                          [_to_mp(mat.a10), _to_mp(mat.a11)]])
