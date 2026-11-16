"""
Controlled single-qubit gates: C-V for V = e^{iγ}·U3(θ,φ,λ).

Why not just control the synthesized members?
---------------------------------------------
Every Clifford+T sequence mixedsynth returns is correct only up to a global
phase, and the mixed-channel optimizer works in SU(2) (strip_phase).  Under a
control, a member's global phase becomes a *relative* phase, so a mixture of
controlled members is not close to C-V (errors are O(1), whatever the T
budget).  Instead C-V is built from gates that act unconditionally, where a
global phase really is global.

Decomposition (Nielsen & Chuang, Cor. 4.2 / Fig. 4.6)
-----------------------------------------------------
Write V = e^{iα}·W with W = Rz(β)·Ry(γ')·Rz(δ) ∈ SU(2).  For V = e^{iγ}U3(θ,φ,λ)
(U3 = e^{i(φ+λ)/2}·Rz(φ)·Ry(θ)·Rz(λ)) this is β = φ, γ' = θ, δ = λ and
α = γ + (φ+λ)/2.  With

    A = Rz(β)·Ry(γ'/2)              ≅ U3(θ/2, φ, 0)
    B = Ry(−γ'/2)·Rz(−(δ+β)/2)      ≅ U3(−θ/2, 0, −(φ+λ)/2)
    C = Rz((δ−β)/2)                 ≅ U3(0, (λ−φ)/2, 0)
    P = diag(1, e^{iα})             = U3(0, 0, α)          (on the control)

ABC = I and A·X·B·X·C = W, so in application order

    target :  C ── ⊕ ── B ── ⊕ ── A
    control:  ──── ● ─────── ● ──── P

implements C-V exactly.  "≅" means equal up to a global phase, which is
harmless here: a phase on A, B or C multiplies both control branches by the
same factor, and P acts on the control unconditionally.  So every piece can be
any (phase-blind) approximation of its U3, and in particular a mixed channel
from optimize_u3_hardcap, drawn independently on each shot.  Each piece's
sequence is rewritten to its normal form (hst.compact_form, phase letters
dropped), which removes the H·H and trivial rotations the U3 template leaves.

Cost and exactness
------------------
A generic C-V needs six non-Clifford rotations (β, γ'/2 twice, (δ±β)/2, α),
about twice a bare U3.  Gates whose pieces are Clifford+T, such as C-H, C-S
and C-X, come out exact.  C-T does not: diag(1,1,1,ω) has determinant ω,
which no ancilla-free two-qubit Clifford+T circuit reaches, and here its
pieces are π/8 rotations that are approximated like any other angle.

Circuits
--------
A circuit is a list of operations in *application order*:

    ("u",  q, seq)    single-qubit Clifford+T string ``seq`` (matrix order,
                      pyLIQTR tokens, as everywhere in mixedsynth) on qubit q
    ("cx", c, t)      CNOT with control c and target t

Qubit 0 is the control and qubit 1 the target; matrices use kron(q0, q1), so
the control is the most significant bit and C-V = |0⟩⟨0|⊗I + |1⟩⟨1|⊗V.

Error metrics
-------------
dn2 is half the diamond distance, as elsewhere in mixedsynth.  For one fixed
circuit it is exact (unitary vs unitary, from the eigenvalues of V†U).  For
the mixed channel two bounds are reported:

    dn2_upper = Σ_pieces dn2(piece)   (each piece's exact qubit Choi
                                       certificate; subadditivity)
    dn2_lower = ‖J_Λ − J_V‖₁ / 2      (normalized 4×4 Choi states, float64)

T budget
--------
``t_total`` is a hard per-shot cap for the whole controlled gate.  It is split
over A, B, C, P in proportion to how many of each piece's three Rz angles are
not multiples of π/4 (those are Clifford or Clifford·T and need at most one T,
which is reserved first), or set explicitly with ``t_split``.  Each piece then
respects its own cap on every shot, so every shot has T ≤ t_total.
"""

import itertools
import math

import numpy as np
import mpmath as mp

from .harvest import rz_angles_for_u3
from .u3 import optimize_u3_hardcap
from .u3_single import synthesize_u3_single
from .diamond import dn2_choi_mp
from .sequences import sequence_to_matrix, sample, t_count
from . import hst

__all__ = [
    "u3_params", "controlled_u3_matrix", "controlled_u3_pieces",
    "controlled_circuit", "circuit_matrix", "circuit_t_count", "split_budget",
    "synthesize_controlled_u3_single", "optimize_controlled_u3",
    "sample_controlled",
]

PIECES = ("A", "B", "C", "P")
_QUBIT = {"A": 1, "B": 1, "C": 1, "P": 0}

_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Exact targets and the decomposition
# ---------------------------------------------------------------------------

def _u3_np(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _u3_mp(theta, phi, lam, dps):
    with mp.workdps(dps + 10):
        t, p, l = mp.mpf(theta), mp.mpf(phi), mp.mpf(lam)
        c, s = mp.cos(t / 2), mp.sin(t / 2)
        return mp.matrix([[c, -mp.expj(l) * s],
                          [mp.expj(p) * s, mp.expj(p + l) * c]])


def u3_params(V):
    """
    (θ, φ, λ, γ) with V = e^{iγ}·U3(θ,φ,λ) for a 2×2 unitary V (numpy or
    mpmath).  θ ∈ [0, π].  Where φ, λ are not separately defined (θ = 0 or π)
    one of them is set to 0.
    """
    V = np.array([[complex(V[i, j]) for j in range(2)] for i in range(2)])
    det = V[0, 0] * V[1, 1] - V[0, 1] * V[1, 0]
    if abs(abs(det) - 1) > 1e-8 or not np.allclose(V.conj().T @ V, _I2, atol=1e-8):
        raise ValueError("V is not unitary")
    theta = 2 * math.atan2(abs(V[1, 0]), abs(V[0, 0]))
    tol = 1e-12
    if abs(V[0, 0]) > tol:
        gamma = np.angle(V[0, 0])
        if abs(V[1, 0]) > tol:
            phi = np.angle(V[1, 0]) - gamma
            lam = np.angle(-V[0, 1]) - gamma
        else:
            phi, lam = 0.0, np.angle(V[1, 1]) - gamma
    else:
        lam = 0.0
        gamma = np.angle(-V[0, 1])
        phi = np.angle(V[1, 0]) - gamma
    return float(theta), float(phi), float(lam), float(gamma)


def controlled_u3_matrix(theta, phi, lam, gamma=0.0):
    """Exact C-V, V = e^{iγ}U3(θ,φ,λ), as a 4×4 complex128 matrix."""
    M = np.eye(4, dtype=complex)
    M[2:, 2:] = np.exp(1j * gamma) * _u3_np(theta, phi, lam)
    return M


def controlled_u3_pieces(theta, phi, lam, gamma=0.0):
    """
    The four single-qubit pieces of C-V, V = e^{iγ}U3(θ,φ,λ).

    Returns {"A", "B", "C", "P"} → {"qubit", "u3": (θ', φ', λ')}: A, B, C act
    on the target (qubit 1), P on the control (qubit 0).  A, B, C equal their
    U3 up to global phase; P = U3(0, 0, α) exactly, α = γ + (φ+λ)/2.
    """
    alpha = gamma + (phi + lam) / 2
    u3s = {
        "A": (theta / 2, phi, 0.0),
        "B": (-theta / 2, 0.0, -(phi + lam) / 2),
        "C": (0.0, (lam - phi) / 2, 0.0),
        "P": (0.0, 0.0, alpha),
    }
    return {k: {"qubit": _QUBIT[k], "u3": u3s[k]} for k in PIECES}


def controlled_circuit(seqs):
    """
    Circuit (application order) from piece sequences {"A", "B", "C", "P"}:
    C, CX, B, CX, A on the target, P on the control.
    """
    return [("u", 1, seqs["C"]), ("cx", 0, 1), ("u", 1, seqs["B"]),
            ("cx", 0, 1), ("u", 1, seqs["A"]), ("u", 0, seqs["P"])]


def circuit_matrix(ops):
    """4×4 complex128 unitary of a two-qubit circuit (kron(q0, q1) ordering)."""
    M = np.eye(4, dtype=complex)
    for op in ops:
        if op[0] == "u":
            U = sequence_to_matrix(op[2])
            G = np.kron(U, _I2) if op[1] == 0 else np.kron(_I2, U)
        elif op[0] == "cx":
            if (op[1], op[2]) == (0, 1):
                G = _CX
            elif (op[1], op[2]) == (1, 0):
                SW = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
                G = SW @ _CX @ SW
            else:
                raise ValueError("bad CNOT %r" % (op,))
        else:
            raise ValueError("unknown operation %r" % (op,))
        M = G @ M
    return M


def circuit_t_count(ops):
    """Total T-count of a circuit."""
    return sum(t_count(op[2]) for op in ops if op[0] == "u")


def _simplify(seq):
    """
    The piece's Matsumoto–Amano normal form (shortest Clifford words) with
    its phase letters dropped: the same unitary up to global phase, which is
    all an unconditional piece needs, with minimal T-count.  Removes the
    redundant H·H and trivial rotations the U3 template leaves behind.
    """
    return hst.compact_form(hst.from_pyliqtr(seq)).replace("W", "")


def _simplify_members(members):
    """Simplify every member's sequence; merge members that coincide."""
    merged = {}
    for m in members:
        s = _simplify(m["sequence"])
        if s in merged:
            merged[s]["weight"] = merged[s]["weight"] + m["weight"]
        else:
            merged[s] = {"weight": m["weight"], "t_count": t_count(s),
                         "sequence": s, "u3_sequence": m["sequence"]}
    return sorted(merged.values(), key=lambda m: -float(m["weight"]))


def _unitary_dn2(U, V):
    """
    Exact half diamond distance between unitary channels U·U† and V·V†:
    sqrt(1 − δ²), δ the distance from 0 to the convex hull of eig(V†U).
    """
    ang = np.sort(np.angle(np.linalg.eigvals(V.conj().T @ U)))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    width = 2 * np.pi - gaps.max()        # smallest arc holding every eigenvalue
    if width >= np.pi:
        return 1.0
    return float(math.sin(width / 2))


# ---------------------------------------------------------------------------
# T-budget split
# ---------------------------------------------------------------------------

def _angle_cost(angle):
    """0 for a multiple of π/2, 1 for an odd multiple of π/4, None otherwise."""
    k = angle / (math.pi / 4)
    if abs(k - round(k)) > 1e-9:
        return None
    return int(round(k)) % 2


def split_budget(pieces, t_total):
    """
    Default per-piece T caps (dict A, B, C, P → int, summing to ≤ t_total).

    Each piece first gets one T per Rz angle that is an odd multiple of π/4
    (when the budget allows); the rest is shared equally over all Rz angles
    that are not multiples of π/4, remainders going to pieces in A, B, C, P
    order.
    """
    reserve, hard = {}, {}
    for k in PIECES:
        costs = [_angle_cost(a) for a in rz_angles_for_u3(*pieces[k]["u3"])]
        reserve[k] = sum(c for c in costs if c is not None)
        hard[k] = sum(c is None for c in costs)
    split = {k: 0 for k in PIECES}
    left = t_total
    for k in PIECES:
        r = min(reserve[k], left)
        split[k] += r
        left -= r
    n_hard = sum(hard.values())
    if n_hard == 0:
        return split
    per, rem = divmod(left, n_hard)
    for k in PIECES:
        split[k] += per * hard[k]
    for k in PIECES:
        if rem == 0:
            break
        if hard[k]:
            extra = min(rem, hard[k])
            split[k] += extra
            rem -= extra
    return split


def _resolve_split(pieces, t_total, t_split):
    if t_split is None:
        if t_total is None:
            raise ValueError("give t_total or t_split")
        return split_budget(pieces, t_total)
    if not isinstance(t_split, dict):
        t_split = dict(zip(PIECES, t_split))
    if set(t_split) != set(PIECES):
        raise ValueError("t_split needs exactly the pieces A, B, C, P")
    if t_total is not None and sum(t_split.values()) > t_total:
        raise ValueError("t_split sums to more than t_total")
    return {k: int(t_split[k]) for k in PIECES}


# ---------------------------------------------------------------------------
# Single circuit
# ---------------------------------------------------------------------------

def synthesize_controlled_u3_single(theta, phi, lam, t_total=None, gamma=0.0,
                                    t_split=None, n_keep=5, t_min=0,
                                    stride=None, dps=60, odd_t=True):
    """
    One Clifford+T circuit for C-V, V = e^{iγ}U3(θ,φ,λ), with T ≤ t_total.

    Each piece is the best single sequence for its U3 from
    :func:`synthesize_u3_single` within its T cap (``t_split``, default
    :func:`split_budget`).  The pieces are chosen independently, not jointly.

    Returns
    -------
    None if a piece has no candidate, else
      circuit  : operations in application order (module docstring)
      t_count  : total T-count
      t_split  : {"A", "B", "C", "P"} → T cap used per piece
      dn2      : exact half diamond distance of the circuit to C-V
      pieces   : {"A", "B", "C", "P"} → {"qubit", "u3", "t_budget",
                 "sequence", "t_count", "dn2", "u3_sequence"}; "sequence" is
                 the synthesized "u3_sequence" rewritten to normal form
                 (same unitary up to phase, T-count never higher); a
                 piece's dn2 is its own (phase-insensitive) qubit error, and
                 Σ dn2 bounds the total
    """
    pieces = controlled_u3_pieces(theta, phi, lam, gamma)
    split = _resolve_split(pieces, t_total, t_split)
    seqs, out_pieces = {}, {}
    for k in PIECES:
        r = synthesize_u3_single(*pieces[k]["u3"], split[k], n_keep=n_keep,
                                 t_min=t_min, stride=stride, dps=dps, odd_t=odd_t)
        if r is None:
            return None
        seqs[k] = _simplify(r["sequence"])
        out_pieces[k] = dict(pieces[k], t_budget=split[k], sequence=seqs[k],
                             t_count=t_count(seqs[k]), dn2=float(r["err"]),
                             u3_sequence=r["sequence"])
    ops = controlled_circuit(seqs)
    return {
        "circuit": ops,
        "t_count": circuit_t_count(ops),
        "t_split": split,
        "dn2": _unitary_dn2(circuit_matrix(ops), controlled_u3_matrix(theta, phi, lam, gamma)),
        "pieces": out_pieces,
    }


# ---------------------------------------------------------------------------
# Mixed channel
# ---------------------------------------------------------------------------

def _choi_state(U):
    d = U.shape[0]
    v = (np.kron(np.eye(d), U) @ (np.eye(d).reshape(-1) / math.sqrt(d)))
    return np.outer(v, v.conj())


def _channel_dn2_lower(pieces, target):
    """Choi trace distance of the product-ensemble channel to C-V (float64)."""
    names = list(PIECES)
    J = -_choi_state(target)
    for combo in itertools.product(*(pieces[k]["members"] for k in names)):
        w = 1.0
        seqs = {}
        for k, m in zip(names, combo):
            w *= float(m["weight"])
            seqs[k] = m["sequence"]
        J += w * _choi_state(circuit_matrix(controlled_circuit(seqs)))
    return 0.5 * float(np.abs(np.linalg.eigvalsh(J)).sum())


def optimize_controlled_u3(theta, phi, lam, t_total=None, gamma=0.0,
                           t_split=None, n_keep=5, n_keep_total=10, dps=60,
                           t_min=0, stride=None, odd_t=True):
    """
    Mixed channel for C-V, V = e^{iγ}U3(θ,φ,λ), with T ≤ t_total on every shot.

    Each piece A, B, C, P is an :func:`optimize_u3_hardcap` mixture (≤ 4
    members) for its U3 within its T cap (``t_split``, default
    :func:`split_budget`).  On each shot draw every piece independently
    (:func:`sample_controlled`); the channel is the product ensemble.

    Returns
    -------
    None if a piece cannot be synthesized, else
      t_split   : {"A", "B", "C", "P"} → T cap per piece (Σ ≤ t_total)
      dn2_upper : Σ of the pieces' certified dn2 (rigorous upper bound)
      dn2_lower : Choi trace distance to C-V (rigorous lower bound, float64)
      eps_sum   : Σ of the pieces' best-single dn2 (optimize_u3_hardcap's
                  "eps"): an upper bound for an unmixed circuit built from
                  the same caps; synthesize_controlled_u3_single gives the
                  actual unmixed baseline
      E_T       : expected T per shot
      t_max     : largest T on any shot (≤ t_total)
      n_prod    : number of distinct circuits in the product ensemble
      pieces    : {"A", "B", "C", "P"} → {"qubit", "u3", "t_budget",
                  "members", "dn2", "eps", "E_T", "t_max", "result"}, with
                  "members" the hardcap members as [{"weight", "t_count",
                  "sequence", "u3_sequence"}, ...] (sequences rewritten to
                  normal form as in synthesize_controlled_u3_single; members
                  that coincide are merged), "dn2" the qubit Choi certificate
                  of the piece's mixture and "result" the full hardcap dict
    """
    pieces = controlled_u3_pieces(theta, phi, lam, gamma)
    split = _resolve_split(pieces, t_total, t_split)
    out_pieces = {}
    for k in PIECES:
        u3 = pieces[k]["u3"]
        r = optimize_u3_hardcap(*u3, split[k], n_keep=n_keep,
                                n_keep_total=n_keep_total, dps=dps, t_min=t_min,
                                stride=stride, odd_t=odd_t)
        if r is None:
            return None
        members = _simplify_members(r["members"])
        Us = [sequence_to_matrix(m["sequence"], dps) for m in members]
        dn2 = float(dn2_choi_mp(Us, [m["weight"] for m in members],
                                _u3_mp(*u3, dps), dps=dps))
        out_pieces[k] = dict(
            pieces[k], t_budget=split[k], members=members, dn2=dn2,
            eps=float(r["eps"]),
            E_T=float(sum(float(m["weight"]) * m["t_count"] for m in members)),
            t_max=max(m["t_count"] for m in members), result=r)

    target = controlled_u3_matrix(theta, phi, lam, gamma)
    n_prod = 1
    for k in PIECES:
        n_prod *= len(out_pieces[k]["members"])
    return {
        "t_split": split,
        "dn2_upper": sum(out_pieces[k]["dn2"] for k in PIECES),
        "dn2_lower": _channel_dn2_lower(out_pieces, target),
        "eps_sum": sum(out_pieces[k]["eps"] for k in PIECES),
        "E_T": sum(out_pieces[k]["E_T"] for k in PIECES),
        "t_max": sum(out_pieces[k]["t_max"] for k in PIECES),
        "n_prod": n_prod,
        "pieces": out_pieces,
    }


def sample_controlled(result, rng=None, size=None):
    """
    Draw circuit(s) from :func:`optimize_controlled_u3`'s result: each piece
    sampled independently.  Run a fresh draw on every shot.

    Returns one circuit (list of operations), or a list of ``size`` circuits.
    """
    rng = np.random.default_rng() if rng is None else rng

    def one():
        return controlled_circuit({k: sample(result["pieces"][k]["members"], rng)
                                   for k in PIECES})

    if size is None:
        return one()
    return [one() for _ in range(size)]
