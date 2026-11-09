"""
Tests for rotation_vector and strip_phase.

1. Round-trip: exp(i (|v|/2) n·σ) reconstructs W.
2. Phase invariance: multiplying U by e^{iχ} leaves v unchanged.
3. Identity maps to zero vector.
"""


import mpmath as mp
from mixedsynth.generators import rotation_vector, strip_phase


DPS = 40


def _rz_mp(angle, dps=DPS):
    with mp.workdps(dps + 5):
        h = mp.mpf(angle) / 2
        return mp.matrix([[mp.exp(-mp.j * h), 0], [0, mp.exp(mp.j * h)]])


def _reconstruct_su2(v, dps=DPS):
    """Reconstruct W = exp(i (|v|/2) n·σ) from rotation vector v."""
    with mp.workdps(dps + 5):
        bx, by, bz = [mp.mpf(c) for c in v]
        b_norm = mp.sqrt(bx**2 + by**2 + bz**2)
        if b_norm < mp.mpf(10)**(-dps + 2):
            return mp.eye(2)
        alpha = b_norm
        c, s = mp.cos(alpha / 2), mp.sin(alpha / 2)
        # W = cos(α/2)I + i sin(α/2)(n·σ)
        nx, ny, nz = bx/b_norm, by/b_norm, bz/b_norm
        return mp.matrix([
            [c + mp.j * s * nz,      s * (mp.j * nx + ny)],
            [s * (mp.j * nx - ny),   c - mp.j * s * nz],
        ])


def test_round_trip():
    """exp(i|v|/2 n·σ) reconstructs W = strip_phase(V†U)."""
    with mp.workdps(DPS + 5):
        V = _rz_mp(1.234)
        U = _rz_mp(1.234 + 0.01)   # small perturbation → near-identity residual
        v = rotation_vector(U, V)
        W_ref = strip_phase(V.H * U)
        W_rec = _reconstruct_su2(v)
        for i in range(2):
            for j in range(2):
                diff = abs(W_rec[i,j] - W_ref[i,j])
                assert float(diff) < 1e-30, f"Round-trip failed at [{i},{j}]: diff={float(diff)}"
    print("PASS: round-trip")


def test_phase_invariance():
    """Multiplying U by e^{iχ} leaves v unchanged."""
    with mp.workdps(DPS + 5):
        V = _rz_mp(0.7)
        U = _rz_mp(0.71)
        v0 = rotation_vector(U, V)
        chi = mp.pi / 7
        U_phased = U * mp.exp(mp.j * chi)
        v1 = rotation_vector(U_phased, V)
        for k in range(3):
            diff = abs(float(v0[k]) - float(v1[k]))
            assert diff < 1e-30, f"Phase non-invariance at k={k}: diff={diff}"
    print("PASS: phase_invariance")


def test_identity():
    """U = V → v = 0."""
    with mp.workdps(DPS + 5):
        V = _rz_mp(1.5)
        v = rotation_vector(V, V)
        norm = sum(float(c)**2 for c in v)**0.5
        assert norm < 1e-30, f"Identity should give zero vector, got norm={norm}"
    print("PASS: identity")


def test_known_rotation():
    """Rz(π/2) relative to I → v = (0, 0, −π/2) in the W = a₀I + i(b·σ) convention."""
    with mp.workdps(DPS + 5):
        V = mp.eye(2)
        U = _rz_mp(mp.pi / 2)
        v = rotation_vector(U, V)
        # W = strip_phase(V†U) = strip_phase(Rz(π/2))
        # Rz(π/2) = diag(e^{-iπ/4}, e^{iπ/4}); W[0,0] = a0 + i·bz → a0=cos(π/4), bz=−sin(π/4)
        # alpha = 2*atan2(|b|, a0) = π/2,  v = alpha·b/|b| = (0, 0, −π/2)
        # (Rz(θ) = exp(−iθσz/2) is a rotation by θ about +z; under the +i(b·σ)
        #  parametrisation that is b ∝ −z.  Only consistency matters downstream.)
        assert abs(float(v[0])) < 1e-20, f"vx should be 0, got {float(v[0])}"
        assert abs(float(v[1])) < 1e-20, f"vy should be 0, got {float(v[1])}"
        assert abs(float(v[2]) + float(mp.pi/2)) < 1e-20, f"vz should be −π/2, got {float(v[2])}"
    print("PASS: known_rotation")


if __name__ == "__main__":
    test_round_trip()
    test_phase_invariance()
    test_identity()
    test_known_rotation()
    print("All generator tests passed.")
