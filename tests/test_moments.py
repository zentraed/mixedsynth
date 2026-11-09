"""
Tests for moment computation and the dn2_moment formula.

Pins k1=1/2, k2=1/4 from first principles:
  - k1: dn2 of rotation by α = sin(α/2) ≈ α/2 = |v|/2  →  k1 = 1/2
  - k2: incoherent mixed channel dn2 ≈ (1/4) Σ_i p_i |v_i|²  →  k2 = 1/4
"""


import numpy as np
import mpmath as mp
from mixedsynth.moments import compute_moments_mp, dn2_moment_mp, K1, K2

DPS = 40


def test_k1_single_rotation():
    """k1 = 1/2: for a single member v, dn2_moment = |v|/2."""
    # One member with rotation vector v = (0.01, 0, 0)  → |v| = 0.01
    vs_mp = [[mp.mpf("0.01"), mp.mpf(0), mp.mpf(0)]]
    p_f64 = np.array([1.0])
    with mp.workdps(DPS):
        mom = compute_moments_mp(vs_mp, p_f64, dps=DPS)
        dn2 = dn2_moment_mp(mom["M1_norm"], mom["floor2"])
    expected = mp.mpf("0.005")   # |v|/2 = 0.01/2
    diff = abs(float(dn2) - float(expected))
    assert diff < 1e-30, f"k1 test: dn2={float(dn2)}, expected={float(expected)}"
    print(f"PASS: k1_single_rotation  dn2={float(dn2):.6e}  expected={float(expected):.6e}")


def test_k2_incoherent():
    """k2 = 1/4: equal-weight mixture with M1=0 → dn2 = tr(C)/4."""
    # Two members with equal weights: v1 = (ε, 0, 0), v2 = (-ε, 0, 0)  → M1 = 0
    eps = mp.mpf("0.01")
    vs_mp = [[eps, mp.mpf(0), mp.mpf(0)], [-eps, mp.mpf(0), mp.mpf(0)]]
    p_f64 = np.array([0.5, 0.5])
    with mp.workdps(DPS):
        mom = compute_moments_mp(vs_mp, p_f64, dps=DPS)
        dn2 = dn2_moment_mp(mom["M1_norm"], mom["floor2"])
    # M1 = 0, S2 = 0.5*(eps² 0 0; 0 0 0; 0 0 0) + 0.5*(eps² 0 0; 0 0 0; 0 0 0) = diag(eps², 0, 0)
    # C = S2 - 0 = S2, tr(C) = eps²
    # dn2_moment = 0 + eps²/4
    expected = float(eps)**2 / 4
    diff = abs(float(dn2) - expected)
    assert diff < 1e-40, f"k2 test: dn2={float(dn2)}, expected={expected}"
    print(f"PASS: k2_incoherent  dn2={float(dn2):.6e}  expected={expected:.6e}")


def test_constants():
    """K1 and K2 have the correct analytic values."""
    assert K1 == 0.5, f"K1 should be 0.5, got {K1}"
    assert K2 == 0.25, f"K2 should be 0.25, got {K2}"
    print("PASS: constants  K1=0.5  K2=0.25")


def test_floor2_definition():
    """floor2 = K2 * tr(C) = tr(C)/4."""
    vs_mp = [[mp.mpf("0.02"), mp.mpf("0.01"), mp.mpf(0)],
             [mp.mpf("-0.02"), mp.mpf("0.01"), mp.mpf(0)]]
    p_f64 = np.array([0.5, 0.5])
    with mp.workdps(DPS):
        mom = compute_moments_mp(vs_mp, p_f64, dps=DPS)
    expected_floor2 = 0.25 * float(mom["tr_C"])
    diff = abs(float(mom["floor2"]) - expected_floor2)
    assert diff < 1e-40, f"floor2 mismatch: {float(mom['floor2'])} vs {expected_floor2}"
    print(f"PASS: floor2_definition  floor2={float(mom['floor2']):.6e}")


if __name__ == "__main__":
    test_constants()
    test_k1_single_rotation()
    test_k2_incoherent()
    test_floor2_definition()
    print("All moment tests passed.")
