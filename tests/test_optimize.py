"""
Tests for Frank-Wolfe min-norm-point and Stage 2 LP.

1. Origin enclosed: FW converges to hull_dist ≈ 0, Stage 2 LP finds M1=0 solution.
2. Origin exterior: FW finds closest point, hull_dist > 0.
3. n_nz ≤ 4 (Carathéodory bound in R³).
4. Consistency check vs scipy minimize (brute-force QP).
"""


import numpy as np
from scipy.optimize import minimize
from mixedsynth.optimize import frank_wolfe_f64, stage2_lp, optimize_ms


def _brute_force_min_norm(vs_f64):
    """Brute-force min |Bp|² on simplex using scipy."""
    N = len(vs_f64)
    def obj(p): return float(np.dot(vs_f64.T @ p, vs_f64.T @ p))
    def grad(p): return 2 * vs_f64 @ (vs_f64.T @ p)
    cons = [{"type": "eq", "fun": lambda p: p.sum() - 1.0}]
    bnds = [(0, None)] * N
    p0 = np.ones(N) / N
    res = minimize(obj, p0, jac=grad, method="SLSQP",
                   constraints=cons, bounds=bnds,
                   options={"ftol": 1e-20, "maxiter": 5000})
    M1 = vs_f64.T @ res.x
    return res.x, float(np.sqrt(float(np.dot(M1, M1)))), M1


def test_enclosed():
    """Origin enclosed: hull_dist ≈ 0 after FW + LP."""
    rng = np.random.default_rng(0)
    # Build 8 vectors that enclose the origin
    vs = np.vstack([rng.standard_normal((4, 3)), -rng.standard_normal((4, 3))])
    vs *= 0.01  # scale to ε ~ 0.01
    p, hd, M1 = frank_wolfe_f64(vs, n_iter=5000)
    assert hd < 1e-10, f"Expected enclosed, got hull_dist={hd}"
    p2 = stage2_lp(vs)
    assert p2 is not None, "Stage 2 LP should be feasible when origin is enclosed"
    M1_lp = vs.T @ p2
    assert np.linalg.norm(M1_lp) < 1e-8, f"LP M1 not near zero: {np.linalg.norm(M1_lp)}"
    print(f"PASS: enclosed  hull_dist={hd:.2e}  n_nz={int(np.sum(p2>1e-4))}")


def test_exterior():
    """Origin exterior: FW gives closest hull point; hull_dist > 0."""
    vs = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 0.5, 0.0]])
    # Closest point in conv(vs) to origin is (1,0,0)
    p, hd, M1 = frank_wolfe_f64(vs, n_iter=2000)
    assert abs(hd - 1.0) < 1e-6, f"hull_dist should be ~1.0, got {hd}"
    print(f"PASS: exterior  hull_dist={hd:.6f}  closest_point={M1}")


def test_caratheodory_bound():
    """n_nz ≤ 4 always (Carathéodory in R³)."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        N = rng.integers(10, 50)
        vs = rng.standard_normal((N, 3)) * 0.01
        opt = optimize_ms(vs, enclosed_tol_ratio=0.1)
        n_nz = int(np.sum(opt["p"] > 1e-4))
        assert n_nz <= 4, f"Carathéodory violated: n_nz={n_nz} for N={N}"
    print("PASS: caratheodory_bound  (n_nz ≤ 4 for 10 random instances)")


def test_fw_vs_brute_force():
    """FW min-norm matches scipy SLSQP on random point sets."""
    rng = np.random.default_rng(7)
    for trial in range(5):
        N = rng.integers(5, 20)
        vs = rng.standard_normal((N, 3)) * 0.005
        p_fw, hd_fw, _ = frank_wolfe_f64(vs, n_iter=5000)
        _, hd_bf, _ = _brute_force_min_norm(vs)
        assert abs(hd_fw - hd_bf) < 1e-6, \
            f"Trial {trial}: FW={hd_fw:.6e} vs BF={hd_bf:.6e}"
    print("PASS: fw_vs_brute_force")


if __name__ == "__main__":
    test_enclosed()
    test_exterior()
    test_caratheodory_bound()
    test_fw_vs_brute_force()
    print("All optimize tests passed.")
