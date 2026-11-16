"""
optimize_u3_hardcap's first-pass composite selection (u3._preselect_pool)
against the original scalar triple loop, including exact fidelity ties
(repeated unitaries, as Clifford pieces produce).
"""

import heapq

import numpy as np
import pytest

from mixedsynth.u3 import _preselect_pool

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def reference_pool(m_np, H_np, tH, t_total, n_preselect):
    """The scalar loop optimize_u3_hardcap used before vectorization."""
    pool = {}
    ctr = 0
    for i1, (t1, U1_np) in enumerate(m_np[0]):
        HU1 = H_np @ U1_np
        for i2, (t2, U2_np) in enumerate(m_np[1]):
            if t1 + t2 > t_total:
                break
            U21 = U2_np @ HU1
            for i3, (t3, U3_np) in enumerate(m_np[2]):
                t_sum = t1 + t2 + t3
                if t_sum > t_total:
                    break
                U_comp = U3_np @ H_np @ U21
                fe = abs(np.trace(tH @ U_comp)) ** 2 / 4
                entry = (fe, ctr, i1, i2, i3)
                ctr += 1
                h = pool.setdefault(t_sum, [])
                if len(h) < n_preselect:
                    heapq.heappush(h, entry)
                elif fe > h[0][0]:
                    heapq.heapreplace(h, entry)
    return pool


def random_su2(rng):
    a = rng.normal(size=4)
    a /= np.linalg.norm(a)
    return np.array([[a[0] + 1j * a[3], a[2] + 1j * a[1]],
                     [-a[2] + 1j * a[1], a[0] - 1j * a[3]]])


def channel(rng, t_max, per_level, n_dup):
    """Candidates sorted by T; n_dup distinct unitaries reused to force ties."""
    base = [random_su2(rng) for _ in range(n_dup)]
    out = []
    for t in range(t_max + 1):
        for _ in range(per_level):
            U = base[rng.integers(n_dup)] if rng.random() < 0.5 else random_su2(rng)
            out.append((t, U))
    return out


def check_same(new, ref):
    """Identical to the scalar loop: entries, fidelities and heap layout."""
    assert new == ref


@pytest.mark.parametrize("seed", range(12))
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    t_max = int(rng.integers(0, 9))
    chans = [channel(rng, t_max, int(rng.integers(1, 5)), int(rng.integers(1, 4)))
             for _ in range(3)]
    tH = random_su2(rng).conj().T
    for t_total in sorted({0, t_max, 2 * t_max, 3 * t_max, int(rng.integers(0, 3 * t_max + 1))}):
        for n_pre in (1, 3, 30):
            check_same(_preselect_pool(chans, H, tH, t_total, n_pre),
                       reference_pool(chans, H, tH, t_total, n_pre))


def test_empty_channel():
    rng = np.random.default_rng(0)
    ch = channel(rng, 2, 2, 1)
    assert _preselect_pool([ch, [], ch], H, np.eye(2), 6, 5) == {}
