"""
Regression tests: the Rz harvest must not depend on the environment.

harvest_u3 once differed between environments (Python 3.10 / numpy 1.26 /
x86_64 vs Python 3.12 / numpy 2 / arm64) for the same `random` seed.  The
T=0/T=1 candidates were scored in float64, and distinct gates with exactly
equal fidelity were tie-broken by last-ulp rounding noise.  These tests pin
the harvested sequences (not just their counts) for a fixed seed, check that
the T=0/T=1 matrices are exact, and check that the result does not depend on
PYTHONHASHSEED.
"""

import hashlib
import os
import random
import subprocess
import sys

import mpmath as mp

from mixedsynth import harvest_u3
from mixedsynth.sequences import sequence_to_matrix, t_count

SEED = 1234
TARGET = (1.1, 2.3, 0.4)
T_BUDGET = 4

# harvest_u3(1.1, 2.3, 0.4, 4) after random.seed(1234), as (t_count, seq).
# Ties in fidelity: channel 3 T=1 has four distinct gates at fidelity
# 0.4996...; HT, SSHSHT, HSHSST are the first three in Clifford order.
PINNED = [
    [(0, 'SSS'), (0, ''), (0, 'SH'), (0, 'HS'), (0, 'HSSHSH'),
     (1, 'SSST'), (1, 'SST'), (1, 'SHT'), (1, 'HST'), (1, 'HSSHSHT'),
     (2, 'SHTSHTYS'),
     (4, 'HTHTHTSHTX'), (4, 'TSHTSHTSHTSHSd'), (4, 'HTSHTHTHTX'),
     (6, 'TSHTHTSHTHTHTS'), (6, 'TSHTSHTHTHTHTSHSd'), (6, 'HTSHTHTSHTHTHTSd'),
     (8, 'TSHTHTHTHTSHTSHTHT'), (8, 'TSHTHTSHTHTSHTSHTSHTSd')],
    [(0, 'S'), (0, ''), (0, 'SHSH'), (0, 'HSHS'), (0, 'SSHS'),
     (1, 'T'), (1, 'ST'), (1, 'HSHT'), (1, 'SSHT'), (1, 'SHST'),
     (2, 'TSHTSdHSd'),
     (4, 'THTSHTHTZH'), (4, 'SHTSHTHTHTSdH'), (4, 'SHTSHTHTHTXS'), (4, 'THTHTHTSdH'),
     (6, 'SHTSHTSHTHTSHTHTZHS'), (6, 'SHTSHTSHTSHTHTHT'), (6, 'THTHTSHTHTHTHZ'),
     (8, 'THTSHTHTHTHTHTSHTXS')],
    [(0, 'SS'), (0, 'SSS'), (0, 'H'), (0, 'SSHSH'), (0, 'HSHSS'),
     (1, 'SST'), (1, 'ST'), (1, 'HT'), (1, 'SSHSHT'), (1, 'HSHSST'),
     (2, 'TSHTSdHS'),
     (4, 'THTSHTHTZHZ'), (4, 'HTHTHTSHTY'), (4, 'SHTSHTHTHTYS'), (4, 'THTHTHTSdHZ'),
     (6, 'SHTSHTSHTHTSHTHTZHSd')],
]


def _harvest():
    random.seed(SEED)
    return harvest_u3(*TARGET, T_BUDGET)


def test_harvest_sequences_pinned():
    got = [[(t, s) for t, s, _U in ch] for ch in _harvest()]
    assert got == PINNED


def test_harvest_matrices_match_sequences_exactly():
    """Every candidate's matrix equals its sequence's exact matrix (up to
    global phase) to far beyond float64 precision, including T=0/T=1."""
    for ch in _harvest():
        for t, seq, U in ch:
            assert t_count(seq) == t
            with mp.workdps(70):
                W = sequence_to_matrix(seq, dps=60).H * U
                assert 2 - abs(W[0, 0] + W[1, 1]) < mp.mpf(10) ** -50, seq


def test_low_t_candidates_are_distinct_gates():
    """T=0/T=1 slots hold distinct gates, not global-phase variants."""
    for ch in _harvest():
        for level in (0, 1):
            mats = [U for t, _s, U in ch if t == level]
            for i in range(len(mats)):
                for j in range(i):
                    with mp.workdps(70):
                        W = mats[i].H * mats[j]
                        assert abs(W[0, 0] + W[1, 1]) < 2 - mp.mpf(10) ** -20


_DIGEST_SCRIPT = """
import hashlib, random, warnings
warnings.simplefilter("ignore")
from mixedsynth import harvest_u3
random.seed({seed})
h = harvest_u3({theta!r}, {phi!r}, {lam!r}, {t})
print(hashlib.sha256(repr([[(t, s, [str(x) for x in U]) for t, s, U in c]
                           for c in h]).encode()).hexdigest())
"""


def test_harvest_independent_of_hash_seed():
    script = _DIGEST_SCRIPT.format(seed=SEED, theta=TARGET[0], phi=TARGET[1],
                                   lam=TARGET[2], t=T_BUDGET)
    digests = []
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, "-c", script], env=env,
                             capture_output=True, text=True, check=True)
        digests.append(out.stdout.strip())
    assert len(set(digests)) == 1, digests
