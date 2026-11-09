# mixedsynth

**Mixed-channel Clifford+T synthesis for single-qubit gates.**

Standard synthesis (gridsynth and similar tools) gives you *one* Clifford+T sequence that approximates a target gate to error ε. mixedsynth gives you a **probability distribution over a few sequences**, chosen so that their errors cancel on average. If you run a freshly sampled sequence on each shot, the resulting channel's diamond-norm error is roughly **ε²** instead of ε, with no increase in the per-shot T budget.

mixedsynth has two parts:

1. **Sequence generation.** It produces ensembles of exact Clifford+T gate strings approximating Rz(θ) across a range of T-counts, from T = 0 (Cliffords) upward.
2. **Channel synthesis.** It combines those ensembles into an optimal mixture of at most 4 sequences for an arbitrary U3(θ, φ, λ), under a hard per-shot T budget.

```
U3(1.1, 2.3, 0.4) with T ≤ 30 per shot
  best single sequence : dn/2 = 1.925e-02
  mixed channel        : dn/2 = 5.937e-04  (gain 32.4x, E[T] = 28.1)
  members:
    p=0.4023  T=29  SSTHSHTHTHTHTHTSHTHTSHTSHTHTHTHTSHTSHTHSHTHTSHTHTSHTSHTSHTHTSHTHTSHTSHTSHTHTSHZ
    p=0.3063  T=30  HTHTSHTHTSHTHTSdHTSHTHTSHTSHTHTHTHTHTSHTHTSHTHTSHTHTHTSHTSHTSdHTHTSHTSHTSHTHTH
    p=0.2233  T=24  SHTSHTSHTSHTHTHTZHHTSHTHTSHTSHTSHTSHTSHTSdHHTHTHTSHTHTSHTSHTSHTHTSHTXS
    p=0.0681  T=28  HTHTSHTHTSHTHTSdHTSHTHTHTHTHTSHTHTHTSHTHTSHTSHTSHTHTHTYHTHTSHTSHTSHTHTH
```

---

## Install

mixedsynth needs `gmpy2 >= 2.1.2` (older versions raise `TypeError` in the grid-point enumeration). Using conda:

```bash
conda create -n mixedsynth python=3.10 gmpy2 numpy scipy mpmath
conda activate mixedsynth
pip install -e ".[test]"
pytest
```

The exact ring arithmetic over Z[ω] and the grid-problem solver come from [pyLIQTR](https://github.com/isi-usc-edu/pyLIQTR) and are vendored under `mixedsynth._pyliqtr` with their license. Only the import paths were changed, so you don't need to install pyLIQTR.

## Quickstart

```python
import random
from mixedsynth import harvest_rz, optimize_u3_hardcap, sample, circuit_order

random.seed(0)   # the norm-equation solver is randomized; seed for reproducibility

# 1. Sequence generation: Clifford+T approximations of Rz(0.7), T = 0..20
for t, seq, U in harvest_rz(0.7, t_min=0, t_budget=20, n_keep=1, stride=4,
                            with_sequences=True):
    print(t, seq)

# 2. Channel synthesis: optimal mixture for U3 with T1+T2+T3 ≤ 30 on every shot
r = optimize_u3_hardcap(1.1, 2.3, 0.4, t_total=30)
r["eps"]       # dn/2 of the best single sequence
r["dn2"]       # dn/2 of the mixed channel
r["members"]   # [{"weight", "t_count", "sequence"}, ...]  (≤ 4 entries)

# 3. Execution: draw a fresh sequence for each shot
shot = sample(r["members"])
circuit_order(shot)   # ['H', 'T', 'H', 'S', ...] in application order
```

You can run the full example with `python examples/quickstart.py`.

---

## Sequence generation

`harvest_rz(angle, t_min, t_budget, n_keep, stride, with_sequences=True)` returns `(t_count, sequence, U)` triples sorted by T-count. Each T-level contributes several *distinct* approximations, which give the optimizer the diversity it needs.

| T-count | Source |
|---|---|
| T = 0 | the nearest single-qubit Cliffords (all 24, found by BFS over ⟨H, S⟩) |
| T = 1 | the nearest C·T circuits |
| T ≥ 2 | a gridsynth-style search: ε is swept as ε(T) = 2^(−T/3); for each ε, up to `n_keep` lattice candidates are taken from the ellipse/grid enumeration, the norm equation is solved, and each solution is decomposed exactly into a gate string |

`genlist.approximate_rz_direct_set(num, den, precision, n_cands)` returns the same kind of ensemble at a single fixed precision for Rz(num·π/den).

**Sequence format.** Sequences are strings in *matrix order*, so `"HTS"` is the matrix H·T·S (S is applied first). The tokens are `H S Sd X Y Z T`. The `mixedsynth.sequences` helpers work with this format:

| Function | Purpose |
|---|---|
| `tokenize(seq)` | split a string into gate tokens |
| `circuit_order(seq)` | gate tokens in application order |
| `t_count(seq)` | number of T gates |
| `sequence_to_matrix(seq, dps=None)` | the unitary, as numpy, or as mpmath when `dps` is given |
| `sample(members, rng, size)` | draw sequences from a synthesized channel |

The test suite checks that every returned sequence reproduces its unitary up to global phase and has the stated T-count.

## Channel synthesis

**Decomposition.** U3(θ, φ, λ) = Rz(φ + 5π/2) · H · Rz(θ) · H · Rz(λ − π/2). Each Rz is harvested separately. Composite U3 candidates are then built under the hard constraint **T₁ + T₂ + T₃ ≤ T_total**. One channel may fall back to a Clifford (T = 0) so the others can use the saved budget.

**Generator picture.** Each candidate Uᵢ is mapped to its error rotation vector vᵢ ∈ ℝ³ ≅ su(2), the log of V†Uᵢ. For a mixture with weights p:

- the coherent error is the first moment M₁ = Σ pᵢ vᵢ;
- the incoherent floor is set by the second moment C = Σ pᵢ vᵢvᵢᵀ;
- the diamond-norm proxy is dn/2 ≈ |M₁|/2 + tr(C)/4.

To leading order, a single sequence's error is linear in |v|. A mixture with M₁ = 0 leaves only the quadratic term, which is why the mixture's error is roughly ε².

**Optimization.**

1. **Frank–Wolfe min-norm point** on conv({vᵢ}) determines whether the origin lies inside the hull, that is, whether the coherent error can be cancelled exactly.
2. **Linear program on the M₁ = 0 face** minimizes tr(C). By Carathéodory's theorem in ℝ³, an optimal vertex uses **at most 4 sequences**, whatever the pool size.
3. **mpmath refinement.** The weights on the ≤ 4 support members are re-solved at 60 digits, so M₁ cancels to working precision (|M₁| ≈ 1e-67 at T = 150) rather than to float64 precision.
4. **Dominance guard.** The result is never worse than the best single sequence.

The dn/2 proxy is checked against the exact Choi-matrix diamond norm (`diamond.py`). In the benchmark below the ratio is 1.000 for T_total ≥ 20, 0.997 at T_total = 10 and 0.99 at T_total = 5. The proxy is a leading-order expansion, so it is loose only when errors are large (ratio ≈ 0.75 at T_total = 1).

`optimize_u3(…, t_total=…)` is an alternative mode that splits the budget equally (T_total/3 per Rz) and optimizes each Rz channel independently.

---

## Results

### Mixing gain under a hard T budget

10 random U3 gates, seed 42, dps 60. Full table: [`benchmarks/results/benchmark_mixedsynth_results.txt`](benchmarks/results/benchmark_mixedsynth_results.txt)

| T_total | best single dn/2 | mixed dn/2 | median gain | E[T] | n_nz | opt time |
|---:|---:|---:|---:|---:|:-:|---:|
| 10 | 5.0e-2 | 5.8e-3 | 8× | 7.7 | 4 | 0.16 s |
| 25 | 1.3e-2 | 4.9e-4 | 27× | 21.0 | 4 | 0.6 s |
| 50 | 2.8e-3 | 1.6e-5 | 184× | 46.5 | 4 | 3.2 s |
| 100 | 1.2e-4 | 3.4e-8 | 3.5 × 10³ | 97.8 | 4 | 5.0 s |
| 150 | 3.4e-6 | 3.0e-11 | 1.2 × 10⁵ | 146.1 | 4 | 5.8 s |
| 200 | 8.7e-8 | 1.3e-14 | 6.3 × 10⁶ | 195.0 | 4 | 8.9 s |

Values are medians over the 10 gates. In every case with T_total ≥ 5, the mixture sits at the incoherent floor (|M₁| ≈ 0).

### Against Trasyn at equal T

For the comparison, Trasyn (Hao et al.) was run at each target precision and its T-count was recorded. mixedsynth was then given the **same** budget as a hard per-shot cap. Full table: [`benchmarks/results/benchmark_vs_trasyn_results.txt`](benchmarks/results/benchmark_vs_trasyn_results.txt)

| Trasyn ε target | median Trasyn T | mixedsynth dn/2 ÷ Trasyn dn/2 (median) | gates where mixedsynth wins |
|---:|---:|---:|:-:|
| 0.05 | 8 | 0.28 | 10 / 10 |
| 0.02 | 10 | 0.39 | 10 / 10 |
| 0.01 | 14 | 0.24 | 10 / 10 |
| 0.005 | 18 | 0.40 | 10 / 10 |
| 0.002 | 22.5 | 0.62 | 10 / 10 |
| 0.001 | 26 | 0.50 | 10 / 10 |

The largest T-count in each mixture never exceeds Trasyn's T, and the expected T-count is lower in 59 of the 60 cases. Trasyn outputs are cached in `benchmarks/trasyn_cache.json`, so you only need Trasyn installed to regenerate that cache.

### Reproducing

```bash
python benchmarks/benchmark.py              # hard-cap gain table, plus a 10-seed robustness sweep
python benchmarks/benchmark_vs_trasyn.py    # vs Trasyn, reads the cache (~1 min)
python benchmarks/harvest_trasyn.py         # optional: rebuild the cache (needs trasyn)
```

The norm-equation solver uses Python's `random`. Seed it (`random.seed(...)`) for bit-identical runs; unseeded runs differ slightly in which candidates are found.

---

## Layout

```
src/mixedsynth/
  harvest.py      Rz sequence harvesting (T=0, T=1, gridsynth sweep)
  genlist.py      ensemble generation over the ring D[ω]; fixed-ε sets; Choi metrics
  sequences.py    gate-string helpers: tokenize / to-matrix / sample
  u3.py           U3 channel synthesis (hard-cap and equal-split modes)
  generators.py   rotation vectors v ∈ ℝ³ from SU(2) residuals
  optimize.py     Frank–Wolfe min-norm point + LP on the M₁=0 face
  moments.py      high-precision first/second moments and the dn/2 proxy
  diamond.py      Choi-based diamond-norm certificate
  ring_utils.py   parse Trasyn/gridsynth-style strings into exact D[ω] matrices
  _pyliqtr/       vendored pyLIQTR ring arithmetic and grid solver
tests/            unit and end-to-end tests
benchmarks/       benchmark scripts, Trasyn cache, results
examples/         quickstart.py
```

## References

- U3 decomposition: Eq. 1 of [arXiv:2503.15843](https://arxiv.org/abs/2503.15843)
- Ross & Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations* (gridsynth)
- Campbell, *Shorter gate sequences for quantum computing by mixing unitaries*, PRA 95, 042306 (2017)
- Hastings, *Turning gate synthesis errors into incoherent errors* (2016)
