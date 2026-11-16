"""
mixedsynth — Mixed-channel Clifford+T synthesis via generator-picture optimization.

Two layers:

Sequence generation
  - harvest_rz: Clifford+T gate sequences approximating Rz(θ) across a sweep of
    T-counts (T=0 Cliffords, T=1 C·T circuits, T≥2 from a gridsynth-style
    ensemble search), with exact gate strings.
  - genlist.approximate_rz_direct_set: an ensemble of distinct approximations
    at a fixed precision.

Channel synthesis
  - optimize_u3_hardcap: a probability distribution over ≤ 4 Clifford+T
    sequences for U3(θ,φ,λ) with T_1+T_2+T_3 ≤ T_total on every shot.
  - Stage 1: Frank-Wolfe min-norm-point on conv({v_i}) cancels coherent
    (first-moment) error; Stage 2: an LP on the M1=0 face minimizes the
    incoherent (second-moment) floor.  Carathéodory (d=3) gives n_nz ≤ 4.
  - dn2_moment = |M1|/2 + tr(C)/4 is the in-loop metric; the Choi-based
    diamond norm is computed once as a certificate.

Single-sequence synthesis
  - synthesize_u3_single: the best single Rz3·H·Rz2·H·Rz1 sequence for
    U3(θ,φ,λ) with T_1+T_2+T_3 ≤ T_total, by exhaustive search over the same
    harvested Rz candidates optimize_u3_hardcap uses (no mixed channel).
  - enumerate_u3_splits: per split t_1+t_2+t_3 == T_total, the best (or
    every) such sequence; splits with no harvested candidate are reported.

Controlled gates
  - C-V for V = e^{iγ}U3(θ,φ,λ) as P(α) on the control and C·CX·B·CX·A on
    the target (ABC = I), so every piece acts unconditionally and its global
    phase stays global.  synthesize_controlled_u3_single gives one circuit;
    optimize_controlled_u3 makes each piece an optimize_u3_hardcap mixture
    under a hard per-shot T cap, sampled with sample_controlled.

Sequences
  - hst: one H/S/T/W alphabet for every tool (from_gridsynth, from_pyliqtr,
    from_trasyn), exact matrices, the Matsumoto–Amano normal form in
    gridsynth's layout with exact phase, and phase alignment to a target.

Exact ring arithmetic (Z[ω], D[ω]) and the grid-problem solver are vendored
from pyLIQTR under ``mixedsynth._pyliqtr`` (see its LICENSE).
"""

__version__ = "0.1.0"

from .generators import rotation_vector
from .optimize import frank_wolfe_f64, stage2_lp, optimize_ms
from .moments import compute_moments_mp, dn2_moment_mp
from .harvest import harvest_rz, rz_angles_for_u3, rz_exact_mp
from .u3 import optimize_u3, optimize_u3_hardcap
from .u3_single import synthesize_u3_single, enumerate_u3_splits, harvest_u3
from .controlled import (
    u3_params, controlled_u3_matrix, controlled_u3_pieces, split_budget,
    circuit_matrix, circuit_t_count, synthesize_controlled_u3_single,
    optimize_controlled_u3, sample_controlled,
)
from .genlist import approximate_rz_direct_set
from . import hst
from .sequences import (
    tokenize, t_count, circuit_order, sequence_to_matrix, sample,
)
