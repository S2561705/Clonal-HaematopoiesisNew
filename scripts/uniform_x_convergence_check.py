"""
uniform_x_convergence_check.py
Rerun interior_h_convergence_check's exact test, swapping in the new
uniform-in-x node scheme, to confirm Δmax now shrinks with resolution.
"""
import sys
sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import pickle as pk
import numpy as np
import jax.numpy as jnp
from src.KI_3 import (
    _get_arrays, compute_deterministic_size, build_clone_h_grids,
    find_valid_clonal_structures, jax_cs_hmm_ll_vec, compute_beta_bounds,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"
RESOLUTIONS = (250, 500, 1_000, 2_000)
S_RESOLUTION, H_RESOLUTION, MIN_S, MAX_S, MAX_H = 15, 3, 0.01, 3.0, 1.0

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)
cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, H_RESOLUTION, MAX_H)
beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

for label, idx in [("interior (mid)", len(h_grids[0]) // 2), ("h_max (pole)", -1)]:
    h0 = jnp.array([np.array(g)[idx] for g in h_grids])
    det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)
    print(f"=== {label}  h={np.array(h0)} ===")
    prev = None
    for res in RESOLUTIONS:
        out = np.array(jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, det, tot,
                                         h_mut, observed, carry_idx,
                                         beta_lo, beta_hi, resolution=res))
        dmax = np.nan if prev is None else np.nanmax(np.abs(out - prev))
        print(f"  resolution={res:>5d}  max_ll={np.nanmax(out):+.6f}  "
              f"{'n/a' if prev is None else f'Δmax={dmax:.3e}'}")
        prev = out
    print()