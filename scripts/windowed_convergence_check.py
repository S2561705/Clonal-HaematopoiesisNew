"""
node_stability_check's Δmax uses raw nanmax(|diff|) over the WHOLE array,
including implausible high-s entries known to swing wildly without
affecting actual probability mass (same issue flagged in
check_qeps_stability). This restricts the comparison to the plausible
region (within log_lik_window of each column's own max), same convention
as check_qeps_stability, to see whether Fix 4's remaining "Δmax=2.78"
is real or an artifact of comparing the wrong region.
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
RESOLUTIONS = (250, 500, 1000, 2000)
S_RESOLUTION, H_RESOLUTION, MIN_S, MAX_S, MAX_H = 15, 3, 0.01, 3.0, 1.0
LOG_LIK_WINDOW = 20.0

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
        in_window = out >= (np.nanmax(out, axis=0, keepdims=True) - LOG_LIK_WINDOW)
        if prev is None:
            dmax_all, dmax_win = np.nan, np.nan
        else:
            dmax_all = np.nanmax(np.abs(out - prev))
            mask = in_window & prev_window
            dmax_win = np.nanmax(np.abs(out[mask] - prev[mask])) if mask.any() else np.nan
        print(f"  resolution={res:>5d}  max_ll={np.nanmax(out):+.6f}  "
              f"Δmax(all)={'n/a' if prev is None else f'{dmax_all:.3e}'}  "
              f"Δmax(in-window)={'n/a' if prev is None else f'{dmax_win:.3e}'}")
        prev, prev_window = out, in_window
    print()