"""
node_spacing_check.py
======================
Inspects the ACTUAL x-nodes and valid/anchored split produced by
compute_global_variables for one participant, at two beta_resolutions.
Tests whether valid-node count and spacing near the pole (half) are
resolution-sensitive in a way that would explain non-convergent behaviour
in node_stability_check / qeps_sensitivity_check.
"""
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
import jax.numpy as jnp
from src.KI_3 import (
    _get_arrays, compute_beta_quantiles, compute_deterministic_size,
    build_clone_h_grids, find_valid_clonal_structures,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"
RESOLUTIONS = (500, 4000)
H_RESOLUTION = 3
MAX_H = 1.0

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, H_RESOLUTION, MAX_H)
h0 = jnp.array([g[-1] for g in h_grids])
det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)

N_w_cond = (np.array(tot)[:, None] - np.array(det))[:, :, None]
half = (1.0 + np.array(h_mut))[None, :, None] / 2.0

for res in RESOLUTIONS:
    beta_q = np.array(compute_beta_quantiles(AO, DP, observed, res))
    x = -N_w_cond * beta_q / (beta_q - half)
    valid = x > 0
    print(f"\n=== beta_resolution={res} ===")
    for t in range(AO.shape[0]):
        for m in range(AO.shape[1]):
            if not observed[t, m]:
                continue
            v = valid[t, m]
            n_valid = v.sum()
            xv = x[t, m][v]
            spacing = np.diff(np.sort(xv)) if n_valid > 1 else np.array([np.nan])
            print(f"tp={t} mut={m}  half={half[0,m,0]:.4f}  "
                  f"n_valid={n_valid}/{res}  "
                  f"valid_x_range=[{xv.min() if n_valid else float('nan'):.3e}, "
                  f"{xv.max() if n_valid else float('nan'):.3e}]  "
                  f"median_spacing={np.median(spacing):.3e}  "
                  f"max_spacing={np.nanmax(spacing) if n_valid>1 else float('nan'):.3e}")