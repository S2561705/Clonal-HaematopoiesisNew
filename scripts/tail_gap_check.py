"""
tail_gap_check.py — where does the worst quadrature gap live?
"""
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
from src.KI_3 import (
    _get_arrays, compute_beta_quantiles, compute_deterministic_size,
    build_clone_h_grids, find_valid_clonal_structures,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"
RES = 4000
H_RESOLUTION, MAX_H = 3, 1.0

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, H_RESOLUTION, MAX_H)
h0 = __import__("jax.numpy", fromlist=["array"]).array([g[-1] for g in h_grids])
det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)

N_w_cond = (np.array(tot)[:, None] - np.array(det))[:, :, None]
half = (1.0 + np.array(h_mut))[None, :, None] / 2.0
beta_q = np.array(compute_beta_quantiles(AO, DP, observed, RES))
x = -N_w_cond * beta_q / (beta_q - half)

t, m = 1, 1   # worst case from your output
xs = np.sort(x[t, m])
gaps = np.diff(xs)
worst = np.argmax(gaps)
print(f"worst gap at sorted index {worst}/{RES}  "
      f"(q-position ~ {worst/RES:.5f})  "
      f"between x={xs[worst]:.3e} and x={xs[worst+1]:.3e}, gap={gaps[worst]:.3e}")
print(f"first 3 sorted x: {xs[:3]}")
print(f"last 3 sorted x: {xs[-3:]}")