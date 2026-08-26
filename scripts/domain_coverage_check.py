"""
domain_coverage_check.py
=========================
Checks whether each timepoint's [lo, hi] domain (from beta_lo/beta_hi)
actually covers where the BD transition kernel from the PREVIOUS
timepoint places its mass -- separate question from node spacing.
"""
import sys
sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import pickle as pk
import numpy as np
import jax.numpy as jnp
from src.KI_3 import (
    _get_arrays, compute_deterministic_size, build_clone_h_grids,
    find_valid_clonal_structures, compute_beta_bounds, BD_process_dynamics,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"
H_RESOLUTION, MAX_H = 3, 1.0
S_TEST = 0.087   # MAP-s from earlier runs

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, H_RESOLUTION, MAX_H)
h0 = jnp.array([np.array(g)[len(g)//2] for g in h_grids])   # interior h, same as your test
det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)
beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

N_w_cond = (np.array(tot)[:, None] - np.array(det))[:, :, None]
half = (1.0 + np.array(h_mut))[None, :, None] / 2.0
beta_lo_ = np.array(beta_lo)[:, :, None]
beta_hi_ = np.minimum(np.array(beta_hi)[:, :, None], half * (1 - 1e-9))
x_lo = -N_w_cond * beta_lo_ / (beta_lo_ - half)
x_hi = -N_w_cond * beta_hi_ / (beta_hi_ - half)

delta_t = np.diff(np.array(time_points))
for j in range(1, AO.shape[0]):
    for m in range(AO.shape[1]):
        prev_lo, prev_hi = x_lo[j-1, m, 0], x_hi[j-1, m, 0]
        cur_lo, cur_hi = x_lo[j, m, 0], x_hi[j, m, 0]
        exp_term = np.exp(delta_t[j-1] * S_TEST)
        predicted_mean_lo = prev_lo * exp_term
        predicted_mean_hi = prev_hi * exp_term
        covered = (predicted_mean_lo >= cur_lo) and (predicted_mean_hi <= cur_hi)
        print(f"tp {j-1}->{j} mut={m}: prev=[{prev_lo:.3e},{prev_hi:.3e}] "
              f"predicted_mean(after growth)=[{predicted_mean_lo:.3e},{predicted_mean_hi:.3e}] "
              f"cur_domain=[{cur_lo:.3e},{cur_hi:.3e}]  covered={covered}")