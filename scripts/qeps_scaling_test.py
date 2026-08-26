"""
qeps_scaling_test.py
=====================
Quick test: does scaling q_eps down with resolution fix the non-convergence
seen in node_stability_check.py, or does it just delay it? Compares fixed
q_eps (baseline) against a resolution-scaled q_eps at the same resolutions.
"""
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
import jax.numpy as jnp
from src.KI_3 import (
    _get_arrays, compute_deterministic_size, build_clone_h_grids,
    find_valid_clonal_structures, jax_cs_hmm_ll_vec,
    compute_beta_quantiles_scaled, Q_EPS,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"
RESOLUTIONS = (250, 500, 1_000, 2_000)
S_RESOLUTION, H_RESOLUTION, MIN_S, MAX_S, MAX_H = 15, 3, 0.01, 3.0, 1.0
SCALE_POWER = 1.0   # q_eps = Q_EPS / resolution**SCALE_POWER

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)
cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, H_RESOLUTION, MAX_H)
h0 = jnp.array([g[-1] for g in h_grids])
det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)

print(f"structure: {cs}  |  resolutions={RESOLUTIONS}  |  scale_power={SCALE_POWER}\n")
print(f"{'resolution':>10}  {'q_eps used':>12}  {'max_ll':>14}  {'Δmax vs prev':>14}")
prev = None
results = {}
for res in RESOLUTIONS:
    q_eps = Q_EPS / (res ** SCALE_POWER)
    beta_q = compute_beta_quantiles_scaled(AO, DP, observed, res,
                                           q_eps_base=Q_EPS, scale_power=SCALE_POWER)
    out = np.array(jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, det, tot,
                                     h_mut, observed, carry_idx, beta_q))
    results[res] = out
    dmax = np.nan if prev is None else np.nanmax(np.abs(out - prev))
    print(f"{res:>10d}  {q_eps:>12.2e}  {np.nanmax(out):>14.6f}  "
          f"{'n/a' if prev is None else f'{dmax:.3e}':>14}")
    prev = out

print("\nCompare Δmax trend here against the FIXED-q_eps run from "
      "node_stability_check.py (2.29 -> 2.44 -> 2.63, growing).")
print("If Δmax here shrinks monotonically instead, resolution-scaling helps.")
print("If it still grows (just more slowly), the tail mechanism needs the")
print("bigger change-of-variables / Gauss-Jacobi fix, not a q_eps tweak.")