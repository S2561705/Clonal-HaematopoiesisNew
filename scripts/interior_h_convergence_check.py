"""
interior_h_convergence_check.py
=================================
node_stability_check.py and qeps_scaling_test.py have only been tested at
h0 = max_h = 1.0 -- the single worst point in the h domain (half=1.0 sits
exactly at the pole). This checks whether the same non-convergence appears
at INTERIOR h values, to separate:
  (a) genuine transform pathology (bad at all h) -> needs the bigger
      change-of-variables / Gauss-Jacobi fix
  (b) pole-proximity artifact (bad near h=max_h, fine elsewhere) -> matters
      mainly for participants whose h_map lands near 1.0
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
    find_valid_clonal_structures, jax_cs_hmm_ll_vec, compute_beta_quantiles,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"
RESOLUTIONS = (250, 500, 1_000, 2_000)
S_RESOLUTION, H_RESOLUTION, MIN_S, MAX_S, MAX_H = 15, 3, 0.01, 3.0, 1.0

# fractions of each clone's own [h_min, max_h] grid to test, not raw h values --
# grids differ per clone (h_min depends on that clone's own max VAF), so we
# pick by grid POSITION for a fair comparison across clones/participants.
H_GRID_FRACTIONS = {"h_min (floor)": 0, "interior (mid)": None, "h_max (pole)": -1}


with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)
cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, H_RESOLUTION, MAX_H)

print(f"structure: {cs}  |  resolutions={RESOLUTIONS}")
print(f"h_grids (per clone): {[np.array(g) for g in h_grids]}\n")

for label, pos in H_GRID_FRACTIONS.items():
    # pick position pos for every clone's grid; "mid" = middle index
    h_vec = []
    for g in h_grids:
        g = np.array(g)
        if len(g) == 1:
            h_vec.append(g[0])          # pinned clone, nothing to sweep
        elif pos is None:
            h_vec.append(g[len(g) // 2])
        else:
            h_vec.append(g[pos])
    h0 = jnp.array(h_vec)

    det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)
    feasible = bool(np.all(np.isfinite(np.array(tot)) & (np.array(tot) > 0)))
    print(f"=== {label}  h={np.array(h0)}  feasible={feasible} ===")
    if not feasible:
        print("  (infeasible at this h -- skipping)\n")
        continue

    prev = None
    for res in RESOLUTIONS:
        beta_q = compute_beta_quantiles(AO, DP, observed, res)
        out = np.array(jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, det, tot,
                                         h_mut, observed, carry_idx, beta_q))
        dmax = np.nan if prev is None else np.nanmax(np.abs(out - prev))
        print(f"  resolution={res:>5d}  max_ll={np.nanmax(out):+.6f}  "
              f"{'Δmax=n/a' if prev is None else f'Δmax={dmax:.3e}'}")
        prev = out
    print()

print("Compare Δmax TREND (shrinking vs growing) across the three rows above.")
print("If h_min/interior converge cleanly but h_max diverges like before,")
print("this is a pole-proximity artifact, not a general transform problem.")