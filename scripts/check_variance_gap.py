"""
check_variance_gap.py
=====================
Sweep all h-combos in the production grid and report, for each clone, whether
ANY (s, h-combo) pair produces variance <= mean in BD_process_dynamics
(n = mean^2/(var-mean) goes negative → NaN NB transitions).

Uses KI_3 (canonical). Independent of Fix 2 — this is the Δt / λ issue Linus
flagged as a separate concern from the quadrature spikes.
"""

import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import itertools
import pickle as pk

from src.KI_3 import (
    _get_arrays, compute_beta_quantiles, compute_deterministic_size,
    build_clone_h_grids, X_MAX_MULTIPLIER, N_w,
)

# ── Config ───────────────────────────────────────────────────────────────
TARGET_PID   = "MDS711P64"
FITTED_FILE  = "../exports/MDS/MDS_cohort_fitted.pk"
S_RESOLUTION = 40
MIN_S, MAX_S = 0.01, 3.0
LAMB         = 1.3
H_RESOLUTION = 6

# ── Load participant + known clonal structure ──────────────────────────
with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)
cs = list(part.uns["model_dict"].values())[0][0]
n_clones = len(cs)
print(f"Clonal structure: {cs}  ({n_clones} clones)")

AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
AO = np.array(AO); DP = np.array(DP); observed = np.array(observed)
carry_idx = np.array(carry_idx)
time_points = np.array(time_points)
delta_t = np.diff(time_points)
s_vec = np.linspace(MIN_S, MAX_S, S_RESOLUTION)

beta_q_vec = np.array(compute_beta_quantiles(AO, DP, observed, resolution=1000))
h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution=H_RESOLUTION)
h_grids_np = [np.array(g) for g in h_grids]
n_combos = int(np.prod([len(g) for g in h_grids_np]))
print(f"Sweeping {n_combos} h-combos x {S_RESOLUTION} s-values "
      f"= {n_combos * S_RESOLUTION} (s, h) pairs total.\n")

bad_s_values = [set() for _ in range(n_clones)]
n_bad_pairs = [0 for _ in range(n_clones)]
n_feasible_combos = 0
X_MAX = X_MAX_MULTIPLIER * N_w

for combo_idx in itertools.product(*[range(len(g)) for g in h_grids_np]):
    h = np.array([h_grids_np[k][combo_idx[k]] for k in range(n_clones)])

    det_size, total_cells, _, h_mut = compute_deterministic_size(
        cs, AO, DP, AO.shape[1], h, observed)
    det_size = np.array(det_size); total_cells = np.array(total_cells); h_mut = np.array(h_mut)

    feasible = np.all(np.isfinite(total_cells) & (total_cells > 0))
    if not feasible:
        continue
    n_feasible_combos += 1

    N_w_cond = (total_cells[:, None] - det_size)[:, :, None]
    half = (1.0 + h_mut)[None, :, None] / 2.0
    x_vec = -N_w_cond * beta_q_vec / (beta_q_vec - half)
    valid = x_vec > 0
    x_vec = np.clip(np.where(valid, x_vec, X_MAX), 1e-6, X_MAX)
    idx = np.broadcast_to(carry_idx[:, :, None], x_vec.shape)
    x_vec = np.take_along_axis(x_vec, idx, axis=0)

    for clone_idx in range(n_clones):
        mut_idx = cs[clone_idx][0]
        x_m = x_vec[:, mut_idx, :]

        for si, s in enumerate(s_vec):
            exp_term = np.exp(delta_t * s)[:, None]
            mean = x_m[:-1] * exp_term
            var = x_m[:-1] * (2 * LAMB + s) * exp_term * (exp_term - 1) / s
            gap = var - mean
            if np.min(gap) <= 0:
                bad_s_values[clone_idx].add(si)
                n_bad_pairs[clone_idx] += 1

print(f"{n_feasible_combos}/{n_combos} h-combos were feasible "
      f"(finite, positive total_cells).\n")

for clone_idx in range(n_clones):
    n_bad_s = len(bad_s_values[clone_idx])
    print(f"Clone {clone_idx} (mutation idx {cs[clone_idx][0]}):")
    print(f"  {n_bad_pairs[clone_idx]} / {n_feasible_combos * S_RESOLUTION} "
          f"(s, h-combo) pairs had variance <= mean")
    print(f"  {n_bad_s} / {S_RESOLUTION} distinct s-VALUES were affected "
          f"by at least one h-combo")
    if n_bad_s > 0:
        bad_s_sorted = sorted(bad_s_values[clone_idx])
        bad_s_actual = [round(float(s_vec[i]), 3) for i in bad_s_sorted]
        print(f"  affected s-values: {bad_s_actual}")
    print()
