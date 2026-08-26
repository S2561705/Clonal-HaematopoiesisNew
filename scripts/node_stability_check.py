"""
node_stability_check.py
=======================
Fix-2 convergence check: the HMM log-likelihood should converge smoothly as
beta_resolution grows. This is the test you need to run BEFORE any q_eps
sweep is interpretable -- if the quadrature itself hasn't converged, the
~2-3% Δp(s) you see in check_qeps_stability is just integrator jitter, not a
q_eps truncation signal.

Kept at a modest max resolution by default because the recursion forms R x R
tensors (vmapped over s and mutations) -- see the OOM at R=4000. Bump
n_resolutions cautiously and/or drop s_resolution if you hit memory limits.
"""
import sys, os, gc
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
from src.KI_3 import (
    check_node_stability,
    build_clone_h_grids, _get_arrays, find_valid_clonal_structures,
)

FITTED_FILE  = "../exports/MDS/MDS_cohort_fitted.pk"
OUT_DIR      = "../exports/figures/MDS/"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_PID   = "MDS1134R53"          # same participant as the q_eps run
N_RESOLUTIONS = (250, 500, 1_000, 2_000)   # keep modest -> R x R memory
S_RESOLUTION  = 15                    # lower than the q_eps run to leave headroom
H_RESOLUTION  = 3
MAX_H         = 1.0


def _has_unpinned_clone(part, h_resolution=H_RESOLUTION, max_h=MAX_H):
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    if 'model_dict' in part.uns and len(part.uns['model_dict']) > 0:
        cs = list(part.uns['model_dict'].values())[0][0]
    else:
        cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                  h_resolution, max_h)
    return any(len(g) > 1 for g in h_grids), cs


with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)
print(f"Participant: {TARGET_PID}\n")

# same guard as qeps_sensitivity_check: the Fix-2 node path only engages when
# a clone is unpinned (h < max_h reachable). A clean run on an all-pinned
# structure proves nothing.
ok, cs = _has_unpinned_clone(part)
assert ok, (
    f"{TARGET_PID} has no unpinned clone -- convergence here is trivial and "
    f"does not exercise the Fix-2 node scheme."
)
print(f"structure: {cs}  |  s_resolution={S_RESOLUTION}  |  "
      f"resolutions={N_RESOLUTIONS}\n")

results = check_node_stability(
    part,
    n_resolutions=N_RESOLUTIONS,
    s_resolution=S_RESOLUTION,
    h_resolution=H_RESOLUTION,
    max_h=MAX_H,
)
gc.collect()

# --- optional: quick convergence summary + overlay plot ---------------------
ress = sorted(results.keys())
print("\nconvergence summary (max-ll vs finest grid):")
finest = np.nanmax(results[ress[-1]])
for r in ress:
    print(f"  R={r:>5d}  max_ll={np.nanmax(results[r]):+.6f}  "
          f"|Δ vs finest|={abs(np.nanmax(results[r]) - finest):.3e}")

try:
    import matplotlib.pyplot as plt
    n_clones = next(iter(results.values())).shape[1]
    fig, axes = plt.subplots(1, n_clones, figsize=(5 * n_clones, 4), squeeze=False)
    axes = axes[0]
    for r in ress:
        out = results[r]
        for k in range(n_clones):
            shifted = out[:, k] - np.nanmax(out[:, k])
            axes[k].plot(np.exp(shifted), label=f"R={r}")
    for k, ax in enumerate(axes):
        ax.set_title(f"clone {k}")
        ax.set_xlabel("s index")
        ax.set_ylabel("normalised p(s)")
        ax.legend(fontsize=7)
    fig.suptitle(f"{TARGET_PID}: p(s) vs beta_resolution (q_eps fixed)")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{TARGET_PID}_node_stability.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nsaved overlay -> {out_png}")
except Exception as e:
    print(f"(plot skipped: {e})")
