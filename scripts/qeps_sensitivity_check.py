"""
qeps_sensitivity_check.py
==========================
Isolates Q_EPS truncation error from ordinary beta_resolution quadrature
error, on a participant where the fix under test can actually engage.

IMPORTANT: the anchoring fix in compute_global_variables only does anything
when at least one clone in the winning structure has h < 1.0 (i.e. an
UNPINNED clone -- h_fixed not set to 1.0, so build_clone_h_grids gives it a
real grid rather than a size-1 pinned grid). When every clone is pinned at
h=1.0, `half = (1+h)/2 = 1.0` sits exactly at the supremum of Beta's open
support (0,1), so beta_q < half ALWAYS holds, `valid` is True everywhere,
and the old X_MAX-clip code and the new last-valid-anchor code are
mathematically identical -- a clean run there proves nothing about the fix.
This script finds a participant/structure where the fix path is actually
exercised before testing it.
"""
import sys, os, gc
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
from src.KI_3 import (
    check_qeps_stability, plot_qeps_stability,
    build_clone_h_grids, _get_arrays, find_valid_clonal_structures,
)

FITTED_FILE  = "../exports/MDS/MDS_cohort_fitted.pk"
OUT_DIR      = "../exports/figures/MDS/"
os.makedirs(OUT_DIR, exist_ok=True)

Q_EPS_LADDER    = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
BETA_RESOLUTION = 500
S_RESOLUTION    = 40
H_RESOLUTION    = 3
MAX_H           = 1.0


def has_unpinned_clone(part, h_resolution=H_RESOLUTION, max_h=MAX_H):
    """True if the participant's winning (or first valid) structure has at
    least one clone whose h grid is NOT a pinned size-1 grid -- i.e. the
    pole (half = (1+h)/2) is actually reachable by some h < max_h."""
    try:
        AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
        if 'model_dict' in part.uns and len(part.uns['model_dict']) > 0:
            cs = list(part.uns['model_dict'].values())[0][0]
        else:
            cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
        h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                      h_resolution, max_h)
        return any(len(g) > 1 for g in h_grids), cs
    except Exception:
        return False, None


with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

# --- find a usable participant: unpinned clone present, and previously
#     flagged as spiky if that's tracked; otherwise just report what's found
candidates = []
for part in cohort:
    if part.uns.get('fit_failed'):
        continue
    ok, cs = has_unpinned_clone(part)
    if ok:
        candidates.append((part.uns.get('participant_id'), part, cs))

print(f"Scanned {len(cohort)} participants: {len(candidates)} have at least "
      f"one unpinned clone in their winning structure.")

if not candidates:
    raise RuntimeError(
        "No participant in this cohort has an unpinned clone -- the "
        "anchoring fix cannot be exercised by this dataset as filtered. "
        "Check h_fixed assignment / whether every mutation here is being "
        "forced homozygous upstream."
    )

for pid, _, cs in candidates[:10]:
    print(f"  {pid}: structure {cs}")

TARGET_PID = candidates[0][0]
part = candidates[0][1]
print(f"\nUsing participant: {TARGET_PID}\n")

# --- explicit guard: fail loudly rather than silently repeating last time's
#     no-op test if something about the chosen participant changes upstream
ok, _ = has_unpinned_clone(part)
assert ok, (
    f"{TARGET_PID} no longer has an unpinned clone at test time -- "
    f"refusing to run a test that can't exercise the fix."
)

print(f"beta_resolution={BETA_RESOLUTION} (fixed), q_eps ladder={Q_EPS_LADDER}\n")

results = check_qeps_stability(
    part,
    q_eps_list=Q_EPS_LADDER,
    beta_resolution=BETA_RESOLUTION,
    s_resolution=S_RESOLUTION,
    h_resolution=H_RESOLUTION,
    max_h=MAX_H,
)
gc.collect()

out_png = os.path.join(OUT_DIR, f"{TARGET_PID}_qeps_stability.png")
plot_qeps_stability(results, out_png, pid=TARGET_PID)