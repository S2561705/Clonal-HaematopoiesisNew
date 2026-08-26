"""
sensitivity_check.py
====================
Fix-2 diagnostic under deterministic Beta-quantile nodes (KI_3).

Absolute LL often drifts with resolution (−log R + pole/tail). What matters:
  • MAP-s locking across resolutions
  • Δshape (max-shifted LL) shrinking
  • clip% of x-nodes hitting X_MAX not exploding with resolution

Also writes an overlay plot of normalised p(s) vs s for each resolution.

NOTE: each forward step materialises an R×R transition grid, so R ≳ 5_000
easily OOMs (OS "zsh: killed"). Keep the ladder modest.
"""

import sys
import os
import gc
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
from src.KI_3 import check_node_stability, plot_node_stability

TARGET_PID  = "MDS711P64"
FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
OUT_DIR     = "../exports/figures/MDS/"
os.makedirs(OUT_DIR, exist_ok=True)

# Modest ladder only — R×R transitions make 10k/20k impractical on a laptop
N_RESOLUTIONS = (500, 1_000, 2_000, 3_000, 4_000)
S_RESOLUTION  = 40

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)
print(f"Participant: {TARGET_PID}")
print(f"s_resolution={S_RESOLUTION}, beta resolutions={N_RESOLUTIONS}")
print("(capped well below OOM; R×R transition grids blow up fast)\n")

results = check_node_stability(
    part,
    n_resolutions=N_RESOLUTIONS,
    s_resolution=S_RESOLUTION,
)
gc.collect()

out_png = os.path.join(OUT_DIR, f"{TARGET_PID}_node_stability.png")
plot_node_stability(results, out_png, pid=TARGET_PID)
