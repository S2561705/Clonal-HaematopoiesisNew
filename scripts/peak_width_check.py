"""
peak_width_check.py
====================
Diagnostic: how wide is the beta-binomial emission peak (as a function of
VAF/x), relative to the quadrature node spacing actually used near that
peak? If the peak is much narrower than node spacing, refining q_eps or
beta_resolution won't converge smoothly -- it'll just relocate which nodes
happen to land near the peak, producing non-monotonic error as seen in
qeps_sensitivity_check / node_stability_check.
"""
import sys, os
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
from src.KI_3 import beta_binom_logpmf, _get_arrays

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PIDS = ["MDS1134R53", "MDS711P64"]
PHI = 200.


def emission_peak_width(ao, dp, phi=PHI, n=2000):
    """Width (in VAF units) of the region where the beta-binomial emission
    is within half its peak height, for one (AO, DP) observation."""
    p_grid = np.linspace(1e-4, 1 - 1e-4, n)
    ll = np.array(beta_binom_logpmf(ao, dp, p_grid, phi))
    ll = ll - ll.max()
    above_half = p_grid[ll > np.log(0.5)]
    if len(above_half) == 0:
        return 0.0, p_grid[np.argmax(ll)]
    return above_half.max() - above_half.min(), p_grid[np.argmax(ll)]


with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

for pid in TARGET_PIDS:
    part = next((p for p in cohort if p.uns.get("participant_id") == pid), None)
    if part is None:
        print(f"{pid}: not found in cohort, skipping")
        continue

    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    AO = np.array(AO); DP = np.array(DP); observed = np.array(observed)

    print(f"\n=== {pid} ===")
    print(f"{'tp':>4} {'mut':>4} {'AO':>6} {'DP':>6} {'peak_vaf':>10} {'half_width':>12}")
    for t in range(AO.shape[0]):
        for m in range(AO.shape[1]):
            if not observed[t, m]:
                continue
            width, peak = emission_peak_width(AO[t, m], DP[t, m])
            print(f"{t:>4} {m:>4} {AO[t,m]:>6.0f} {DP[t,m]:>6.0f} "
                  f"{peak:>10.4f} {width:>12.6f}")