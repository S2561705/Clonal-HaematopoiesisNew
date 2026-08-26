#!/usr/bin/env python
"""
per_h_slice_check.py
=====================
Decomposes the p(s) comb into its per-h-slice components, to directly test
whether it's "sum of N shifted unimodal curves" (h-grid leaking through the
s-marginalisation) rather than assume it from the marginal shape alone.

clone_posteriors already returns, per clone, `joint` of shape (n_h, n_s) --
each row IS that h-value's own s-curve, unsummed. p_s = joint.sum(axis=0)
is exactly summing these rows. This script plots them individually.
"""
import sys, os
sys.path.append("..")
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import jax.numpy as jnp

from src.KI_3 import (
    _get_arrays, compute_beta_bounds, build_clone_h_grids,
    compute_cs_posterior_grid_vec, clone_posteriors,
)

INPUT_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS711P64"
H_RES = 8          # the resolution to decompose -- pick one that shows a comb
S_RESOLUTION = 40
MIN_S, MAX_S = 0.01, 3.0
MAX_H = 1.0
BETA_RES = 1_000
OUT_DIR = "h_convergence_out"


def decompose(part, pid, h_res):
    os.makedirs(OUT_DIR, exist_ok=True)
    cs = list(part.uns['model_dict'].values())[0][0]
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)
    s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)
    s_arr = np.array(s_vec)

    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_res, MAX_H)
    n_combo = int(np.prod([len(g) for g in h_grids]))
    print(f"structure: {cs}  |  h_res={h_res}  |  n_combo={n_combo}")

    out_grid, h_combos = compute_cs_posterior_grid_vec(
        s_vec, h_grids, AO, DP, time_points, cs, observed, carry_idx,
        beta_lo, beta_hi, resolution=BETA_RES)
    posteriors = clone_posteriors(out_grid, h_combos, s_vec, h_grids)

    n_clones = len(cs)
    fig, axes = plt.subplots(1, n_clones, figsize=(6 * n_clones, 5), squeeze=False)
    axes = axes[0]

    for k in range(n_clones):
        h_vals, joint = posteriors[k]          # joint: (n_h, n_s)
        joint = np.nan_to_num(np.asarray(joint), nan=0.0, posinf=0.0, neginf=0.0)
        h_vals = np.asarray(h_vals)

        ax = axes[k]
        cmap = cm.get_cmap('viridis', len(h_vals))
        row_peaks = []
        for hi, hv in enumerate(h_vals):
            row = joint[hi]
            row_sum = row.sum()
            if row_sum <= 0:
                continue
            row_norm = row / row_sum            # normalise EACH slice to compare shapes
            peak_s = s_arr[np.argmax(row)]
            row_peaks.append((hv, peak_s, row_sum))
            ax.plot(s_arr, row_norm, color=cmap(hi), lw=1.2, alpha=0.8,
                    label=f"h={hv:.2f} (peak s={peak_s:.3f})")

        # the actual marginal, for comparison -- UNnormalised-per-row sum,
        # i.e. exactly what clone_posteriors/refine_* actually compute
        p_s = joint.sum(axis=0)
        p_s = p_s / p_s.sum() if p_s.sum() > 0 else p_s
        ax.plot(s_arr, p_s, color='red', lw=2.5, ls='--', label='MARGINAL (sum)', zorder=10)

        ax.set_title(f"clone {k}: per-h-slice p(s|h) vs marginal")
        ax.set_xlabel("s"); ax.set_ylabel("p(s | h)  [each row independently normalised]")
        ax.legend(fontsize=6, loc='upper right')

        print(f"\nclone {k} per-h-slice peaks (h, peak_s, row_mass):")
        for hv, peak_s, row_sum in row_peaks:
            print(f"  h={hv:.3f}  peak_s={peak_s:.3f}  row_mass={row_sum:.3e}")

    fig.suptitle(f"{pid}: decomposing p(s) comb into per-h-slice components (h_res={h_res})")
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{pid}_per_h_slice_h{h_res}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {png}")


if __name__ == "__main__":
    with open(INPUT_FILE, "rb") as f:
        cohort = pk.load(f)
    cohort = [p for p in cohort if p.uns.get("participant_id") == TARGET_PID]
    assert len(cohort) == 1
    part = cohort[0]
    decompose(part, TARGET_PID, H_RES)