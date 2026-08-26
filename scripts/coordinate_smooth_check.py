#!/usr/bin/env python
"""
coordinate_smooth_check.py
============================
Produces smooth, high-resolution p(s)/p(h) posteriors via the
coordinate-refinement method (linear cost in h_resolution, unlike the
full-joint method's h_resolution**n_clones), specifically for
presentation/figure purposes once the comb-artifact diagnosis is
confirmed (see per_h_slice_check.py).

IMPORTANT CAVEAT (from refine_optimal_model_posterior_coordinate's own
docstring): this gives a PROFILE posterior per clone -- other clones held
at their coordinate-ascent-converged point estimate, not integrated over.
That's the right tradeoff for readability, but it is not numerically
identical to the full joint marginal. This script overlays against the
h_res=16 full-joint result (already computed by
h_resolution_convergence_check.py) so you can confirm the MAP and overall
shape agree before treating this as the figure of record, rather than
assuming the profile approximation is fine.
"""
import sys, os, pickle
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pickle as pk

from src.KI_3 import (
    _get_arrays, compute_beta_bounds, build_clone_h_grids,
    coordinate_refine_h, _eval_h_vec,
)

INPUT_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS711P64"

S_RESOLUTION = 40
MIN_S, MAX_S = 0.01, 3.0
MAX_H = 1.0
BETA_RES = 1_000               # match production / validated resolution
H_RES_SMOOTH = 48              # the whole point: much finer than the h_res**n_clones ceiling allowed
N_CYCLES = 4                   # coordinate-ascent cycles; a bit more than the default 3 for a final figure

FULL_JOINT_PKL = "h_convergence_out/MDS711P64_h_convergence.pkl"  # for overlay comparison
OUT_DIR = "h_convergence_out"


def _ci90_from_pmf(vals, p):
    vals = np.asarray(vals, float); p = np.asarray(p, float)
    if len(vals) == 1 or p.sum() <= 0:
        return float(vals[0]), float(vals[-1])
    order = np.argsort(vals)
    v, c = vals[order], np.cumsum(p[order] / p.sum())
    return float(np.interp(0.05, c, v)), float(np.interp(0.95, c, v))


def run_coordinate_smooth(part, pid):
    os.makedirs(OUT_DIR, exist_ok=True)
    cs = list(part.uns['model_dict'].values())[0][0]
    n_clones = len(cs)
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)
    s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)
    s_arr = np.array(s_vec)

    fine_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                     H_RES_SMOOTH, MAX_H)
    print(f"structure: {cs}  |  h_res={H_RES_SMOOTH}  |  n_clones={n_clones}")
    print(f"cost ~ n_cycles * n_clones * h_res = "
          f"{N_CYCLES * n_clones * H_RES_SMOOTH} likelihood evals "
          f"(vs {H_RES_SMOOTH**n_clones} for full-joint)")

    h_converged = coordinate_refine_h(
        cs, AO, DP, time_points, observed, carry_idx, s_vec, fine_grids,
        beta_lo, beta_hi, n_cycles=N_CYCLES)
    print(f"converged h per clone: {h_converged}")

    results = []
    for k, c_idx in enumerate(cs):
        grid_k = np.array(fine_grids[k])
        joint_log = np.full((len(grid_k), S_RESOLUTION), -np.inf)
        for gi, hk in enumerate(grid_k):
            trial = h_converged.copy()
            trial[k] = hk
            _, out = _eval_h_vec(cs, AO, DP, time_points, observed,
                                 carry_idx, s_vec, jnp.array(trial),
                                 beta_lo, beta_hi)
            if out is not None:
                joint_log[gi] = out[:, k]
        mx = np.nanmax(joint_log)
        joint = np.exp(joint_log - mx) if np.isfinite(mx) else np.zeros_like(joint_log)

        p_h = joint.sum(axis=1); p_h = p_h / p_h.sum() if p_h.sum() > 0 else p_h
        p_s = joint.sum(axis=0); p_s = p_s / p_s.sum() if p_s.sum() > 0 else p_s

        h_map = float(grid_k[np.argmax(p_h)])
        s_map = float(s_arr[np.argmax(p_s)])
        h_ci = _ci90_from_pmf(grid_k, p_h)
        s_ci = _ci90_from_pmf(s_arr, p_s)

        results.append(dict(clone=k, h_vals=grid_k, p_h=p_h,
                            s_vals=s_arr, p_s=p_s,
                            h_map=h_map, s_map=s_map,
                            h_ci90=h_ci, s_ci90=s_ci))
        print(f"  clone {k}: MAP-h={h_map:.3f}  90%CI=[{h_ci[0]:.2f},{h_ci[1]:.2f}]  "
              f"MAP-s={s_map:.3f}")

    payload = dict(pid=pid, cs=cs, h_res=H_RES_SMOOTH, n_cycles=N_CYCLES,
                   s_resolution=S_RESOLUTION, results=results,
                   method="coordinate_refinement_profile")
    pkl = os.path.join(OUT_DIR, f"{pid}_coordinate_smooth.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved -> {pkl}")

    # ---- overlay against full-joint h_res=16 if available -----------------
    full_joint = None
    if os.path.exists(FULL_JOINT_PKL):
        with open(FULL_JOINT_PKL, "rb") as f:
            fj = pickle.load(f)
        full_joint = fj['results'].get(16)

    fig, axes = plt.subplots(2, n_clones, figsize=(5.5 * n_clones, 8), squeeze=False)
    for k in range(n_clones):
        ax_h, ax_s = axes[0][k], axes[1][k]
        r = results[k]
        ax_h.plot(r['h_vals'], r['p_h'], color='C0', lw=1.8,
                  label=f'coordinate, h_res={H_RES_SMOOTH} (smooth)')
        ax_h.fill_between(r['h_vals'], r['p_h'], alpha=0.15, color='C0')
        ax_s.plot(r['s_vals'], r['p_s'], color='C0', lw=1.8,
                  label=f'coordinate, h_res={H_RES_SMOOTH}')
        ax_s.fill_between(r['s_vals'], r['p_s'], alpha=0.15, color='C0')

        if full_joint is not None:
            fj_cl = next((c for c in full_joint if c['clone'] == k), None)
            if fj_cl is not None:
                ax_h.plot(fj_cl['h_vals'], fj_cl['p_h'], color='C1', lw=1.2,
                          ls='--', marker='o', ms=3,
                          label='full-joint, h_res=16 (reference)')
                ax_s.plot(fj_cl['s_vals'], fj_cl['p_s'], color='C1', lw=1.2,
                          ls='--', label='full-joint, h_res=16 (reference)')

        ax_h.set_title(f"clone {k}: p(h)")
        ax_h.set_xlabel("h"); ax_h.set_ylabel("p(h)"); ax_h.legend(fontsize=7)
        ax_s.set_title(f"clone {k}: p(s)")
        ax_s.set_xlabel("s"); ax_s.set_ylabel("p(s)"); ax_s.legend(fontsize=7)

    fig.suptitle(f"{pid}: coordinate-refinement smooth posterior "
                f"(h_res={H_RES_SMOOTH}) vs full-joint reference (h_res=16)")
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{pid}_coordinate_smooth.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"saved overlay -> {png}")


if __name__ == "__main__":
    with open(INPUT_FILE, "rb") as f:
        cohort = pk.load(f)
    cohort = [p for p in cohort if p.uns.get("participant_id") == TARGET_PID]
    assert len(cohort) == 1, f"expected exactly 1 match for {TARGET_PID}"
    part = cohort[0]
    run_coordinate_smooth(part, TARGET_PID)