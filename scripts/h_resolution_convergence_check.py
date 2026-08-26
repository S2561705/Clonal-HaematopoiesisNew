#!/usr/bin/env python
"""
h_resolution convergence check.

Runs the FULL joint refinement at several h_resolutions on one participant,
saves per-clone p(h), p(s), MAPs and railed flags to a pickle for comparison,
and writes overlay plots + a TV-distance convergence table.

Usage:
    python h_convergence.py            # edit PARTICIPANT_LOADER / PID below
"""
import sys, os, pickle
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from src.KI_3 import (
    _get_arrays, compute_beta_bounds, find_valid_clonal_structures,
    build_clone_h_grids, compute_deterministic_size,
    compute_cs_posterior_grid_vec, clone_posteriors,
    compute_clonal_models_prob_vec, describe_structure,
)

# ----------------------------- config --------------------------------------
H_RESOLUTIONS = (4, 8, 16, 24)     # add 32 if 1-2 clones & memory allows
S_RESOLUTION  = 40                 # HELD FIXED across all runs (apples-to-apples)
MIN_S, MAX_S  = 0.01, 3.0
MAX_H         = 1.0
BETA_RES      = 1_000              # match your production beta_resolution
MAX_COMBOS    = 6_000             # guard against h_res ** n_clones blow-up
OUT_DIR       = "h_convergence_out"
# ---------------------------------------------------------------------------


def _ci90_from_pmf(vals, p):
    """Deterministic 5/95 quantiles from a discrete pmf (no RNG)."""
    vals = np.asarray(vals, float); p = np.asarray(p, float)
    if len(vals) == 1 or p.sum() <= 0:
        return float(vals[0]), float(vals[-1])
    order = np.argsort(vals)
    v, c = vals[order], np.cumsum(p[order] / p.sum())
    lo = float(np.interp(0.05, c, v))
    hi = float(np.interp(0.95, c, v))
    return lo, hi


def run_one_resolution(part, h_res, s_vec, beta_lo, beta_hi):
    """Full joint refinement at a single h_resolution. Returns a dict of
    per-clone posteriors, or None if the grid would be too large."""
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    cs = list(part.uns['model_dict'].values())[0][0]

    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_res, MAX_H)
    n_combo = int(np.prod([len(g) for g in h_grids]))
    if n_combo > MAX_COMBOS:
        print(f"  [skip] h_res={h_res}: {n_combo} h-combos exceeds "
              f"MAX_COMBOS={MAX_COMBOS} ({len(cs)} clones). Use the "
              f"coordinate-refinement method for this one instead.")
        return None, cs

    out_grid, h_combos = compute_cs_posterior_grid_vec(
        s_vec, h_grids, AO, DP, time_points, cs, observed, carry_idx,
        beta_lo, beta_hi, resolution=BETA_RES)
    posteriors = clone_posteriors(out_grid, h_combos, s_vec, h_grids)

    s_arr = np.array(s_vec)
    s_top = float(s_arr.max()); eps = 1e-9; H_CI_WIDTH = 0.5
    clones = []
    for k, (h_vals, joint) in enumerate(posteriors):
        joint = np.nan_to_num(np.asarray(joint), nan=0.0, posinf=0.0, neginf=0.0)
        p_s = joint.sum(axis=0); p_h = joint.sum(axis=1)
        if p_s.sum() <= 0 or p_h.sum() <= 0:
            print(f"  [warn] h_res={h_res} clone {k}: zero posterior")
            continue
        p_s = p_s / p_s.sum(); p_h = p_h / p_h.sum()

        s_map = float(s_arr[np.argmax(p_s)])
        h_map = float(np.array(h_vals)[np.argmax(p_h)])
        s_ci  = _ci90_from_pmf(s_arr, p_s)
        h_ci  = _ci90_from_pmf(h_vals, p_h)

        h_unident = (len(h_vals) > 1) and ((h_ci[1] - h_ci[0]) > H_CI_WIDTH)
        h_railed  = (len(h_vals) > 1) and (h_map >= MAX_H - eps) and not h_unident
        s_railed  = s_map >= s_top - eps

        clones.append(dict(
            clone=k, h_vals=np.array(h_vals), p_h=p_h,
            s_vals=s_arr, p_s=p_s,
            h_map=h_map, s_map=s_map, h_ci90=h_ci, s_ci90=s_ci,
            h_railed=h_railed, h_unident=h_unident, s_railed=s_railed,
            n_h=len(h_vals)))
    return clones, cs


def _tv_on_common_grid(h_a, p_a, h_b, p_b, n=400):
    """Total-variation distance between two p(h) densities defined on
    (possibly different) grids over the same [h_min, 1] support."""
    lo = max(h_a.min(), h_b.min()); hi = min(h_a.max(), h_b.max())
    if hi <= lo:
        return np.nan
    g = np.linspace(lo, hi, n)
    fa = np.interp(g, h_a, p_a); fb = np.interp(g, h_b, p_b)
    fa = fa / np.trapz(fa, g); fb = fb / np.trapz(fb, g)
    return 0.5 * np.trapz(np.abs(fa - fb), g)


def convergence_check(part, pid=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    if 'model_dict' not in part.uns:
        print("model_dict not found -- running compute_clonal_models_prob_vec first")
        compute_clonal_models_prob_vec(part, beta_resolution=BETA_RES,
                                       disable_progressbar=True)

    AO, DP, observed, *_ = _get_arrays(part)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)
    s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)

    results = {}
    cs = None
    for h_res in H_RESOLUTIONS:
        print(f"[{pid}] h_resolution = {h_res} ...")
        clones, cs = run_one_resolution(part, h_res, s_vec, beta_lo, beta_hi)
        if clones is not None:
            results[h_res] = clones

    if not results:
        print("no resolutions completed -- nothing to save")
        return

    # ---- save raw results -------------------------------------------------
    payload = dict(pid=pid, cs=cs, s_resolution=S_RESOLUTION,
                   min_s=MIN_S, max_s=MAX_S, beta_res=BETA_RES,
                   results=results)
    pkl = os.path.join(OUT_DIR, f"{pid}_h_convergence.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved raw results -> {pkl}")

    n_clones = len(cs)
    res_sorted = sorted(results.keys())

    # ---- convergence table (MAP-h, MAP-s, TV vs previous res) -------------
    print("\n" + "=" * 78)
    print(f"{pid}  CONVERGENCE (s_resolution fixed at {S_RESOLUTION})")
    print("=" * 78)
    for k in range(n_clones):
        print(f"\nclone {k}:")
        print(f"  {'h_res':>6} {'MAP-h':>8} {'MAP-s':>8} "
              f"{'h_CI90':>16} {'flags':>20} {'TV(p(h)) vs prev':>18}")
        prev = None
        for r in res_sorted:
            cl = next((c for c in results[r] if c['clone'] == k), None)
            if cl is None:
                continue
            tv = np.nan
            if prev is not None:
                tv = _tv_on_common_grid(prev['h_vals'], prev['p_h'],
                                        cl['h_vals'], cl['p_h'])
            flags = []
            if cl['h_railed']:   flags.append("h-railed")
            if cl['h_unident']:  flags.append("h-unident")
            if cl['s_railed']:   flags.append("s-railed")
            print(f"  {r:>6} {cl['h_map']:>8.3f} {cl['s_map']:>8.3f} "
                  f"[{cl['h_ci90'][0]:.2f},{cl['h_ci90'][1]:.2f}]".rjust(16)
                  + f"{','.join(flags) or '-':>20}"
                  + (f"{tv:>18.3e}" if np.isfinite(tv) else f"{'-':>18}"))
            prev = cl
    print("\nInterpretation: if MAP-h keeps moving and TV(p(h)) doesn't shrink")
    print("toward 0 as h_res grows, h is NOT converged (or is unidentifiable")
    print("from the data). A shrinking TV with a stable MAP = converged.\n")

    # ---- overlay plots ----------------------------------------------------
    fig, axes = plt.subplots(2, n_clones, figsize=(5 * n_clones, 8),
                             squeeze=False)
    for k in range(n_clones):
        ax_h, ax_s = axes[0][k], axes[1][k]
        for r in res_sorted:
            cl = next((c for c in results[r] if c['clone'] == k), None)
            if cl is None:
                continue
            ax_h.plot(cl['h_vals'], cl['p_h'], marker='o', ms=3,
                      label=f"h_res={r}")
            ax_s.plot(cl['s_vals'], cl['p_s'], label=f"h_res={r}")
        ax_h.set_title(f"clone {k}: p(h)")
        ax_h.set_xlabel("h"); ax_h.set_ylabel("p(h)"); ax_h.legend(fontsize=7)
        ax_s.set_title(f"clone {k}: p(s)")
        ax_s.set_xlabel("s"); ax_s.set_ylabel("p(s)"); ax_s.legend(fontsize=7)
    fig.suptitle(f"{pid}: posterior vs h_resolution (s_res fixed={S_RESOLUTION})")
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{pid}_h_convergence.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"saved overlay plot -> {png}")


if __name__ == "__main__":
    import pickle as pk

    INPUT_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
    TARGET_PID = "MDS711P64"

    with open(INPUT_FILE, "rb") as f:
        cohort = pk.load(f)

    cohort = [p for p in cohort
              if p.uns.get("participant_id") == TARGET_PID]
    print(f"Filtered to target participant: {len(cohort)} found")
    assert len(cohort) == 1, \
        f"expected exactly 1 match for {TARGET_PID}, got {len(cohort)}"
    part = cohort[0]

    convergence_check(part, pid=TARGET_PID)

