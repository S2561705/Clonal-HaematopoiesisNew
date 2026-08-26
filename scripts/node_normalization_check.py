# node_normalization_check.py
#
# Confirms the diagnosis: the resolution-dependent bias in each clone's
# log-likelihood is exactly -M_k * log(resolution), where M_k = number of
# mutations in that clone -- caused by the `- jnp.log(resolution)` term in
# compute_global_variables (applied per mutation, then summed per clone).
#
# It does this WITHOUT editing KI_3.py, by:
#   (1) reading each clone's max-over-s log-lik at a ladder of resolutions,
#   (2) showing the per-doubling drop tracks -M_k * ln(2), and
#   (3) adding +M_k*log(res) back ("the fix, applied analytically") and
#       showing the corrected curve converges (Δ shrinks toward 0).
#
# If (2) holds and (3) converges, the one-line fix (delete `- jnp.log(res)`)
# is confirmed correct and inference-neutral.

import sys
sys.path.append("..")
import numpy as np
import jax.numpy as jnp

from src.KI_3 import (
    _get_arrays, find_valid_clonal_structures, build_clone_h_grids,
    compute_deterministic_size, compute_beta_bounds, jax_cs_hmm_ll_vec,
)


def _clone_max_ll(part, cs, resolutions, s_resolution=15,
                  h_resolution=3, min_s=0.01, max_s=3.0, max_h=1.0):
    """Return dict res -> array of per-clone max-over-s log-lik (length K)."""
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)
    h0 = jnp.array([g[-1] for g in h_grids])
    det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

    out = {}
    for res in resolutions:
        ll = np.array(jax_cs_hmm_ll_vec(
            s_vec, AO, DP, time_points, cs, det, tot, h_mut, observed,
            carry_idx, beta_lo, beta_hi, resolution=res))          # (s_res, K)
        out[res] = np.nanmax(ll, axis=0)                            # (K,)
    return out


def _report(name, cs, res_ll):
    resolutions = sorted(res_ll.keys())
    K = len(cs)
    print(f"\\n=== {name} : structure {cs} ===")
    print(f"{'clone':>6} {'M_k':>4} | "
          f"{'per-doubling Δ (raw)':>22} | pred -M·ln2 | "
          f"{'per-doubling Δ (fixed)':>22}")
    print("-" * 86)
    for k in range(K):
        M = len(cs[k])
        raw = np.array([res_ll[r][k] for r in resolutions])
        # "fix applied analytically": add back the M*log(res) the bug subtracted
        fixed = raw + M * np.log(np.array(resolutions, dtype=float))

        raw_d, fix_d = [], []
        for a, b in zip(resolutions[:-1], resolutions[1:]):
            ratio = np.log(b / a)                 # ln 2 for doublings
            i, j = resolutions.index(a), resolutions.index(b)
            # normalise each step to a "per ln2" slope so mixed ratios compare
            raw_d.append((raw[j] - raw[i]) / ratio * np.log(2))
            fix_d.append((fixed[j] - fixed[i]) / ratio * np.log(2))
        raw_s = "  ".join(f"{d:+7.3f}" for d in raw_d)
        fix_s = "  ".join(f"{d:+7.3f}" for d in fix_d)
        print(f"{k:>6} {M:>4} | {raw_s:>22} | {-M*np.log(2):>10.3f}  | {fix_s:>22}")
    print("  raw Δ should ≈ pred (-M·ln2); fixed Δ should shrink toward 0.")


def run(part, resolutions=(500, 1_000, 2_000, 4_000)):
    """Drive the check on a few structures chosen to span mutation-per-clone
    counts, so the exponent-tracks-M_k prediction is visible in one run."""
    n_mut = part.shape[0]

    structures = {}
    # all singletons -> every clone M=1 -> slope should be -ln2 everywhere
    structures["all singletons (M=1)"] = [[i] for i in range(n_mut)]
    if n_mut >= 2:
        # one big clone -> single clone with M=n_mut -> slope -n_mut*ln2
        structures[f"one clone (M={n_mut})"] = [list(range(n_mut))]
    if n_mut >= 3:
        # mixed: a 2-clone and singletons -> slopes -2ln2 and -ln2 side by side
        structures["mixed (M=2 + singletons)"] = (
            [[0, 1]] + [[i] for i in range(2, n_mut)])

    for name, cs in structures.items():
        det_ok = True
        try:
            res_ll = _clone_max_ll(part, cs, resolutions)
        except Exception as e:
            print(f"\\n=== {name} : structure {cs} ===  SKIPPED ({e})")
            continue
        _report(name, cs, res_ll)


if __name__ == "__main__":
    import pickle as pk

    FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
    TARGET_PID = "MDS1134R53"

    with open(FITTED_FILE, "rb") as f:
        cohort = pk.load(f)
    part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

    run(part)

