# inference_diff.py
#
# Empirically confirms the two-line normalization fix is inference-neutral.
# Runs the FULL ranking + refinement pipeline, snapshots the outputs, and
# diffs a "before" snapshot against an "after" one.
#
# Usage:
#   python inference_diff.py capture before      # with the buggy code
#   # ...apply the two-line fix (delete `- jnp.log(resolution)` in both twins)...
#   python inference_diff.py capture after       # with the fixed code
#   python inference_diff.py compare
#
# Each capture is a fresh process, so JAX recompiles against the current
# source -- no stale jit cache carrying the old term into the "after" run.

import sys
sys.path.append("..")
import pickle as pk
import numpy as np

from src.KI_3 import (
    compute_clonal_models_prob_vec,
    refine_optimal_model_posterior_vec,
)

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS1134R53"

OBS_COLS = [
    "fitness", "fitness_5", "fitness_95",
    "homozygosity", "homozygosity_5", "homozygosity_95",
    "clonal_index",
    "fitness_railed", "homozygosity_railed", "homozygosity_unidentified",
]


def _load_part():
    with open(FITTED_FILE, "rb") as f:
        cohort = pk.load(f)
    return next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)


def _run_pipeline(part):
    part = compute_clonal_models_prob_vec(
        part, s_resolution=20, h_resolution=4, disable_progressbar=True)
    part = refine_optimal_model_posterior_vec(
        part, s_resolution=40, h_resolution=6)
    return part


def _capture(part):
    obs = {c: np.array(part.obs[c].values) for c in OBS_COLS if c in part.obs}
    ranking = [(cs, float(mp)) for cs, mp in part.uns["model_dict"].values()]
    return {"obs": obs, "ranking": ranking, "warning": part.uns.get("warning")}


def capture(label):
    part = _run_pipeline(_load_part())
    snap = _capture(part)
    out = f"inference_diff_{label}.pkl"
    with open(out, "wb") as f:
        pk.dump(snap, f)
    print(f"captured -> {out}   (warning={snap['warning']})")


def compare(a="before", b="after"):
    with open(f"inference_diff_{a}.pkl", "rb") as f: A = pk.load(f)
    with open(f"inference_diff_{b}.pkl", "rb") as f: B = pk.load(f)

    print(f"=== comparing '{a}' vs '{b}' ===\n")

    print("obs columns:")
    all_ok = True
    for c in OBS_COLS:
        if c not in A["obs"] or c not in B["obs"]:
            print(f"  {c:>28}: MISSING in one snapshot"); all_ok = False; continue
        va, vb = A["obs"][c], B["obs"][c]
        if va.dtype == bool or np.issubdtype(va.dtype, np.integer):
            exact = np.array_equal(va, vb)
            print(f"  {c:>28}: {'identical' if exact else 'DIFFERS'}")
            all_ok &= exact
        else:
            d = float(np.nanmax(np.abs(va - vb))) if va.size else 0.0
            tag = "identical" if d == 0 else ("~equal" if d < 1e-9 else "DIFFERS")
            print(f"  {c:>28}: max|Δ|={d:.3e}  ({tag})")
            all_ok &= (d < 1e-9)

    print("\nmodel ranking:")
    ra, rb = A["ranking"], B["ranking"]
    order_a = [cs for cs, _ in ra]
    order_b = [cs for cs, _ in rb]
    same_order = order_a == order_b
    print(f"  same structures, same order : {same_order}")
    if same_order:
        diffs = np.array([mp_b - mp_a for (_, mp_a), (_, mp_b) in zip(ra, rb)])
        print(f"  log-prob offset (after-before): "
              f"min={diffs.min():.6f}  max={diffs.max():.6f}  "
              f"spread={diffs.max()-diffs.min():.3e}")
        print("  -> spread≈0 means the ONLY change is a constant shift")
        print("     (= n_mut * log(beta_resolution)); ranking is preserved.")
    else:
        print("  ORDER CHANGED -- not expected; investigate.")
        all_ok = False

    print("\n" + ("PASS: inference-neutral"
                  if same_order and all_ok
                  else "CHECK: differences beyond tolerance -- see above"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage:\n"
            "  python inference_diff.py capture before   # buggy code\n"
            "  # apply the two-line fix\n"
            "  python inference_diff.py capture after    # fixed code\n"
            "  python inference_diff.py compare")
    cmd = sys.argv[1]
    if cmd == "capture":
        capture(sys.argv[2] if len(sys.argv) > 2 else "snap")
    elif cmd == "compare":
        compare(*(sys.argv[2:4] or ["before", "after"]))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
