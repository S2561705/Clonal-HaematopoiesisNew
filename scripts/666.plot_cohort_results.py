#!/usr/bin/env python3
"""
plot_clonal_summary.py
======================
Per-participant figure layout
  COL 0 (spans all rows) : VAF over time, coloured by clone
                           leading mutation = solid line, sub-clonal = dashed
  COL 1 (one row/clone)  : Marginal fitness  p(s) posterior
  COL 2 (one row/clone)  : Marginal zygosity p(h) posterior
  COL 3 (one row/clone)  : Joint p(s, h) posterior heatmap   <-- NEW

Inference flags annotated on panel titles
  ⚠ railed          MAP s at the max_s ceiling  → treat as lower bound
  ⚠ h railed        MAP h at max_h ceiling      → full LOH
  ⚠ h unidentified  90 % CI > 0.5              → do not read as LOH

The joint panel is the key diagnostic: a DIAGONAL ridge means s and h are
jointly (not individually) identified -- the marginals will look multimodal
even though the joint is a clean ridge. A COMPACT blob means both are well
determined (trust the MAP).
"""
import sys, os, warnings
sys.path.append("..")
warnings.filterwarnings("ignore")

from src.general_imports import *
from src.KI_3 import clone_posteriors

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
OUTPUT_DIR = "../exports/figures/MDS/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOURS   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DPI       = 150
FIG_W     = 20        # inches total width (wider now: 4 columns)
N_SAMPLES = 5_000     # bootstrap samples for percentile CI


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_norm(arr):
    """Zero NaN / Inf entries, normalise to sum = 1."""
    arr = np.nan_to_num(np.asarray(arr, float),
                        nan=0.0, posinf=0.0, neginf=0.0)
    s = arr.sum()
    return arr / s if s > 0 else arr


def _map_ci(vals, weights, lo=0.05, hi=0.95):
    """MAP and (lo, hi) percentile CI from a weighted discrete distribution."""
    weights = _safe_norm(weights)
    if weights.max() == 0:
        return float("nan"), (float("nan"), float("nan"))
    map_val = float(vals[np.argmax(weights)])
    draws   = np.random.choice(vals, size=N_SAMPLES, p=weights)
    return map_val, (float(np.quantile(draws, lo)),
                     float(np.quantile(draws, hi)))


def _vaf_dp_tp(part):
    """Return (vaf, DP, time_points) all shaped (n_tp, n_mut)."""
    AO  = part.layers["AO"].T.astype(float)
    DP  = part.layers["DP"].T.astype(float)
    vaf = AO / np.maximum(DP, 1.0)
    tp  = np.array(part.var["time_points"], dtype=float)
    return vaf, DP, tp


def _get_posteriors(part):
    """Run clone_posteriors on the stored log-posterior grid.
    Each returned `joint` is the 2-D p(s, h) for that clone, shape (n_h, n_s)."""
    m = part.uns["optimal_model"]
    posts = clone_posteriors(
        np.array(m["posterior"]),            # (n_combo, s_res, K)
        np.array(m["h_combos"]),             # (n_combo, K)
        np.array(m["s_range"]),
        [np.array(g) for g in m["h_grids"]],
    )
    return posts, np.array(m["s_range"])


def _obs_flag(part, col, rep_idx, default=False):
    """Safely read a boolean flag from part.obs."""
    if col in part.obs.columns:
        return bool(part.obs[col].iloc[rep_idx])
    return default


# ═══════════════════════════════════════════════════════════════════════════════
# Panel drawers
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_vaf_panel(ax, part, cs, ms, colours):
    """VAF-over-time panel.  Leading mut = solid circle-line; others = dashed."""
    vaf, DP, tp = _vaf_dp_tp(part)

    legend_patches = []
    for k, (c_idx, c_names) in enumerate(zip(cs, ms)):
        col   = colours[k % len(colours)]
        label = " | ".join(c_names) if c_names else f"Clone {k}"
        legend_patches.append(mpatches.Patch(color=col, label=label))

        # leading mutation = highest cumulative observed VAF
        obs_vaf = np.where(DP[:, c_idx] > 0, vaf[:, c_idx], np.nan)
        lead_local  = int(np.nanargmax(np.nansum(obs_vaf, axis=0)))

        for j, mut_idx in enumerate(c_idx):
            obs  = DP[:, mut_idx] > 0
            x, y = tp[obs], vaf[obs, mut_idx] * 100.0
            if j == lead_local:
                ax.plot(x, y, "o-",  color=col, lw=2.0, ms=6, zorder=3)
            else:
                ax.plot(x, y, "s--", color=col, lw=1.2, ms=4,
                        alpha=0.45, zorder=2)

    ax.set_xlabel("Time (years)", fontsize=9)
    ax.set_ylabel("VAF (%)",      fontsize=9)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=legend_patches,
        title="Clone  (─ lead  ╌ sub)",
        title_fontsize=7, fontsize=7,
        framealpha=0.7, loc="upper left",
    )


def _draw_fitness_panel(ax, s_range, p_s, col, k, label, railed):
    p_s = _safe_norm(p_s)
    s_map, s_ci = _map_ci(s_range, p_s)

    ax.fill_between(s_range, p_s, alpha=0.25, color=col)
    ax.plot(s_range, p_s, color=col, lw=1.8)
    if np.isfinite(s_map):
        ax.axvline(s_map, color=col, ls="--", lw=1.5,
                   label=f"MAP = {s_map:.3f}")
        ax.axvspan(s_ci[0], s_ci[1], alpha=0.12, color=col,
                   label=f"90 % CI [{s_ci[0]:.2f}, {s_ci[1]:.2f}]")

    title = f"Fitness · clone {k}\n{label}"
    if railed:
        title += "\n⚠ railed (lower bound)"
    ax.set_title(title, fontsize=8,
                 color="darkred" if railed else "black")
    ax.set_xlabel("Selection coefficient  s", fontsize=8)
    ax.set_ylabel("Posterior density",        fontsize=8)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def _draw_zygosity_panel(ax, h_vals, p_h, col, k, label,
                          h_railed, h_unident):
    pinned = len(h_vals) == 1

    if pinned:
        ax.axvline(float(h_vals[0]), color=col, lw=3,
                   label=f"Fixed h = {float(h_vals[0]):.2f}")
    else:
        p_h = _safe_norm(p_h)
        h_map, h_ci = _map_ci(h_vals, p_h)
        ax.fill_between(h_vals, p_h, alpha=0.25, color=col)
        ax.plot(h_vals, p_h, color=col, lw=1.8)
        if np.isfinite(h_map):
            ax.axvline(h_map, color=col, ls="--", lw=1.5,
                       label=f"MAP = {h_map:.2f}")
            ax.axvspan(h_ci[0], h_ci[1], alpha=0.12, color=col,
                       label=f"90 % CI [{h_ci[0]:.2f}, {h_ci[1]:.2f}]")

    # reference lines at het / hom boundaries
    ax.axvline(0.0, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(1.0, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n(het)", "0.25", "0.5", "0.75", "1\n(hom)"],
                       fontsize=7)

    flags = ("  ⚠ railed"       if h_railed  else "") + \
            ("  ⚠ unidentified" if h_unident else "")
    title = f"Zygosity · clone {k}\n{label}{flags}"
    ax.set_title(title, fontsize=8,
                 color="darkred" if flags else "black")
    ax.set_xlabel("Homozygous fraction  h", fontsize=8)
    ax.set_ylabel("Posterior density",      fontsize=8)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def _draw_joint_panel(ax, s_range, h_vals, joint, col, k, label):
    """2-D joint posterior p(s, h) for one clone.
    `joint` is shaped (n_h, n_s) as returned by clone_posteriors."""
    joint = np.nan_to_num(np.asarray(joint, float),
                          nan=0.0, posinf=0.0, neginf=0.0)
    tot = joint.sum()
    if tot > 0:
        joint = joint / tot

    s_range = np.asarray(s_range, float)
    h_vals  = np.asarray(h_vals, float)
    pinned  = len(h_vals) == 1

    if pinned:
        # h fixed -> joint is a single row; show the conditional s-slice
        ax.plot(s_range, _safe_norm(joint[0]), color=col, lw=1.8)
        ax.fill_between(s_range, _safe_norm(joint[0]), alpha=0.25, color=col)
        ax.set_ylabel(f"p(s | h={h_vals[0]:.2f})", fontsize=8)
        ax.set_xlabel("Selection coefficient  s", fontsize=8)
        ax.set_title(f"Joint (h fixed) · clone {k}\n{label}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        return

    # pcolormesh handles non-uniform h grids robustly
    im = ax.pcolormesh(s_range, h_vals, joint, shading="auto", cmap="viridis")

    # overlay joint MAP
    hi, si = np.unravel_index(np.argmax(joint), joint.shape)
    ax.plot(s_range[si], h_vals[hi], "x", color="white", ms=9, mew=2.2,
            label=f"MAP (s={s_range[si]:.2f}, h={h_vals[hi]:.2f})")

    ax.set_xlabel("Selection coefficient  s", fontsize=8)
    ax.set_ylabel("Homozygous fraction  h",   fontsize=8)
    ax.set_ylim(h_vals.min(), h_vals.max())
    ax.set_title(f"Joint p(s, h) · clone {k}\n{label}", fontsize=8)
    ax.legend(fontsize=6, framealpha=0.6, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-participant figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_participant(part, pid):
    """Build the full summary figure for one participant."""
    model = part.uns["optimal_model"]
    cs    = model["clonal_structure"]
    ms    = model.get("mutation_structure", [[] for _ in cs])
    ms    = list(ms) + [[] for _ in range(len(cs) - len(ms))]  # guard mismatch
    K     = len(cs)

    try:
        posteriors, s_range = _get_posteriors(part)
    except Exception as exc:
        print(f"    [warn] clone_posteriors failed: {exc}")
        return None

    # ── Layout (now 4 columns) ────────────────────────────────────────────────
    n_rows = max(K, 1)
    fig = plt.figure(figsize=(FIG_W, max(4.5 * n_rows, 6.0)))
    gs  = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        width_ratios=[1.5, 1.0, 1.0, 1.25],   # 4th col = joint heatmap
        hspace=0.6, wspace=0.45,
    )

    ax_vaf  = fig.add_subplot(gs[:, 0])                              # spans rows
    axes_s  = [fig.add_subplot(gs[k, 1]) for k in range(n_rows)]
    axes_h  = [fig.add_subplot(gs[k, 2]) for k in range(n_rows)]
    axes_j  = [fig.add_subplot(gs[k, 3]) for k in range(n_rows)]     # NEW

    # ── VAF ────────────────────────────────────────────────────────────────────
    ax_vaf.set_title(f"Participant {pid}\nVAF over time",
                     fontsize=10, fontweight="bold")
    _draw_vaf_panel(ax_vaf, part, cs, ms, COLOURS)

    # ── Posterior panels per clone ─────────────────────────────────────────────
    for k in range(K):
        col = COLOURS[k % len(COLOURS)]

        h_vals, joint = posteriors[k]
        h_vals = np.asarray(h_vals, float)
        joint  = np.nan_to_num(np.asarray(joint, float),
                               nan=0.0, posinf=0.0, neginf=0.0)

        p_s = joint.sum(axis=0)   # marginalise over h  -> (n_s,)
        p_h = joint.sum(axis=1)   # marginalise over s  -> (n_h,)

        clone_label = " | ".join(ms[k]) if ms[k] else f"clone {k}"
        rep         = cs[k][0]   # representative mutation index for flags

        s_railed  = _obs_flag(part, "fitness_railed",            rep)
        h_railed  = _obs_flag(part, "homozygosity_railed",       rep)
        h_unident = _obs_flag(part, "homozygosity_unidentified", rep)

        _draw_fitness_panel(axes_s[k], s_range, p_s, col, k,
                            clone_label, s_railed)
        _draw_zygosity_panel(axes_h[k], h_vals, p_h, col, k,
                             clone_label, h_railed, h_unident)
        _draw_joint_panel(axes_j[k], s_range, h_vals, joint, col, k,
                          clone_label)                                # NEW

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading {INPUT_FILE} …")
    with open(INPUT_FILE, "rb") as f:
        cohort = pk.load(f)
    print(f"Loaded {len(cohort)} participants\n")

    n_ok = n_skip = n_err = 0

    for i, part in enumerate(cohort):
        pid = part.uns.get("participant_id", f"participant_{i+1}")
        print(f"[{i+1:>3}/{len(cohort)}]  {pid}", end="  ")

        if part.uns.get("fit_failed", False):
            print("SKIP — fit failed"); n_skip += 1; continue
        if "optimal_model" not in part.uns:
            print("SKIP — no optimal_model"); n_skip += 1; continue

        try:
            fig = plot_participant(part, pid)
            if fig is None:
                n_err += 1; continue
            out = os.path.join(OUTPUT_DIR, f"{pid}_clonal_summary.png")
            fig.savefig(out, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"→  {out}"); n_ok += 1
        except Exception as exc:
            import traceback
            print(f"ERROR — {exc}"); traceback.print_exc(); n_err += 1

    print(f"\nFinished.  saved={n_ok}  skipped={n_skip}  errors={n_err}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
