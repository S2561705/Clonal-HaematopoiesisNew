"""
plot_quantile_result.py
========================
Re-refine one participant with KI_3 (Fix-2 deterministic nodes) and render
the VAF / fitness / zygosity summary figure.
"""

import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from src.KI_3 import refine_optimal_model_posterior_vec, clone_posteriors

# ── Config ───────────────────────────────────────────────────────────────
TARGET_PID  = "MDS711P64"
FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
OUT_FILE    = f"../exports/figures/MDS/{TARGET_PID}_quantile_summary.png"
REFINE_S    = 40
REFINE_H    = 6

COLOURS   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DPI       = 150
FIG_W     = 16
N_SAMPLES = 5_000

# ── Load participant + known clonal structure ──────────────────────────
with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

part_template = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)
cs = list(part_template.uns["model_dict"].values())[0][0]

part = part_template.copy()
part.uns["model_dict"] = {"model_0": (cs, None)}
part = refine_optimal_model_posterior_vec(part, s_resolution=REFINE_S, h_resolution=REFINE_H)

# ── Same helper functions as plot_clonal_summary.py ─────────────────────

def _safe_norm(arr):
    arr = np.nan_to_num(np.asarray(arr, float), nan=0.0, posinf=0.0, neginf=0.0)
    s = arr.sum()
    return arr / s if s > 0 else arr


def _map_ci(vals, weights, lo=0.05, hi=0.95):
    weights = _safe_norm(weights)
    if weights.max() == 0:
        return float("nan"), (float("nan"), float("nan"))
    map_val = float(vals[np.argmax(weights)])
    draws = np.random.choice(vals, size=N_SAMPLES, p=weights)
    return map_val, (float(np.quantile(draws, lo)), float(np.quantile(draws, hi)))


def _vaf_dp_tp(part):
    AO = part.layers["AO"].T.astype(float)
    DP = part.layers["DP"].T.astype(float)
    vaf = AO / np.maximum(DP, 1.0)
    tp = np.array(part.var["time_points"], dtype=float)
    return vaf, DP, tp


def _get_posteriors(part):
    m = part.uns["optimal_model"]
    posts = clone_posteriors(
        np.array(m["posterior"]),
        np.array(m["h_combos"]),
        np.array(m["s_range"]),
        [np.array(g) for g in m["h_grids"]],
    )
    return posts, np.array(m["s_range"])


def _obs_flag(part, col, rep_idx, default=False):
    if col in part.obs.columns:
        return bool(part.obs[col].iloc[rep_idx])
    return default


def _draw_vaf_panel(ax, part, cs, ms, colours):
    vaf, DP, tp = _vaf_dp_tp(part)
    legend_patches = []
    for k, (c_idx, c_names) in enumerate(zip(cs, ms)):
        col = colours[k % len(colours)]
        label = " | ".join(c_names) if c_names else f"Clone {k}"
        legend_patches.append(mpatches.Patch(color=col, label=label))
        obs_vaf = np.where(DP[:, c_idx] > 0, vaf[:, c_idx], np.nan)
        lead_local = int(np.nanargmax(np.nansum(obs_vaf, axis=0)))
        for j, mut_idx in enumerate(c_idx):
            obs = DP[:, mut_idx] > 0
            x, y = tp[obs], vaf[obs, mut_idx] * 100.0
            if j == lead_local:
                ax.plot(x, y, "o-", color=col, lw=2.0, ms=6, zorder=3)
            else:
                ax.plot(x, y, "s--", color=col, lw=1.2, ms=4, alpha=0.45, zorder=2)
    ax.set_xlabel("Time (years)", fontsize=9)
    ax.set_ylabel("VAF (%)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_patches, title="Clone  (─ lead  ╌ sub)",
              title_fontsize=7, fontsize=7, framealpha=0.7, loc="upper left")


def _draw_fitness_panel(ax, s_range, p_s, col, k, label, railed):
    p_s = _safe_norm(p_s)
    s_map, s_ci = _map_ci(s_range, p_s)
    ax.fill_between(s_range, p_s, alpha=0.25, color=col)
    ax.plot(s_range, p_s, color=col, lw=1.8)
    if np.isfinite(s_map):
        ax.axvline(s_map, color=col, ls="--", lw=1.5, label=f"MAP = {s_map:.3f}")
        ax.axvspan(s_ci[0], s_ci[1], alpha=0.12, color=col,
                   label=f"90 % CI [{s_ci[0]:.2f}, {s_ci[1]:.2f}]")
    title = f"Fitness · clone {k}\n{label}"
    if railed:
        title += "\n⚠ railed (lower bound)"
    ax.set_title(title, fontsize=8, color="darkred" if railed else "black")
    ax.set_xlabel("Selection coefficient  s", fontsize=8)
    ax.set_ylabel("Posterior density", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def _draw_zygosity_panel(ax, h_vals, p_h, col, k, label, h_railed, h_unident):
    pinned = len(h_vals) == 1
    if pinned:
        ax.axvline(float(h_vals[0]), color=col, lw=3, label=f"Fixed h = {float(h_vals[0]):.2f}")
    else:
        p_h = _safe_norm(p_h)
        h_map, h_ci = _map_ci(h_vals, p_h)
        ax.fill_between(h_vals, p_h, alpha=0.25, color=col)
        ax.plot(h_vals, p_h, color=col, lw=1.8)
        if np.isfinite(h_map):
            ax.axvline(h_map, color=col, ls="--", lw=1.5, label=f"MAP = {h_map:.2f}")
            ax.axvspan(h_ci[0], h_ci[1], alpha=0.12, color=col,
                       label=f"90 % CI [{h_ci[0]:.2f}, {h_ci[1]:.2f}]")
    ax.axvline(0.0, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(1.0, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n(het)", "0.25", "0.5", "0.75", "1\n(hom)"], fontsize=7)
    flags = ("  ⚠ railed" if h_railed else "") + ("  ⚠ unidentified" if h_unident else "")
    title = f"Zygosity · clone {k}\n{label}{flags}"
    ax.set_title(title, fontsize=8, color="darkred" if flags else "black")
    ax.set_xlabel("Homozygous fraction  h", fontsize=8)
    ax.set_ylabel("Posterior density", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


# ── Build figure ─────────────────────────────────────────────────────────
model = part.uns["optimal_model"]
ms = model.get("mutation_structure", [[] for _ in cs])
ms = list(ms) + [[] for _ in range(len(cs) - len(ms))]
K = len(cs)

posteriors, s_range = _get_posteriors(part)

n_rows = max(K, 1)
fig = plt.figure(figsize=(FIG_W, max(4.5 * n_rows, 6.0)))
gs = gridspec.GridSpec(n_rows, 3, figure=fig, width_ratios=[1.6, 1.0, 1.0],
                       hspace=0.6, wspace=0.4)

ax_vaf = fig.add_subplot(gs[:, 0])
axes_s = [fig.add_subplot(gs[k, 1]) for k in range(n_rows)]
axes_h = [fig.add_subplot(gs[k, 2]) for k in range(n_rows)]

ax_vaf.set_title(f"Participant {TARGET_PID} (quantile method)\nVAF over time",
                 fontsize=10, fontweight="bold")
_draw_vaf_panel(ax_vaf, part, cs, ms, COLOURS)

for k in range(K):
    col = COLOURS[k % len(COLOURS)]
    h_vals, joint = posteriors[k]
    h_vals = np.asarray(h_vals, float)
    joint = np.nan_to_num(np.asarray(joint, float), nan=0.0, posinf=0.0, neginf=0.0)
    p_s = joint.sum(axis=0)
    p_h = joint.sum(axis=1)
    clone_label = " | ".join(ms[k]) if ms[k] else f"clone {k}"
    rep = cs[k][0]
    s_railed = _obs_flag(part, "fitness_railed", rep)
    h_railed = _obs_flag(part, "homozygosity_railed", rep)
    h_unident = _obs_flag(part, "homozygosity_unidentified", rep)
    _draw_fitness_panel(axes_s[k], s_range, p_s, col, k, clone_label, s_railed)
    _draw_zygosity_panel(axes_h[k], h_vals, p_h, col, k, clone_label, h_railed, h_unident)

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight")
print(f"Saved -> {OUT_FILE}")