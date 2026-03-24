"""
Sub-mutation fitness results explorer
======================================
Loads MDS_cohort_fitted_sub.pk and produces:

  1. Cohort-level summary DataFrame (CSV)
  2. Per-participant figure:
       Left  : VAF trajectories (±1 SD)
       Right : sub-mutation fitness panel showing s_sub and s_rel per mutation,
               colour-coded by role (leading / sub) with 90% CI bars

Outputs saved to OUTPUT_DIR.
"""

import sys
sys.path.append("..")

import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_FILE = '../exports/MDS/MDS_cohort_fitted_sub.pk'
OUTPUT_DIR = '../exports/MDS/figures/sub_mutation/'
CSV_FILE   = '../exports/MDS/sub_mutation_fitness_cohort.csv'
DPI        = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10})

LEAD_COLOR = '#4C72B0'
SUB_COLOR  = '#CC6633'


# ==============================================================================
# Helpers
# ==============================================================================

def gene_name(full_name):
    return full_name.replace('_', ' ').split()[0]


# ==============================================================================
# 1. Build cohort DataFrame
# ==============================================================================

def build_cohort_df(cohort):
    rows = []
    for part in cohort:
        pid = part.uns.get('participant_id', 'unknown')

        if 'sub_mutation_inference' not in part.uns:
            print(f"  [SKIP] {pid} — no sub_mutation_inference found")
            continue

        for mut, res in part.uns['sub_mutation_inference'].items():
            rows.append({
                'participant': pid,
                'mutation':    mut,
                'gene':        gene_name(mut),
                'clone_idx':   res['clone_idx'],
                'role':        res['role'],
                's_sub':       res['s_sub'],
                's_sub_ci_low':  res['ci_90'][0],
                's_sub_ci_high': res['ci_90'][1],
                's_rel':       res['s_rel'],
                's_lead':      res.get('s_lead', res['s_sub']),  # lead mutation s_lead = s_sub
            })

    df = pd.DataFrame(rows)
    return df


# ==============================================================================
# 2. Per-participant figure
# ==============================================================================

def plot_vaf_panel(ax, part, mut_colors):
    AO        = part.layers['AO']
    DP        = part.layers['DP']
    tps       = part.var.time_points.values.astype(float)
    obs_index = list(part.obs.index)

    for i, mut_name in enumerate(obs_index):
        ao = AO[i].astype(float)
        dp = np.maximum(DP[i].astype(float), 1.0)
        p  = ao / dp
        sd = np.sqrt(p * (1.0 - p) / dp)

        ax.plot(tps, p, color=mut_colors[i], linewidth=2.0,
                marker='o', markersize=5, zorder=3,
                label=gene_name(mut_name))
        ax.fill_between(tps,
                        np.clip(p - sd, 0, 1),
                        np.clip(p + sd, 0, 1),
                        color=mut_colors[i], alpha=0.18, zorder=2)

    ax.set_xlabel('Time point', fontweight='bold')
    ax.set_ylabel('VAF', fontweight='bold')
    ax.set_title('VAF trajectories  (±1 SD)', fontweight='bold', pad=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.legend(title='Mutation', fontsize=8, title_fontsize=8.5,
              framealpha=0.9, loc='best')
    sns.despine(ax=ax)


def plot_sub_fitness_panel(ax_sub, ax_rel, part, mut_colors):
    """
    Two vertically stacked panels:
      Top    : s_sub per mutation with 90% CI
      Bottom : s_rel per mutation with 90% CI, reference line at 0
    """
    obs_index = list(part.obs.index)
    sub_inf   = part.uns['sub_mutation_inference']

    labels    = []
    s_subs    = []
    ci_lows   = []
    ci_highs  = []
    s_rels    = []
    colors    = []
    roles     = []

    for i, mut_name in enumerate(obs_index):
        if mut_name not in sub_inf:
            continue
        res = sub_inf[mut_name]
        labels.append(gene_name(mut_name))
        s_subs.append(res['s_sub'])
        ci_lows.append(res['ci_90'][0])
        ci_highs.append(res['ci_90'][1])
        s_rels.append(res['s_rel'])
        colors.append(LEAD_COLOR if res['role'] == 'leading' else SUB_COLOR)
        roles.append(res['role'])

    if not labels:
        ax_sub.text(0.5, 0.5, 'No results', ha='center', va='center',
                    transform=ax_sub.transAxes, fontsize=9, color='red')
        ax_rel.axis('off')
        return

    y_pos = np.arange(len(labels))
    xerr_low  = np.array(s_subs) - np.array(ci_lows)
    xerr_high = np.array(ci_highs) - np.array(s_subs)

    # ── s_sub panel ───────────────────────────────────────────────────────────
    ax_sub.barh(y_pos, s_subs, height=0.5,
                color=colors, alpha=0.75, edgecolor='white')
    ax_sub.errorbar(s_subs, y_pos,
                    xerr=[xerr_low, xerr_high],
                    fmt='none', color='black', linewidth=1.2,
                    capsize=3, zorder=5)

    ax_sub.set_yticks(y_pos)
    ax_sub.set_yticklabels(labels, fontsize=8.5)
    ax_sub.set_xlabel('s_sub', fontweight='bold')
    ax_sub.set_title('Sub-mutation fitness  (s_sub)', fontweight='bold', pad=6)
    ax_sub.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax_sub.tick_params(axis='y', pad=4)

    # Legend
    from matplotlib.patches import Patch
    ax_sub.legend(handles=[
        Patch(color=LEAD_COLOR, label='Leading'),
        Patch(color=SUB_COLOR,  label='Sub-mutation'),
    ], fontsize=8, loc='lower right')
    sns.despine(ax=ax_sub, left=False, bottom=False)

    # ── s_rel panel ───────────────────────────────────────────────────────────
    rel_colors = [
        '#CC6633' if r > 0.05 else ('#4C72B0' if r < -0.05 else '#888888')
        for r in s_rels
    ]

    ax_rel.barh(y_pos, s_rels, height=0.5,
                color=rel_colors, alpha=0.75, edgecolor='white')
    ax_rel.axvline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.7,
                   zorder=5)

    for i, (r, role) in enumerate(zip(s_rels, roles)):
        if role == 'leading':
            ax_rel.text(0.01, y_pos[i], 'ref',
                        va='center', ha='left', fontsize=7,
                        color='grey', fontstyle='italic')

    ax_rel.set_yticks(y_pos)
    ax_rel.set_yticklabels(labels, fontsize=8.5)
    ax_rel.set_xlabel('s_rel  (s_sub − s_lead)', fontweight='bold')
    ax_rel.set_title('Relative fitness within clone', fontweight='bold', pad=6)
    ax_rel.tick_params(axis='y', pad=4)

    # Direction legend
    from matplotlib.patches import Patch
    ax_rel.legend(handles=[
        Patch(color='#CC6633', label='Faster than leader  (s_rel > 0)'),
        Patch(color='#4C72B0', label='Slower than leader  (s_rel < 0)'),
        Patch(color='#888888', label='Similar  (|s_rel| ≤ 0.05)'),
    ], fontsize=7.5, loc='lower right')
    sns.despine(ax=ax_rel, left=False, bottom=False)


def plot_participant(part):
    pid       = part.uns.get('participant_id', 'unknown')
    n_mut     = part.shape[0]
    palette   = sns.color_palette("tab10")
    mut_colors = [palette[i % len(palette)] for i in range(n_mut)]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'{pid}   —   Sub-mutation fitness inference',
                 fontsize=13, fontweight='bold', y=1.002)

    outer = gridspec.GridSpec(1, 2,
                              width_ratios=[1, 2.2],
                              wspace=0.14, figure=fig)

    # VAF panel
    ax_vaf = fig.add_subplot(outer[0])
    plot_vaf_panel(ax_vaf, part, mut_colors)

    # Sub-fitness panels (stacked)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                             subplot_spec=outer[1],
                                             hspace=0.5)
    ax_sub = fig.add_subplot(inner[0])
    ax_rel = fig.add_subplot(inner[1])
    plot_sub_fitness_panel(ax_sub, ax_rel, part, mut_colors)

    plt.tight_layout()
    return fig


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("SUB-MUTATION FITNESS RESULTS EXPLORER")
    print("=" * 70)

    try:
        with open(INPUT_FILE, 'rb') as f:
            cohort = pk.load(f)
        print(f"Loaded {len(cohort)} participants from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found")
        return

    # ── 1. Cohort DataFrame ───────────────────────────────────────────────────
    print("\nBuilding cohort summary DataFrame...")
    df = build_cohort_df(cohort)

    if df.empty:
        print("No sub-mutation inference results found in cohort.")
        return

    df.to_csv(CSV_FILE, index=False)
    print(f"Saved cohort CSV → {CSV_FILE}")
    print(f"\n{df.to_string(index=False)}")

    # ── 2. Per-participant figures ─────────────────────────────────────────────
    print("\nGenerating per-participant figures...")
    for part in cohort:
        pid = part.uns.get('participant_id', 'unknown')

        if 'sub_mutation_inference' not in part.uns:
            print(f"  [SKIP] {pid}")
            continue

        try:
            fig     = plot_participant(part)
            outpath = os.path.join(OUTPUT_DIR, f"{pid}_sub_mutation_fitness.png")
            fig.savefig(outpath, bbox_inches='tight', dpi=DPI)
            print(f"  Saved → {outpath}")
            plt.close(fig)
        except Exception as e:
            print(f"  [ERROR] {pid}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone.")
    print(f"  Figures : {OUTPUT_DIR}")
    print(f"  CSV     : {CSV_FILE}")


if __name__ == '__main__':
    main()