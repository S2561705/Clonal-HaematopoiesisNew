"""
Filtering Diagnostic
====================
Runs compute_clonal_models_prob_vec_mixed from each inference file as-is,
then compares the model probability rankings side by side.

Three conditions per participant:
  1. KI2  — VAF-vs-time correlation filter   (KI_clonal_inference_2)
  2. KI3  — Pairwise VAF correlation filter  (KI_clonal_inference_3)
  3. None — No filtering (filter_invalid=False, uses KI3)

Each figure:
  Left  : VAF trajectories with binomial ±1 SD shading
  Right : one bar chart per condition, top 5 models by posterior probability

Figures saved to OUTPUT_DIR, one per participant.
Thresholds are set at the top of this file and changed by hand.
"""

import sys
sys.path.append("..")

import copy
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import traceback

import src.KI_clonal_inference_2 as KI2
import src.KI_clonal_inference_3 as KI3

# ==============================================================================
# Configuration — edit thresholds here
# ==============================================================================

INPUT_FILE  = '../exports/MDS/MDS_cohort_processed.pk'
OUTPUT_DIR  = '../exports/MDS/figures/filtering_diagnostic/'
DPI         = 200
TOP_N       = 5

# Inference parameters — passed identically to all three conditions
INFERENCE_PARAMS = dict(
    s_resolution     = 50,
    min_s            = 0.01,
    max_s            = 3.0,
    resolution       = 600,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10})

MUT_PALETTE = sns.color_palette("tab10")

CONDITIONS = [
    {
        'label': 'KI2  (VAF-vs-time filter)',
        'color': '#4C72B0',
        'fn':    lambda part: KI2.compute_clonal_models_prob_vec_mixed(
                     part, filter_invalid=True, **INFERENCE_PARAMS),
    },
    {
        'label': 'KI3  (Pairwise VAF filter)',
        'color': '#CC6633',
        'fn':    lambda part: KI3.compute_clonal_models_prob_vec_mixed(
                     part, filter_invalid=True, **INFERENCE_PARAMS),
    },
    {
        'label': 'No filter',
        'color': '#6B6B6B',
        'fn':    lambda part: KI3.compute_clonal_models_prob_vec_mixed(
                     part, filter_invalid=False, **INFERENCE_PARAMS),
    },
]


# ==============================================================================
# Helpers
# ==============================================================================

def gene_name(full_name):
    return full_name.replace('_', ' ').split()[0]


def format_structure_gene(cs, obs_index):
    clone_strs = ['+'.join(gene_name(obs_index[j]) for j in clone) for clone in cs]
    return ' | '.join(clone_strs)


def extract_ranking(part):
    """
    Pull sorted (cs, norm_prob) list from part.uns['model_dict'].
    Handles both 2-tuple (cs, raw) and 3-tuple (cs, raw, norm) formats.
    """
    entries = []
    raw_vals = []

    for v in part.uns['model_dict'].values():
        if len(v) == 3:
            cs, raw, norm = v
        else:
            cs, raw = v
            norm = None
        entries.append({'cs': cs, 'raw': raw, 'norm': norm})
        raw_vals.append(raw)

    # If norms weren't stored, compute them now
    if any(e['norm'] is None for e in entries):
        total = sum(raw_vals)
        for e in entries:
            e['norm'] = e['raw'] / total if total > 0 else 0.0

    entries.sort(key=lambda x: x['norm'], reverse=True)
    return entries


# ==============================================================================
# VAF panel
# ==============================================================================

def plot_vaf_panel(ax, part, mut_colors):
    """
    One line per mutation, ±1 SD shading.
    SD(VAF) = sqrt(p * (1-p) / DP)
    """
    AO        = part.layers['AO']   # (n_mut, n_tp)
    DP        = part.layers['DP']
    tps       = part.var.time_points.values
    obs_index = list(part.obs.index)

    for i, mut_name in enumerate(obs_index):
        ao    = AO[i].astype(float)
        dp    = np.maximum(DP[i].astype(float), 1.0)
        p     = ao / dp
        sd    = np.sqrt(p * (1.0 - p) / dp)
        color = mut_colors[i]

        ax.plot(tps, p, color=color, linewidth=2.0,
                marker='o', markersize=5, zorder=3,
                label=gene_name(mut_name))
        ax.fill_between(tps,
                        np.clip(p - sd, 0, 1),
                        np.clip(p + sd, 0, 1),
                        color=color, alpha=0.18, zorder=2)

    ax.set_xlabel('Time point', fontweight='bold')
    ax.set_ylabel('VAF', fontweight='bold')
    ax.set_title('VAF trajectories  (±1 SD)', fontweight='bold', pad=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.legend(title='Mutation', fontsize=8, title_fontsize=8.5,
              framealpha=0.9, loc='best')
    sns.despine(ax=ax)


# ==============================================================================
# Bar chart for one condition
# ==============================================================================

def plot_condition_bars(ax, ranking, obs_index, color, title, x_max):
    ax.set_title(title, fontsize=8.5, color=color,
                 fontweight='bold', pad=7, linespacing=1.5)

    if not ranking:
        ax.text(0.5, 0.5, 'No models', ha='center', va='center',
                transform=ax.transAxes, fontsize=9,
                color='red', fontstyle='italic')
        ax.axis('off')
        return

    top      = ranking[:TOP_N]
    labels   = [format_structure_gene(m['cs'], obs_index) for m in top]
    norm_pct = [m['norm'] * 100 for m in top]

    # Reverse so winner sits at the top
    labels   = labels[::-1]
    norm_pct = norm_pct[::-1]

    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, norm_pct,
                   height=0.52,
                   color=color, alpha=0.72,
                   edgecolor='white', linewidth=0.5)

    # Outline winner
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(1.8)
    bars[-1].set_alpha(0.90)

    for bar, pct in zip(bars, norm_pct):
        ax.text(bar.get_width() + 0.7,
                bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%',
                va='center', ha='left', fontsize=7.5, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, linespacing=1.5)
    ax.set_xlim(0, x_max)
    ax.set_xlabel('Posterior probability (%)', fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.tick_params(axis='y', pad=5)
    sns.despine(ax=ax, left=False, bottom=False)


# ==============================================================================
# Per-participant figure
# ==============================================================================

def plot_participant(part, rankings, participant_id):
    obs_index  = list(part.obs.index)
    n_mut      = len(obs_index)
    mut_colors = [MUT_PALETTE[i % len(MUT_PALETTE)] for i in range(n_mut)]

    # Shared x-axis max across all conditions
    all_pcts = [m['norm'] * 100
                for r in rankings if r
                for m in r[:TOP_N]]
    x_max = max(all_pcts) * 1.28 + 3 if all_pcts else 105

    # Figure: VAF panel (left) + 3 bar charts (right, stacked vertically)
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(f'{participant_id}   —   Filtering condition comparison',
                 fontsize=13, fontweight='bold', y=1.002)

    outer = gridspec.GridSpec(1, 2,
                              width_ratios=[1, 2.2],
                              wspace=0.12,
                              figure=fig)

    # VAF panel
    ax_vaf = fig.add_subplot(outer[0])
    plot_vaf_panel(ax_vaf, part, mut_colors)

    # 3 stacked bar charts
    inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                                             subplot_spec=outer[1],
                                             hspace=0.65)

    for i, (cond, ranking) in enumerate(zip(CONDITIONS, rankings)):
        ax = fig.add_subplot(inner[i])

        n_models = len(ranking)
        winner   = f'{ranking[0]["norm"]*100:.0f}%' if ranking else '—'
        title    = f"{cond['label']}  ·  {n_models} models  ·  winner {winner}"

        plot_condition_bars(ax, ranking, obs_index,
                            cond['color'], title, x_max)

    plt.tight_layout()
    return fig


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("FILTERING DIAGNOSTIC")
    print("=" * 70)

    try:
        with open(INPUT_FILE, 'rb') as f:
            cohort = pk.load(f)
        print(f"Loaded {len(cohort)} participants from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found")
        return

    for p_idx, part in enumerate(cohort):
        pid   = part.uns.get('participant_id', f'participant_{p_idx}')
        n_mut = part.shape[0]
        print(f"\n{'='*70}")
        print(f"[{p_idx+1}/{len(cohort)}]  {pid}  ({n_mut} mutations)")

        rankings = []

        for cond in CONDITIONS:
            print(f"\n  ── {cond['label']} ──")
            try:
                part_copy = copy.deepcopy(part)
                part_copy = cond['fn'](part_copy)
                ranking   = extract_ranking(part_copy)
                rankings.append(ranking)

                print(f"     {len(ranking)} models. Top 3:")
                for rank, m in enumerate(ranking[:3]):
                    print(f"       {rank+1}. "
                          f"{format_structure_gene(m['cs'], list(part.obs.index))}"
                          f"  {m['norm']*100:.1f}%  (raw {m['raw']:.3e})")

            except Exception as e:
                print(f"     [ERROR] {e}")
                traceback.print_exc()
                rankings.append([])

        try:
            fig     = plot_participant(part, rankings, pid)
            outpath = os.path.join(OUTPUT_DIR, f"{pid}_filtering_diagnostic.png")
            fig.savefig(outpath, bbox_inches='tight', dpi=DPI)
            print(f"\n  Saved → {outpath}")
            plt.close(fig)
        except Exception as e:
            print(f"\n  [ERROR] Plot failed: {e}")
            traceback.print_exc()

    print(f"\nDone. All figures in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()