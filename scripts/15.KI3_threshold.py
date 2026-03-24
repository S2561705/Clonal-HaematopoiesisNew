"""
Pairwise VAF Threshold Sweep
=============================
For each participant, sweeps pairwise correlation threshold from 0.0 to 1.0
in steps of 0.1 and shows how model filtering and posterior rankings change.

Compared against KI2 (VAF-vs-time) as a fixed reference.

Figure layout per participant:
  Left         : VAF trajectories (±1 SD)
  Right top    : Number of surviving models vs threshold
                 KI2 model count shown as horizontal dashed reference line
  Right bottom : Top 3 model posteriors (%) vs threshold as separate lines
                 Vertical dashed lines mark:
                   - where pairwise first matches KI2's winner
                   - where pairwise first diverges from no-filter winner
                 KI2 winner posterior shown as horizontal dashed reference

Figures saved to OUTPUT_DIR, one per participant.
"""

import sys
sys.path.append("..")

import copy
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import seaborn as sns
import os
import traceback

import src.KI_clonal_inference_2 as KI2
import src.KI_clonal_inference_3 as KI3

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_FILE = '../exports/MDS/MDS_cohort_processed.pk'
OUTPUT_DIR = '../exports/MDS/figures/threshold_sweep/'
DPI        = 200

THRESHOLDS = np.round(np.arange(0.0, 1.05, 0.1), 2)   # 0.0 … 1.0

INFERENCE_PARAMS = dict(
    s_resolution = 50,
    min_s        = 0.01,
    max_s        = 3.0,
    resolution   = 600,
)

# Top N model lines to draw
TOP_N = 3

# Colour palette for top-N model lines (consistent across thresholds)
MODEL_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MUT_PALETTE   = sns.color_palette("tab10")

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10})


# ==============================================================================
# Helpers
# ==============================================================================

def gene_name(full_name):
    return full_name.replace('_', ' ').split()[0]


def format_structure_gene(cs, obs_index):
    return ' | '.join('+'.join(gene_name(obs_index[j]) for j in clone)
                      for clone in cs)


def extract_ranking(part):
    """Return list of {'cs', 'raw', 'norm'} sorted by norm desc."""
    entries, raw_vals = [], []
    for v in part.uns['model_dict'].values():
        cs, raw = v[0], v[1]
        entries.append({'cs': cs, 'raw': raw, 'norm': None})
        raw_vals.append(raw)
    total = sum(raw_vals)
    for e in entries:
        e['norm'] = e['raw'] / total if total > 0 else 0.0
    entries.sort(key=lambda x: x['norm'], reverse=True)
    return entries


def cs_label(cs, obs_index):
    return format_structure_gene(cs, obs_index)


# ==============================================================================
# Run KI2 (fixed reference)
# ==============================================================================

def run_ki2(part):
    p = copy.deepcopy(part)
    p = KI2.compute_clonal_models_prob_vec_mixed(
        p, filter_invalid=True, **INFERENCE_PARAMS)
    return extract_ranking(p)


# ==============================================================================
# Sweep pairwise thresholds
# ==============================================================================

def run_sweep(part):
    """
    For each threshold t in THRESHOLDS, run KI3 pairwise filter at t.
    Returns list of dicts, one per threshold:
        {
          'threshold'   : float,
          'n_models'    : int,
          'n_nonzero'   : int,
          'ranking'     : list of {'cs', 'raw', 'norm'},
          'total_mass'  : float,
        }
    """
    results = []
    for t in THRESHOLDS:
        print(f"    threshold={t:.1f} ...", end=' ', flush=True)
        try:
            p = copy.deepcopy(part)
            # Temporarily patch the threshold used inside KI3
            # KI3.compute_clonal_models_prob_vec_mixed accepts a
            # correlation_threshold kwarg that is forwarded to
            # find_valid_clonal_structures / compute_invalid_combinations
            p = KI3.compute_clonal_models_prob_vec_mixed(
                p,
                filter_invalid=True,
                correlation_threshold=float(t),
                **INFERENCE_PARAMS,
            )
            ranking     = extract_ranking(p)
            total_mass  = sum(r['raw'] for r in ranking)
            n_nonzero   = sum(1 for r in ranking if r['raw'] > 0)
            print(f"{len(ranking)} models, mass={total_mass:.3e}")
            results.append({
                'threshold' : float(t),
                'n_models'  : len(ranking),
                'n_nonzero' : n_nonzero,
                'ranking'   : ranking,
                'total_mass': total_mass,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            results.append({
                'threshold' : float(t),
                'n_models'  : 0,
                'n_nonzero' : 0,
                'ranking'   : [],
                'total_mass': 0.0,
            })
    return results


# ==============================================================================
# VAF panel
# ==============================================================================

def plot_vaf_panel(ax, part, mut_colors):
    AO        = part.layers['AO']
    DP        = part.layers['DP']
    tps       = part.var.time_points.values
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


# ==============================================================================
# Model count panel
# ==============================================================================

def plot_model_count(ax, sweep_results, ki2_n_models, ki2_n_nonzero):
    thresholds = [r['threshold']  for r in sweep_results]
    n_models   = [r['n_models']   for r in sweep_results]
    n_nonzero  = [r['n_nonzero']  for r in sweep_results]

    ax.plot(thresholds, n_models,
            color='#555555', linewidth=2.0, marker='o', markersize=5,
            label='All models (incl. prob=0)', zorder=3)
    ax.plot(thresholds, n_nonzero,
            color='#CC6633', linewidth=2.0, marker='s', markersize=5,
            linestyle='--', label='Non-zero models', zorder=3)

    # KI2 reference lines
    ax.axhline(ki2_n_models, color='#4C72B0', linewidth=1.4,
               linestyle=':', label=f'KI2 all models ({ki2_n_models})')
    ax.axhline(ki2_n_nonzero, color='#4C72B0', linewidth=1.4,
               linestyle='--', alpha=0.6,
               label=f'KI2 non-zero ({ki2_n_nonzero})')

    ax.set_xlabel('Pairwise correlation threshold', fontweight='bold')
    ax.set_ylabel('Number of models', fontweight='bold')
    ax.set_title('Surviving models vs threshold', fontweight='bold', pad=6)
    ax.set_xticks(thresholds)
    ax.legend(fontsize=7.5, loc='upper left')
    sns.despine(ax=ax)


# ==============================================================================
# Posterior ranking panel
# ==============================================================================

def plot_posterior_lines(ax, sweep_results, ki2_ranking, obs_index):
    """
    Top-N model posterior lines vs threshold.

    Model identity is tracked by structure label — a new model entering the
    top-N at a given threshold gets a new line; existing models are connected.
    Vertical lines mark:
      - first threshold where pairwise winner == KI2 winner
      - first threshold where pairwise winner != no-filter winner (t=0)
    KI2 winner posterior shown as horizontal dashed reference.
    """
    thresholds  = [r['threshold'] for r in sweep_results]
    nofilter    = sweep_results[0]   # t=0.0 == no filter

    nofilter_winner_label = (
        cs_label(nofilter['ranking'][0]['cs'], obs_index)
        if nofilter['ranking'] else None
    )
    ki2_winner_label = (
        cs_label(ki2_ranking[0]['cs'], obs_index)
        if ki2_ranking else None
    )
    ki2_winner_norm = ki2_ranking[0]['norm'] * 100 if ki2_ranking else None

    # Collect all unique top-N structures across thresholds (in order of appearance)
    seen_labels  = []
    label_to_idx = {}
    for r in sweep_results:
        for m in r['ranking'][:TOP_N]:
            lbl = cs_label(m['cs'], obs_index)
            if lbl not in label_to_idx:
                label_to_idx[lbl] = len(seen_labels)
                seen_labels.append(lbl)

    # Build per-structure series
    series = {lbl: [np.nan] * len(thresholds) for lbl in seen_labels}
    for t_i, r in enumerate(sweep_results):
        for m in r['ranking'][:TOP_N]:
            lbl = cs_label(m['cs'], obs_index)
            series[lbl][t_i] = m['norm'] * 100

    # Assign colours to structures — KI2 winner gets its own colour
    struct_colors = {}
    palette       = sns.color_palette("tab10", len(seen_labels))
    for i, lbl in enumerate(seen_labels):
        struct_colors[lbl] = palette[i]
    if ki2_winner_label and ki2_winner_label in struct_colors:
        struct_colors[ki2_winner_label] = '#4C72B0'

    # Plot lines
    for lbl, vals in series.items():
        color   = struct_colors[lbl]
        is_ki2  = (lbl == ki2_winner_label)
        lw      = 2.2 if is_ki2 else 1.6
        ls      = '-'
        zorder  = 4 if is_ki2 else 3
        short   = lbl if len(lbl) <= 32 else lbl[:31] + '…'
        ax.plot(thresholds, vals,
                color=color, linewidth=lw, linestyle=ls,
                marker='o', markersize=4,
                label=short, zorder=zorder)

    # KI2 winner horizontal reference
    if ki2_winner_norm is not None:
        ax.axhline(ki2_winner_norm,
                   color='#4C72B0', linewidth=1.2,
                   linestyle='--', alpha=0.55,
                   label=f'KI2 winner ({ki2_winner_norm:.0f}%)',
                   zorder=2)

    # Vertical: first threshold where pairwise winner matches KI2 winner
    match_t = None
    for r in sweep_results:
        if r['ranking'] and cs_label(r['ranking'][0]['cs'], obs_index) == ki2_winner_label:
            match_t = r['threshold']
            break
    if match_t is not None:
        ax.axvline(match_t, color='#4C72B0', linewidth=1.5,
                   linestyle=':', alpha=0.8,
                   label=f'Matches KI2 winner (t={match_t:.1f})')

    # Vertical: first threshold where pairwise winner diverges from no-filter
    diverge_t = None
    for r in sweep_results:
        if r['threshold'] == 0.0:
            continue
        if (r['ranking'] and nofilter_winner_label and
                cs_label(r['ranking'][0]['cs'], obs_index) != nofilter_winner_label):
            diverge_t = r['threshold']
            break
    if diverge_t is not None:
        ax.axvline(diverge_t, color='#CC6633', linewidth=1.5,
                   linestyle=':', alpha=0.8,
                   label=f'Diverges from no-filter (t={diverge_t:.1f})')

    ax.set_xlabel('Pairwise correlation threshold', fontweight='bold')
    ax.set_ylabel('Posterior probability (%)', fontweight='bold')
    ax.set_title(f'Top {TOP_N} model posteriors vs threshold',
                 fontweight='bold', pad=6)
    ax.set_xticks(thresholds)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper left',
              framealpha=0.9, ncol=1)
    sns.despine(ax=ax)


# ==============================================================================
# Per-participant figure
# ==============================================================================

def plot_participant(part, sweep_results, ki2_ranking, participant_id):
    obs_index  = list(part.obs.index)
    n_mut      = len(obs_index)
    mut_colors = [MUT_PALETTE[i % len(MUT_PALETTE)] for i in range(n_mut)]

    ki2_n_models  = len(ki2_ranking)
    ki2_n_nonzero = sum(1 for m in ki2_ranking if m['raw'] > 0)

    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(
        f'{participant_id}   —   Pairwise threshold sweep  '
        f'(KI2 reference: {ki2_n_models} models, '
        f'winner {ki2_ranking[0]["norm"]*100:.0f}%  '
        f'"{format_structure_gene(ki2_ranking[0]["cs"], obs_index)}")',
        fontsize=11, fontweight='bold', y=1.002
    )

    outer = gridspec.GridSpec(1, 2,
                              width_ratios=[1, 2.4],
                              wspace=0.12,
                              figure=fig)

    # VAF panel
    ax_vaf = fig.add_subplot(outer[0])
    plot_vaf_panel(ax_vaf, part, mut_colors)

    # Right: model count (top) + posterior lines (bottom)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[1],
        hspace=0.45,
    )

    ax_count = fig.add_subplot(inner[0])
    plot_model_count(ax_count, sweep_results, ki2_n_models, ki2_n_nonzero)

    ax_post = fig.add_subplot(inner[1])
    plot_posterior_lines(ax_post, sweep_results, ki2_ranking, obs_index)

    plt.tight_layout()
    return fig


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("PAIRWISE THRESHOLD SWEEP")
    print("=" * 70)
    print(f"Thresholds: {list(THRESHOLDS)}")

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

        # ── KI2 reference ────────────────────────────────────────────────────
        print("\n  KI2 reference ...")
        try:
            ki2_ranking = run_ki2(part)
            print(f"  → {len(ki2_ranking)} models, "
                  f"winner: {format_structure_gene(ki2_ranking[0]['cs'], list(part.obs.index))} "
                  f"({ki2_ranking[0]['norm']*100:.1f}%)")
        except Exception as e:
            print(f"  [ERROR] KI2 failed: {e}")
            traceback.print_exc()
            ki2_ranking = []

        # ── Threshold sweep ───────────────────────────────────────────────────
        print("\n  Pairwise threshold sweep:")
        sweep_results = run_sweep(part)

        # ── Summary table ─────────────────────────────────────────────────────
        obs_index = list(part.obs.index)

        # Collect all unique structure labels seen across all thresholds
        all_labels = []
        label_seen = set()
        for r in sweep_results:
            for m in r['ranking']:
                lbl = format_structure_gene(m['cs'], obs_index)
                if lbl not in label_seen:
                    all_labels.append(lbl)
                    label_seen.add(lbl)

        # Header
        col_w = 36
        header = f"  {'t':>5}  {'models':>6}  {'mass':>11}  "
        for lbl in all_labels:
            header += f"  {lbl[:col_w]:<{col_w}}"
        print(f"\n{header}")
        print("  " + "-" * (len(header) - 2))

        for r in sweep_results:
            lbl_map = {format_structure_gene(m['cs'], obs_index): (m['raw'], m['norm'])
                       for m in r['ranking']}
            row = f"  {r['threshold']:>5.1f}  {r['n_models']:>6}  {r['total_mass']:>11.3e}  "
            for lbl in all_labels:
                if lbl in lbl_map:
                    raw, norm = lbl_map[lbl]
                    cell = f"raw={raw:.2e} norm={norm*100:5.1f}%"
                else:
                    cell = "FILTERED"
                row += f"  {cell:<{col_w}}"
            print(row)

        # ── Plot ──────────────────────────────────────────────────────────────
        try:
            fig     = plot_participant(part, sweep_results, ki2_ranking, pid)
            outpath = os.path.join(OUTPUT_DIR, f"{pid}_threshold_sweep.png")
            fig.savefig(outpath, bbox_inches='tight', dpi=DPI)
            print(f"\n  Saved → {outpath}")
            plt.close(fig)
        except Exception as e:
            print(f"\n  [ERROR] Plot failed: {e}")
            traceback.print_exc()

    print(f"\nDone. Figures in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()