import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pandas as pd
import os

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = '../exports/MDS/MDS_cohort_fitted.pk'
OUTPUT_DIR  = '../exports/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_mutation_table(part, model, cs, ms, joint_inf):
    """
    Extract mutation info: gene, cdna, fitness (CI), h (CI), role (leading/sub)
    Returns: pandas DataFrame sorted by gene, with leading mutations first per clone
    """
    rows = []
    
    # Get per-mutation info from part.obs if available
    fitness_map = dict(zip(part.obs.index, part.obs['fitness'])) if 'fitness' in part.obs.columns else {}
    fitness_5_map = dict(zip(part.obs.index, part.obs['fitness_5'])) if 'fitness_5' in part.obs.columns else {}
    fitness_95_map = dict(zip(part.obs.index, part.obs['fitness_95'])) if 'fitness_95' in part.obs.columns else {}
    h_map = dict(zip(part.obs.index, part.obs['h'])) if 'h' in part.obs.columns else {}
    h_5_map = dict(zip(part.obs.index, part.obs['h_5'])) if 'h_5' in part.obs.columns else {}
    h_95_map = dict(zip(part.obs.index, part.obs['h_95'])) if 'h_95' in part.obs.columns else {}
    
    for clone_idx, (clone_muts, result) in enumerate(zip(cs, joint_inf)):
        # Find leading mutation (highest summed VAF)
        AO = part.layers['AO'].T
        DP = part.layers['DP'].T
        vaf_ratio = AO / np.maximum(DP, 1.0)
        clone_vafs = vaf_ratio[:, clone_muts]
        vaf_sums = clone_vafs.sum(axis=0)
        lead_idx_in_clone = int(np.argmax(vaf_sums))
        lead_mut = clone_muts[lead_idx_in_clone]
        
        # Get clone-level estimates from result
        s_map = result['s_map']
        s_ci = result['s_ci']
        h_map_clone = result['h_map']
        h_ci = result['h_ci']
        
        for mut_idx_in_clone, mut in enumerate(clone_muts):
            mut_name = ms[clone_idx][mut_idx_in_clone] if ms else str(mut)
            
            # Parse gene and cDNA change
            # Expected format: "GENE c.123A>G" or "GENE p.AminoAcidChange"
            parts = mut_name.split()
            if len(parts) >= 2:
                gene = parts[0]
                cdna = ' '.join(parts[1:])  # Rest is the change
            else:
                gene = mut_name
                cdna = '-'
            
            # Determine role
            is_leading = (mut == lead_mut)
            role = 'Leading' if is_leading else 'Sub-clone'
            
            # Get estimates: use per-mutation if available, else clone-level
            if mut in fitness_map and fitness_map[mut] > 0:
                fit = fitness_map[mut]
                fit_5 = fitness_5_map.get(mut, s_ci[0])
                fit_95 = fitness_95_map.get(mut, s_ci[1])
            else:
                fit = s_map
                fit_5 = s_ci[0]
                fit_95 = s_ci[1]
            
            if mut in h_map:
                h_val = h_map[mut]
                h_5 = h_5_map.get(mut, h_ci[0])
                h_95 = h_95_map.get(mut, h_ci[1])
            else:
                h_val = h_map_clone
                h_5 = h_ci[0]
                h_95 = h_ci[1]
            
            rows.append({
                'Gene': gene,
                'cDNA_Change': cdna,
                'Clone': clone_idx,
                'Role': role,
                'Fitness': fit,
                'Fitness_CI': f"[{fit_5:.2f}, {fit_95:.2f}]",
                'Fitness_5': fit_5,
                'Fitness_95': fit_95,
                'h': h_val,
                'h_CI': f"[{h_5:.2f}, {h_95:.2f}]",
                'h_5': h_5,
                'h_95': h_95,
                'Sort_Key': (gene, 0 if is_leading else 1, mut_name)  # Leading first per gene
            })
    
    df = pd.DataFrame(rows)
    # Sort by gene, then leading first
    df = df.sort_values('Sort_Key').reset_index(drop=True)
    return df[['Gene', 'cDNA_Change', 'Role', 'Fitness', 'Fitness_CI', 'h', 'h_CI', 'Clone']]


def plot_mutation_table(ax, df, colours):
    """
    Plot a clean table of mutations on the given axis.
    """
    ax.axis('off')
    
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No mutation data', ha='center', va='center', fontsize=10)
        return
    
    # Prepare display data
    display_df = df[['Gene', 'cDNA_Change', 'Role', 'Fitness_CI', 'h_CI']].copy()
    display_df.columns = ['Gene', 'cDNA Change', 'Role', 'Fitness (95% CI)', 'h (95% CI)']
    
    # Create table
    table_data = [display_df.columns.tolist()] + display_df.values.tolist()
    
    # Color mapping: highlight rows by clone
    cell_colors = []
    for i, row in enumerate(table_data):
        if i == 0:  # Header
            cell_colors.append(['#4472C4'] * len(row))  # Blue header
        else:
            clone_idx = df.iloc[i-1]['Clone']
            base_color = colours[clone_idx % len(colours)]
            # Lighten the color for background
            from matplotlib.colors import to_rgb, to_hex
            rgb = to_rgb(base_color)
            light_rgb = tuple(min(1.0, c + 0.7) for c in rgb)
            cell_colors.append([to_hex(light_rgb)] * len(row))
    
    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        cellColours=cell_colors,
        colWidths=[0.15, 0.35, 0.15, 0.20, 0.15]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Header styling
    for j in range(len(display_df.columns)):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.08)
    
    # Row styling
    for i in range(1, len(table_data)):
        for j in range(len(display_df.columns)):
            cell = table[(i, j)]
            cell.set_height(0.06)
            # Bold for leading mutations
            if df.iloc[i-1]['Role'] == 'Leading' and j in [0, 2]:  # Gene and Role columns
                cell.set_text_props(fontweight='bold')
    
    ax.set_title('Mutation Fitness and Zygosity Estimates', fontsize=10, fontweight='bold', pad=20)


# ── Plotting function ──────────────────────────────────────────────────────────
def plot_optimal_model_full(part, participant_id=None, figsize=(16, 5)):
    """
    Layout:
      Left column (spans all clone rows) — all clones' VAFs on one shared axis
      Middle columns (one row per clone) — fitness (s) and zygosity (h) posteriors
      Right column (bottom) — mutation table
    """

    if part.uns.get('warning') is not None:
        print(f'  WARNING: {part.uns["warning"]}')

    model       = part.uns['optimal_model']
    cs          = model['clonal_structure']
    ms          = model['mutation_structure']
    joint_inf   = model['joint_inference']
    s_range     = model['s_range']
    h_posterior = model['h_posterior']

    AO          = part.layers['AO'].T          # (n_tps, n_muts)
    DP          = part.layers['DP'].T
    vaf_ratio   = AO / np.maximum(DP, 1.0)
    time_points = np.array(part.var.time_points)

    n_clones  = len(cs)
    n_rows    = max(n_clones, 1)
    
    # Calculate figure height based on clones and table
    fig_h = max(figsize[1] * n_rows, 6)
    if len(part.obs) > 5:  # More mutations = taller for table
        fig_h += 2

    # GridSpec: 4 columns - VAF, s-posterior, h-posterior, table (spans bottom)
    fig = plt.figure(figsize=(figsize[0], fig_h))
    
    # Create grid: rows for clones, 4 columns
    gs = gridspec.GridSpec(n_rows + 1, 4, figure=fig, 
                          height_ratios=[1]*n_rows + [0.8],  # Extra row for table
                          width_ratios=[1.4, 1, 1, 1.2])

    ax_vaf = fig.add_subplot(gs[0:n_rows, 0])   # VAF spans all clone rows
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    title_str = f'Participant {participant_id}' if participant_id else 'Participant'
    fig.suptitle(title_str, fontsize=11, fontweight='bold', y=0.98)

    # ── Shared VAF panel ──────────────────────────────────────────────────
    for clone_idx, (clone_muts, result) in enumerate(zip(cs, joint_inf)):
        col         = colours[clone_idx % len(colours)]
        clone_label = ', '.join(ms[clone_idx]) if ms else f'Clone {clone_idx}'

        clone_vafs = vaf_ratio[:, clone_muts]
        lead_idx   = int(np.argmax(clone_vafs.sum(axis=0)))
        lead_mut   = clone_muts[lead_idx]

        # Leading mutation — solid, labelled
        valid = DP[:, lead_mut] > 0
        ax_vaf.plot(
            time_points[valid], vaf_ratio[valid, lead_mut] * 100,
            'o-', color=col, lw=1.8, ms=5,
            label=clone_label
        )

        # Sub-mutations within clone — dashed, same colour
        for m in clone_muts:
            if m == lead_mut:
                continue
            ok = DP[:, m] > 0
            ax_vaf.plot(
                time_points[ok], vaf_ratio[ok, m] * 100,
                's--', color=col, alpha=0.45, lw=1, ms=3
            )

    ax_vaf.set_title('VAF over time', fontsize=9)
    ax_vaf.set_xlabel('Time (years)', fontsize=8)
    ax_vaf.set_ylabel('VAF (%)', fontsize=8)
    ax_vaf.set_ylim(0, 100)
    ax_vaf.tick_params(labelsize=7)
    ax_vaf.grid(True, alpha=0.3)
    ax_vaf.legend(fontsize=7, framealpha=0.6, title='Clone', title_fontsize=7, 
                  loc='upper left', bbox_to_anchor=(1.02, 1))

    # ── Per-clone posterior panels ────────────────────────────────────────
    for clone_idx, (clone_muts, result) in enumerate(zip(cs, joint_inf)):
        col = colours[clone_idx % len(colours)]

        ax_s = fig.add_subplot(gs[clone_idx, 1])
        ax_h = fig.add_subplot(gs[clone_idx, 2])

        # Fitness (s) posterior
        s_post = result['s_posterior']
        s_map  = result['s_map']
        s_ci   = result['s_ci']

        ax_s.fill_between(s_range, s_post, alpha=0.25, color=col)
        ax_s.plot(s_range, s_post, color=col, lw=1.5)
        ax_s.axvline(s_map, color=col, lw=1.5, ls='--',
                     label=f'MAP = {s_map:.3f}')
        ax_s.axvspan(s_ci[0], s_ci[1], alpha=0.12, color=col,
                     label=f'90% CI [{s_ci[0]:.2f}, {s_ci[1]:.2f}]')
        ax_s.set_title(f'Fitness — clone {clone_idx}', fontsize=9)
        ax_s.set_xlabel('Selection coefficient s', fontsize=8)
        ax_s.set_ylabel('Posterior density', fontsize=8)
        ax_s.legend(fontsize=7, framealpha=0.6)
        ax_s.tick_params(labelsize=7)
        ax_s.grid(True, alpha=0.3)

        # Zygosity (h) posterior
        h_range_arr = result['h_range']
        h_post      = h_posterior[clone_idx]
        h_map       = result['h_map']
        h_ci        = result['h_ci']

        ax_h.fill_between(h_range_arr, h_post, alpha=0.25, color=col)
        ax_h.plot(h_range_arr, h_post, color=col, lw=1.5)
        ax_h.axvline(h_map, color=col, lw=1.5, ls='--',
                     label=f'MAP = {h_map:.2f}')
        ax_h.axvspan(h_ci[0], h_ci[1], alpha=0.12, color=col,
                     label=f'90% CI [{h_ci[0]:.2f}, {h_ci[1]:.2f}]')
        ax_h.axvline(0.0, color='grey', lw=0.8, ls=':', alpha=0.7)
        ax_h.axvline(1.0, color='grey', lw=0.8, ls=':', alpha=0.7)
        ax_h.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_h.set_xticklabels(['0\\n(het)', '0.25', '0.5', '0.75', '1\\n(hom)'],
                              fontsize=7)
        ax_h.set_title(f'Zygosity — clone {clone_idx}', fontsize=9)
        ax_h.set_xlabel('Homozygous fraction h', fontsize=8)
        ax_h.set_ylabel('Posterior density', fontsize=8)
        ax_h.legend(fontsize=7, framealpha=0.6)
        ax_h.tick_params(labelsize=7)
        ax_h.grid(True, alpha=0.3)
    
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading fitted cohort from {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'rb') as f:
            cohort = pk.load(f)
        print(f"Loaded {len(cohort)} participants")
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found. Run the inference pipeline first.")
        return

    for i, part in enumerate(cohort):
        participant_id = part.uns.get('participant_id', f'participant_{i+1}')
        print(f"\\nPlotting {participant_id}...")

        try:
            fig = plot_optimal_model_full(part, participant_id=participant_id)
            out_path = os.path.join(OUTPUT_DIR, f'{participant_id}_posteriors.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  ERROR plotting {participant_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\\nDone. Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
