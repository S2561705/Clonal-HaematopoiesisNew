"""
Standalone plotting script for MDS clonal evolution results.
Generates comprehensive single-page summaries for each participant.
UPDATED: Now shows actual h MAP and CI values in the table.
"""

import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_FILE = '../exports/MDS/MDS_cohort_fitted_unified.pk'
OUTPUT_DIR = '../exports/MDS/figures/'
SHOW_PLOTS = False  # Set to True to display plots interactively
DPI = 300

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['savefig.dpi'] = DPI
plt.rcParams['font.size'] = 10

# ==============================================================================
# Plotting Function
# ==============================================================================

def plot_participant_comprehensive(part, figsize=(14, 8)):
    """
    Simplified comprehensive plot showing:
    - VAF trajectories over time with mutation labels
    - Clonal structure
    - Fitness estimates with zygosity MAP and CI
    """
    if 'optimal_model' not in part.uns:
        print(f"  No optimal model found")
        return None
    
    participant_id = part.uns.get('participant_id', 'Unknown')
    model = part.uns['optimal_model']
    cs = model['clonal_structure']
    h_vec = model['h_vec']
    
    AO = part.layers['AO']
    DP = part.layers['DP']
    VAF = AO / np.maximum(DP, 1)
    time_points = part.var.time_points.values
    
    n_clones = len(cs)
    colors = sns.color_palette("husl", n_clones)
    
    # Create figure with simple 2-panel layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1], 
                  hspace=0.3, wspace=0.3)
    
    # ==========================================================================
    # 1. VAF Trajectories (main plot, top left)
    # ==========================================================================
    ax_vaf = fig.add_subplot(gs[0, 0])
    
    for i, (clone_muts, h, color) in enumerate(zip(cs, h_vec, colors)):
        for mut_idx in clone_muts:
            vaf = VAF[mut_idx, :]
            mut_name = part.obs.index[mut_idx]
            
            # Calculate binomial uncertainty
            depth = DP[mut_idx, :]
            vaf_std = np.sqrt(vaf * (1 - vaf) / np.maximum(depth, 1))
            
            # Plot shaded uncertainty region (95% CI: 1.96 * std)
            ax_vaf.fill_between(time_points, 
                               vaf - 1.96*vaf_std, 
                               vaf + 1.96*vaf_std,
                               color=color, alpha=0.15, linewidth=0)
            
            # Plot line on top of shaded region
            line, = ax_vaf.plot(time_points, vaf, 'o-', color=color, 
                               markersize=10, linewidth=3, alpha=0.9, zorder=10)
            
            # Add error bars for clarity at each point
            ax_vaf.errorbar(time_points, vaf, yerr=1.96*vaf_std, 
                           fmt='none', color=color, alpha=0.4, capsize=5, 
                           linewidth=2, capthick=2, zorder=9)
            
            # Add mutation name label at the end of the line
            final_time = time_points[-1]
            final_vaf = vaf[-1]
            ax_vaf.text(final_time + 0.1, final_vaf, mut_name, 
                       fontsize=10, va='center', color=color, fontweight='bold')
    
    # Reference line
    ax_vaf.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)
    ax_vaf.text(time_points[0], 0.51, 'Het max', fontsize=9, color='gray', va='bottom')
    
    ax_vaf.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax_vaf.set_ylabel('Variant Allele Frequency (VAF)', fontsize=14, fontweight='bold')
    ax_vaf.set_title(f'{participant_id}: Mutation Trajectories', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_vaf.grid(True, alpha=0.3)
    ax_vaf.set_ylim([-0.05, 1.05])
    
    # Extend x-axis slightly to fit labels
    x_range = time_points.max() - time_points.min()
    ax_vaf.set_xlim([time_points.min() - 0.1*x_range, 
                     time_points.max() + 0.3*x_range])
    
    # ==========================================================================
    # 2. Clonal Structure (top right)
    # ==========================================================================
    ax_structure = fig.add_subplot(gs[0, 1])
    ax_structure.axis('off')
    
    y_pos = 0.95
    ax_structure.text(0.5, y_pos, 'Clonal Structure', 
                     ha='center', va='top', fontsize=14, fontweight='bold',
                     transform=ax_structure.transAxes)
    y_pos -= 0.12
    
    for i, (clone_muts, h, color) in enumerate(zip(cs, h_vec, colors)):
        mut_names = [part.obs.index[j] for j in clone_muts]
        
        # Zygosity interpretation
        if h < 0.1:
            zyg_str = 'Heterozygous'
        elif h > 0.9:
            zyg_str = 'Homozygous'
        else:
            zyg_str = f'Mixed (h={h:.2f})'
        
        # Box height depends on number of mutations
        box_height = min(0.15, 0.04 + 0.02 * len(mut_names))
        
        # Draw box
        box = plt.Rectangle((0.05, y_pos - box_height), 0.9, box_height, 
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=3,
                           transform=ax_structure.transAxes)
        ax_structure.add_patch(box)
        
        # Clone label
        ax_structure.text(0.1, y_pos - 0.02, f'Clone {i+1}', 
                         fontsize=12, fontweight='bold', va='top',
                         transform=ax_structure.transAxes)
        
        # Mutations
        y_offset = 0.04
        for mut_name in mut_names:
            ax_structure.text(0.12, y_pos - y_offset, f'• {mut_name}', 
                             fontsize=9, va='top',
                             transform=ax_structure.transAxes)
            y_offset += 0.025
        
        # Zygosity
        ax_structure.text(0.12, y_pos - y_offset - 0.01, zyg_str, 
                         fontsize=9, va='top', style='italic', color='gray',
                         transform=ax_structure.transAxes)
        
        y_pos -= (box_height + 0.05)
    
    ax_structure.set_xlim([0, 1])
    ax_structure.set_ylim([0, 1])
    
    # ==========================================================================
    # 3. Fitness Estimates with Zygosity MAP and CI (bottom, spans both columns)
    # ==========================================================================
    ax_fitness = fig.add_subplot(gs[1, :])
    ax_fitness.axis('off')
    
    # Create fitness table with zygosity MAP and CI
    fitness_data = []
    fitness_data.append(['Clone', 'Mutations', 'h (MAP)', 'h (90% CI)', 's (MAP)', 's (90% CI)'])
    
    # Get joint_inference results if available
    joint_inference = model.get('joint_inference', None)
    
    for i, (clone_muts, h, color) in enumerate(zip(cs, h_vec, colors)):
        mut_names = [part.obs.index[j] for j in clone_muts]
        mut_str = ', '.join(mut_names[:3])
        if len(mut_names) > 3:
            mut_str += f' +{len(mut_names)-3}'
        
        # Zygosity MAP
        h_map_str = f'{h:.3f}'
        
        # Zygosity CI
        if joint_inference is not None and i < len(joint_inference):
            result = joint_inference[i]
            if 'h_ci' in result:
                h_ci_low, h_ci_high = result['h_ci']
                h_ci_str = f'[{h_ci_low:.3f}, {h_ci_high:.3f}]'
            else:
                h_ci_str = 'N/A'
        else:
            h_ci_str = 'N/A'
        
        # Fitness MAP
        if 'fitness' in part.obs and len(clone_muts) > 0:
            fitness = part.obs['fitness'].iloc[clone_muts[0]]
            if not np.isnan(fitness):
                s_map_str = f'{fitness:.3f}'
            else:
                s_map_str = 'N/A'
        else:
            s_map_str = 'N/A'
        
        # Fitness CI
        if 'fitness_5' in part.obs and 'fitness_95' in part.obs and len(clone_muts) > 0:
            ci_low = part.obs['fitness_5'].iloc[clone_muts[0]]
            ci_high = part.obs['fitness_95'].iloc[clone_muts[0]]
            if not (np.isnan(ci_low) or np.isnan(ci_high)):
                s_ci_str = f'[{ci_low:.3f}, {ci_high:.3f}]'
            else:
                s_ci_str = 'N/A'
        else:
            s_ci_str = 'N/A'
        
        fitness_data.append([f'Clone {i+1}', mut_str, h_map_str, h_ci_str, s_map_str, s_ci_str])
    
    # Create table
    table = ax_fitness.table(cellText=fitness_data, 
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.08, 0.30, 0.10, 0.18, 0.10, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for j in range(6):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Color code rows by clone
    for i, color in enumerate(colors[:len(cs)], start=1):
        for j in range(6):
            table[(i, j)].set_facecolor(tuple(list(color) + [0.2]))
            table[(i, j)].set_edgecolor(color)
            table[(i, j)].set_linewidth(2)
    
    ax_fitness.text(0.5, 0.95, 'Parameter Estimates', 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   transform=ax_fitness.transAxes)
    
    fig.suptitle(f'Clonal Evolution Analysis: {participant_id}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("="*80)
    print("MDS CLONAL EVOLUTION PLOTTING (WITH h MAP AND CI)")
    print("="*80)
    
    # Load data
    print(f"\nLoading results from: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'rb') as f:
            processed_parts = pk.load(f)
        print(f"✅ Loaded {len(processed_parts)} participants")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {INPUT_FILE}")
        print(f"   Make sure you've run the processing script first!")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    # Generate plots
    print(f"\nGenerating plots...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    success = 0
    failed = 0
    
    for i, part in enumerate(processed_parts):
        participant_id = part.uns.get('participant_id', f'participant_{i}')
        print(f"\n[{i+1}/{len(processed_parts)}] {participant_id}")
        
        try:
            fig = plot_participant_comprehensive(part)
            
            if fig is not None:
                filename = f"{OUTPUT_DIR}{participant_id}_comprehensive.png"
                fig.savefig(filename, bbox_inches='tight', dpi=DPI)
                print(f"  ✅ Saved: {filename}")
                success += 1
                
                if not SHOW_PLOTS:
                    plt.close(fig)
            else:
                print(f"  ⚠️ Skipped (no valid model)")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully plotted: {success}/{len(processed_parts)}")
    print(f"Failed: {failed}/{len(processed_parts)}")
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    
    if SHOW_PLOTS:
        print("\nDisplaying plots...")
        plt.show()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()