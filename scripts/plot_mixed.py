import sys
sys.path.append("..")
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG = {
    'input_file': '../exports/MDS/MDS_cohort_fitted_unified.pk',
    'output_dir': '../exports/MDS/figures/',
    'show_plots': True,
    'save_plots': True
}

import os
if CONFIG['save_plots']:
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_fitness_posteriors(part, figsize=(12, 4)):
    """Plot fitness posteriors for all clones in a participant"""
    if 'optimal_model' not in part.uns:
        print(f"No optimal model found for participant")
        return None
    
    model = part.uns['optimal_model']
    cs = model['clonal_structure']
    posterior = model['posterior']
    s_range = model['s_range']
    h_vec = model['h_vec']
    
    participant_id = part.uns.get('participant_id', 'Unknown')
    
    n_clones = len(cs)
    fig, axes = plt.subplots(1, n_clones, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, (clone_muts, h) in enumerate(zip(cs, h_vec)):
        ax = axes[i]
        
        p = posterior[:, i]
        if np.sum(p) > 0:
            p = p / np.sum(p)  # Normalize
            
            # Plot posterior
            ax.plot(s_range, p, linewidth=2, color='steelblue')
            ax.fill_between(s_range, p, alpha=0.3, color='steelblue')
            
            # Mark MAP estimate
            map_idx = np.argmax(p)
            map_s = s_range[map_idx]
            ax.axvline(map_s, color='red', linestyle='--', linewidth=1.5, 
                      label=f'MAP: {map_s:.3f}')
            
            # Add 95% CI
            if 'fitness_5' in part.obs and len(clone_muts) > 0:
                ci_low = part.obs['fitness_5'].iloc[clone_muts[0]]
                ci_high = part.obs['fitness_95'].iloc[clone_muts[0]]
                if not np.isnan(ci_low):
                    ax.axvspan(ci_low, ci_high, alpha=0.2, color='red', 
                             label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
        else:
            ax.text(0.5, 0.5, 'No valid\nposterior', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Zygosity classification
        if h < 0.1:
            zyg = 'Het'
        elif h > 0.9:
            zyg = 'Hom'
        else:
            zyg = f'Mixed\n(h={h:.2f})'
        
        # Mutation names
        mut_names = [part.obs.index[j] for j in clone_muts]
        title = f"Clone {i+1} ({zyg})\n" + '\n'.join(mut_names[:3])
        if len(mut_names) > 3:
            title += f'\n+{len(mut_names)-3} more'
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Fitness (s)')
        ax.set_ylabel('Posterior Density')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Remove extra axes
    for i in range(n_clones, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f'Fitness Posteriors: {participant_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_vaf_trajectories(part, figsize=(10, 6)):
    """Plot VAF trajectories with model fits"""
    if 'optimal_model' not in part.uns:
        print(f"No optimal model found")
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (clone_muts, h, color) in enumerate(zip(cs, h_vec, colors)):
        for mut_idx in clone_muts:
            vaf = VAF[mut_idx, :]
            mut_name = part.obs.index[mut_idx]
            
            # Plot observed VAF
            ax.plot(time_points, vaf, 'o-', color=color, 
                   label=f'Clone {i+1}: {mut_name}', markersize=8, linewidth=2)
            
            # Add error bars based on binomial uncertainty
            depth = DP[mut_idx, :]
            vaf_std = np.sqrt(vaf * (1 - vaf) / depth)
            ax.errorbar(time_points, vaf, yerr=1.96*vaf_std, 
                       fmt='none', color=color, alpha=0.3, capsize=5)
    
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('VAF', fontsize=12)
    ax.set_title(f'VAF Trajectories: {participant_id}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_clone_sizes(part, figsize=(10, 6)):
    """Plot inferred clone sizes over time"""
    if 'optimal_model' not in part.uns:
        return None
    
    participant_id = part.uns.get('participant_id', 'Unknown')
    model = part.uns['optimal_model']
    cs = model['clonal_structure']
    h_vec = model['h_vec']
    het_frac = model['het_frac']
    hom_frac = model['hom_frac']
    time_points = part.var.time_points.values
    
    n_clones = len(cs)
    colors = sns.color_palette("husl", n_clones)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Total clone size
    for i, (clone_muts, h, color) in enumerate(zip(cs, h_vec, colors)):
        total_size = het_frac[:, clone_muts[0]] + hom_frac[:, clone_muts[0]]
        mut_name = part.obs.index[clone_muts[0]]
        ax1.plot(time_points, total_size, 'o-', color=color, 
                label=f'Clone {i+1}: {mut_name}', markersize=8, linewidth=2)
    
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('Clone Size (cells)', fontsize=12)
    ax1.set_title('Total Clone Sizes', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Stacked het/hom composition
    for i, (clone_muts, h, color) in enumerate(zip(cs, h_vec, colors)):
        het_size = het_frac[:, clone_muts[0]]
        hom_size = hom_frac[:, clone_muts[0]]
        
        ax2.bar(time_points + i*0.15, het_size, width=0.15, 
               color=color, alpha=0.6, label=f'Clone {i+1} Het')
        ax2.bar(time_points + i*0.15, hom_size, width=0.15, 
               bottom=het_size, color=color, alpha=1.0, 
               label=f'Clone {i+1} Hom')
    
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Cell Count', fontsize=12)
    ax2.set_title('Het/Hom Cell Composition', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Clone Sizes: {participant_id}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_cohort_fitness_distribution(processed_parts, figsize=(12, 8)):
    """Plot fitness distribution across the cohort"""
    
    # Collect all fitness values
    fitness_data = []
    for part in processed_parts:
        if 'fitness' not in part.obs:
            continue
        
        participant_id = part.uns.get('participant_id', 'Unknown')
        
        for idx in part.obs.index:
            fitness = part.obs.loc[idx, 'fitness']
            if not np.isnan(fitness):
                fitness_data.append({
                    'participant': participant_id,
                    'mutation': idx,
                    'fitness': fitness,
                    'fitness_5': part.obs.loc[idx, 'fitness_5'],
                    'fitness_95': part.obs.loc[idx, 'fitness_95'],
                    'zygosity': part.obs.loc[idx, 'zygosity_type'],
                    'h': part.obs.loc[idx, 'homozygous_fraction']
                })
    
    df = pd.DataFrame(fitness_data)
    
    if len(df) == 0:
        print("No valid fitness data to plot")
        return None
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Fitness distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['fitness'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(df['fitness'].median(), color='red', linestyle='--', 
               linewidth=2, label=f'Median: {df["fitness"].median():.3f}')
    ax1.set_xlabel('Fitness (s)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Fitness Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Fitness by zygosity
    ax2 = fig.add_subplot(gs[0, 1])
    zyg_order = ['heterozygous', 'mixed', 'homozygous']
    zyg_data = [df[df['zygosity'] == z]['fitness'].values 
                for z in zyg_order if z in df['zygosity'].unique()]
    zyg_labels = [z for z in zyg_order if z in df['zygosity'].unique()]
    
    bp = ax2.boxplot(zyg_data, labels=zyg_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("Set2", len(zyg_data))):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Fitness (s)', fontsize=12)
    ax2.set_title('Fitness by Zygosity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Fitness vs homozygous fraction
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['h'], df['fitness'], s=100, alpha=0.6, 
                         c=df['fitness'], cmap='viridis', edgecolors='black')
    ax3.set_xlabel('Homozygous Fraction (h)', fontsize=12)
    ax3.set_ylabel('Fitness (s)', fontsize=12)
    ax3.set_title('Fitness vs Homozygous Fraction', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Fitness')
    
    # 4. Participant comparison
    ax4 = fig.add_subplot(gs[1, 1])
    participant_fitness = df.groupby('participant')['fitness'].mean().sort_values()
    colors_p = sns.color_palette("Set3", len(participant_fitness))
    ax4.barh(range(len(participant_fitness)), participant_fitness.values, color=colors_p)
    ax4.set_yticks(range(len(participant_fitness)))
    ax4.set_yticklabels(participant_fitness.index)
    ax4.set_xlabel('Mean Fitness (s)', fontsize=12)
    ax4.set_title('Mean Fitness by Participant', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Cohort-wide Fitness Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_zygosity_summary(processed_parts, figsize=(12, 5)):
    """Plot zygosity distribution summary"""
    
    # Collect zygosity data
    zyg_data = []
    for part in processed_parts:
        if 'zygosity_type' not in part.obs:
            continue
        
        participant_id = part.uns.get('participant_id', 'Unknown')
        
        for idx in part.obs.index:
            zyg_data.append({
                'participant': participant_id,
                'mutation': idx,
                'zygosity': part.obs.loc[idx, 'zygosity_type'],
                'h': part.obs.loc[idx, 'homozygous_fraction']
            })
    
    df = pd.DataFrame(zyg_data)
    
    if len(df) == 0:
        print("No zygosity data to plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Overall zygosity distribution
    zyg_counts = df['zygosity'].value_counts()
    colors = {'heterozygous': 'skyblue', 'mixed': 'orange', 
              'homozygous': 'salmon', 'unknown': 'gray'}
    plot_colors = [colors.get(z, 'gray') for z in zyg_counts.index]
    
    ax1.pie(zyg_counts.values, labels=zyg_counts.index, autopct='%1.1f%%',
           colors=plot_colors, startangle=90)
    ax1.set_title('Overall Zygosity Distribution', fontsize=12, fontweight='bold')
    
    # 2. H value distribution
    ax2.hist(df['h'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(0.1, color='green', linestyle='--', linewidth=2, 
               label='Het threshold (h=0.1)')
    ax2.axvline(0.9, color='red', linestyle='--', linewidth=2, 
               label='Hom threshold (h=0.9)')
    ax2.set_xlabel('Homozygous Fraction (h)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of h Values', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Zygosity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ==============================================================================
# Main Plotting Function
# ==============================================================================

def plot_all_results():
    """Generate all plots for the MDS cohort"""
    
    print("="*80)
    print("MDS COHORT RESULTS VISUALIZATION")
    print("="*80)
    
    # Load results
    print(f"\nLoading results from: {CONFIG['input_file']}")
    try:
        with open(CONFIG['input_file'], 'rb') as f:
            processed_parts = pk.load(f)
        print(f"✅ Loaded {len(processed_parts)} participants")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {CONFIG['input_file']}")
        return
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return
    
    # Cohort-level plots
    print("\n" + "="*80)
    print("GENERATING COHORT-LEVEL PLOTS")
    print("="*80)
    
    print("\n[1/2] Plotting cohort fitness distribution...")
    fig = plot_cohort_fitness_distribution(processed_parts)
    if fig and CONFIG['save_plots']:
        fig.savefig(f"{CONFIG['output_dir']}cohort_fitness_distribution.png", 
                   bbox_inches='tight')
        print(f"  ✅ Saved to {CONFIG['output_dir']}cohort_fitness_distribution.png")
    
    print("\n[2/2] Plotting zygosity summary...")
    fig = plot_zygosity_summary(processed_parts)
    if fig and CONFIG['save_plots']:
        fig.savefig(f"{CONFIG['output_dir']}zygosity_summary.png", 
                   bbox_inches='tight')
        print(f"  ✅ Saved to {CONFIG['output_dir']}zygosity_summary.png")
    
    # Individual participant plots
    print("\n" + "="*80)
    print("GENERATING PARTICIPANT-LEVEL PLOTS")
    print("="*80)
    
    for i, part in enumerate(processed_parts):
        participant_id = part.uns.get('participant_id', f'participant_{i}')
        print(f"\n[{i+1}/{len(processed_parts)}] {participant_id}")
        
        if 'optimal_model' not in part.uns:
            print("  ⚠️ No optimal model found, skipping...")
            continue
        
        # Fitness posteriors
        fig = plot_fitness_posteriors(part)
        if fig and CONFIG['save_plots']:
            filename = f"{CONFIG['output_dir']}{participant_id}_fitness_posteriors.png"
            fig.savefig(filename, bbox_inches='tight')
            print(f"  ✅ Saved fitness posteriors")
        
        # VAF trajectories
        fig = plot_vaf_trajectories(part)
        if fig and CONFIG['save_plots']:
            filename = f"{CONFIG['output_dir']}{participant_id}_vaf_trajectories.png"
            fig.savefig(filename, bbox_inches='tight')
            print(f"  ✅ Saved VAF trajectories")
        
        # Clone sizes
        fig = plot_clone_sizes(part)
        if fig and CONFIG['save_plots']:
            filename = f"{CONFIG['output_dir']}{participant_id}_clone_sizes.png"
            fig.savefig(filename, bbox_inches='tight')
            print(f"  ✅ Saved clone sizes")
        
        if not CONFIG['show_plots']:
            plt.close('all')
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    
    if CONFIG['show_plots']:
        print("\nDisplaying plots...")
        plt.show()
    
    print(f"\nAll plots saved to: {CONFIG['output_dir']}")


# ==============================================================================
# Run
# ==============================================================================

if __name__ == '__main__':
    plot_all_results()