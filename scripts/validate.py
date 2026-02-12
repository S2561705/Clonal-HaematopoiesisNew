"""
Inference Validation Script

Runs clonal inference on synthetic data and compares results to ground truth.
Generates accuracy reports and diagnostic plots.
"""

import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path if needed
sys.path.append("../")

# Import your inference functions
from src.KI_clonal_inference_2 import (
    compute_clonal_models_prob_vec_mixed,
    refine_optimal_model_posterior_vec
)

# ==============================================================================
# Configuration
# ==============================================================================

SYNTHETIC_DATA_FILE = '../exports/synthetic/synthetic_single_mutation_cohort.pk'
GROUND_TRUTH_FILE = '../exports/synthetic/synthetic_ground_truth.csv'
OUTPUT_DIR = '../exports/synthetic/validation/'
RESULTS_FILE = 'inference_results.csv'

# Inference parameters
S_RESOLUTION = 50
MIN_S = 0.01
MAX_S = 1.0
RESOLUTION = 600  # Grid resolution for het/hom sampling

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Inference Runner
# ==============================================================================

def run_inference_on_cohort(cohort):
    """
    Run inference on synthetic cohort.
    
    Parameters:
    -----------
    cohort : list of AnnData
        Synthetic participant data
        
    Returns:
    --------
    results : list of dict
        Inference results for each participant
    """
    
    print("="*80)
    print("RUNNING INFERENCE ON SYNTHETIC COHORT")
    print("="*80)
    
    results = []
    
    for i, part in enumerate(cohort):
        participant_id = part.uns['participant_id']
        print(f"\n[{i+1}/{len(cohort)}] Processing {participant_id}...")
        
        try:
            # Run model selection
            print(f"  Computing clonal models...")
            part = compute_clonal_models_prob_vec_mixed(
                part,
                s_resolution=S_RESOLUTION,
                min_s=MIN_S,
                max_s=MAX_S,
                filter_invalid=False,  # Single mutation, no need to filter
                disable_progressbar=True,
                resolution=RESOLUTION
            )
            
            # Refine optimal model
            print(f"  Refining optimal model...")
            part = refine_optimal_model_posterior_vec(
                part,
                s_resolution=S_RESOLUTION * 2  # Higher resolution for refinement
            )
            
            # Extract results
            if 'optimal_model' in part.uns:
                model = part.uns['optimal_model']
                
                # Handle both new joint inference and legacy formats
                if 'joint_inference' in model:
                    # New joint (s, h) inference format
                    joint_results = model['joint_inference']
                    h_inferred = joint_results[0]['h_map'] if len(joint_results) > 0 else np.nan
                    s_inferred = joint_results[0]['s_map'] if len(joint_results) > 0 else np.nan
                    s_ci_low = joint_results[0]['s_ci'][0] if len(joint_results) > 0 else np.nan
                    s_ci_high = joint_results[0]['s_ci'][1] if len(joint_results) > 0 else np.nan
                    h_ci_low = joint_results[0]['h_ci'][0] if len(joint_results) > 0 else np.nan
                    h_ci_high = joint_results[0]['h_ci'][1] if len(joint_results) > 0 else np.nan
                else:
                    # Legacy format
                    h_inferred = model['h_vec'][0] if len(model.get('h_vec', [])) > 0 else np.nan
                    s_inferred = part.obs['fitness'].iloc[0] if 'fitness' in part.obs else np.nan
                    s_ci_low = part.obs['fitness_5'].iloc[0] if 'fitness_5' in part.obs else np.nan
                    s_ci_high = part.obs['fitness_95'].iloc[0] if 'fitness_95' in part.obs else np.nan
                    h_ci_low = np.nan
                    h_ci_high = np.nan
                
                result = {
                    'participant_id': participant_id,
                    'inference_success': True,
                    'clonal_structure': model['clonal_structure'],
                    'h_inferred': h_inferred,
                    's_inferred': s_inferred,
                    's_ci_low': s_ci_low,
                    's_ci_high': s_ci_high,
                    'h_ci_low': h_ci_low,
                    'h_ci_high': h_ci_high,
                    'model_probability': list(part.uns['model_dict'].values())[0][1]
                }
                
                print(f"  ✅ Success: h={result['h_inferred']:.3f}, s={result['s_inferred']:.3f}")
            else:
                result = {
                    'participant_id': participant_id,
                    'inference_success': False,
                    'h_inferred': np.nan,
                    's_inferred': np.nan,
                    's_ci_low': np.nan,
                    's_ci_high': np.nan
                }
                print(f"  ❌ Failed: No optimal model found")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'participant_id': participant_id,
                'inference_success': False,
                'error': str(e)
            })
    
    return results


# ==============================================================================
# Validation & Metrics
# ==============================================================================

def compute_accuracy_metrics(ground_truth_df, results_df):
    """
    Compute accuracy metrics comparing inference to ground truth.
    
    Parameters:
    -----------
    ground_truth_df : DataFrame
        Ground truth parameters
    results_df : DataFrame
        Inference results
        
    Returns:
    --------
    metrics : dict
        Accuracy metrics
    """
    
    print("\n" + "="*80)
    print("COMPUTING ACCURACY METRICS")
    print("="*80)
    
    # Merge ground truth and results
    merged = ground_truth_df.merge(
        results_df, 
        on='participant_id',
        how='left'
    )
    
    # Filter successful inferences
    success = merged[merged['inference_success'] == True].copy()
    
    if len(success) == 0:
        print("❌ No successful inferences to evaluate!")
        return {}, pd.DataFrame()
    
    print(f"\nSuccessful inferences: {len(success)}/{len(merged)}")
    
    # Compute errors
    success['h_error'] = success['h_inferred'] - success['h_true']
    success['s_error'] = success['s_inferred'] - success['s_true']
    
    success['h_abs_error'] = np.abs(success['h_error'])
    success['s_abs_error'] = np.abs(success['s_error'])
    
    success['h_rel_error'] = success['h_abs_error'] / (success['h_true'] + 0.01)  # Avoid div by 0
    success['s_rel_error'] = success['s_abs_error'] / (success['s_true'] + 0.01)
    
    # Check if true value is within CI
    success['s_in_ci'] = (
        (success['s_true'] >= success['s_ci_low']) & 
        (success['s_true'] <= success['s_ci_high'])
    )
    
    # Classify zygosity inference
    def classify_zygosity(h):
        if h < 0.1:
            return 'heterozygous'
        elif h > 0.9:
            return 'homozygous'
        else:
            return 'mixed'
    
    success['zygosity_inferred'] = success['h_inferred'].apply(classify_zygosity)
    success['zygosity_correct'] = (
        success['zygosity_type'] == success['zygosity_inferred']
    )
    
    # Compute metrics
    metrics = {
        'n_participants': len(merged),
        'n_success': len(success),
        'success_rate': len(success) / len(merged),
        
        # Zygosity accuracy
        'zygosity_accuracy': success['zygosity_correct'].mean(),
        'h_mae': success['h_abs_error'].mean(),
        'h_rmse': np.sqrt((success['h_error'] ** 2).mean()),
        'h_median_abs_error': success['h_abs_error'].median(),
        
        # Fitness accuracy
        's_mae': success['s_abs_error'].mean(),
        's_rmse': np.sqrt((success['s_error'] ** 2).mean()),
        's_median_abs_error': success['s_abs_error'].median(),
        's_ci_coverage': success['s_in_ci'].mean(),
        
        # By zygosity type
        'h_mae_by_type': success.groupby('zygosity_type')['h_abs_error'].mean().to_dict(),
        's_mae_by_type': success.groupby('zygosity_type')['s_abs_error'].mean().to_dict(),
    }
    
    # Print metrics
    print(f"\n{'='*80}")
    print("ACCURACY METRICS")
    print(f"{'='*80}")
    print(f"\nOverall:")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"\nZygosity Inference:")
    print(f"  Classification accuracy: {metrics['zygosity_accuracy']:.1%}")
    print(f"  h MAE: {metrics['h_mae']:.4f}")
    print(f"  h RMSE: {metrics['h_rmse']:.4f}")
    print(f"  h Median AE: {metrics['h_median_abs_error']:.4f}")
    print(f"\nFitness Inference:")
    print(f"  s MAE: {metrics['s_mae']:.4f}")
    print(f"  s RMSE: {metrics['s_rmse']:.4f}")
    print(f"  s Median AE: {metrics['s_median_abs_error']:.4f}")
    print(f"  95% CI coverage: {metrics['s_ci_coverage']:.1%}")
    
    print(f"\nBy Zygosity Type:")
    for zyg_type in ['heterozygous', 'homozygous', 'mixed']:
        if zyg_type in metrics['h_mae_by_type']:
            print(f"  {zyg_type.capitalize()}:")
            print(f"    h MAE: {metrics['h_mae_by_type'][zyg_type]:.4f}")
            print(f"    s MAE: {metrics['s_mae_by_type'][zyg_type]:.4f}")
    
    return metrics, success


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_validation_results(success_df, output_dir):
    """
    Create diagnostic plots comparing inference to ground truth.
    
    Parameters:
    -----------
    success_df : DataFrame
        Merged ground truth and inference results (successful only)
    output_dir : str
        Output directory for plots
    """
    
    print("\n" + "="*80)
    print("GENERATING VALIDATION PLOTS")
    print("="*80)
    
    output_path = Path(output_dir)
    
    # Set style
    sns.set_style("whitegrid")
    
    # =========================================================================
    # 1. Zygosity: True vs Inferred
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by zygosity type
    colors = {'heterozygous': 'blue', 'homozygous': 'red', 'mixed': 'green'}
    
    for zyg_type, color in colors.items():
        subset = success_df[success_df['zygosity_type'] == zyg_type]
        ax.scatter(subset['h_true'], subset['h_inferred'], 
                  c=color, label=zyg_type.capitalize(), 
                  s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Perfect prediction line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect prediction')
    
    # Confidence bands (±0.1)
    ax.fill_between([0, 1], [0, 1-0.1], [0, 1+0.1], 
                    alpha=0.2, color='gray', label='±0.1 band')
    
    ax.set_xlabel('True h (zygosity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inferred h (zygosity)', fontsize=14, fontweight='bold')
    ax.set_title('Zygosity Inference Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path / 'zygosity_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: zygosity_accuracy.png")
    plt.close()
    
    # =========================================================================
    # 2. Fitness: True vs Inferred
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for zyg_type, color in colors.items():
        subset = success_df[success_df['zygosity_type'] == zyg_type]
        
        # Plot points with error bars (95% CI)
        # Ensure error bars are positive by computing absolute deviations
        yerr_lower = np.maximum(subset['s_inferred'] - subset['s_ci_low'], 0)
        yerr_upper = np.maximum(subset['s_ci_high'] - subset['s_inferred'], 0)
        
        ax.errorbar(subset['s_true'], subset['s_inferred'],
                   yerr=[yerr_lower, yerr_upper],
                   fmt='o', c=color, label=zyg_type.capitalize(),
                   markersize=8, alpha=0.6, capsize=5, linewidth=2)
    
    # Perfect prediction line
    max_s = max(success_df['s_true'].max(), success_df['s_inferred'].max())
    ax.plot([0, max_s], [0, max_s], 'k--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('True fitness (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inferred fitness (s)', fontsize=14, fontweight='bold')
    ax.set_title('Fitness Inference Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'fitness_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: fitness_accuracy.png")
    plt.close()
    
    # =========================================================================
    # 3. Error Distributions
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # h error distribution
    axes[0, 0].hist(success_df['h_error'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('h Error (Inferred - True)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Zygosity Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # h absolute error by type
    success_df.boxplot(column='h_abs_error', by='zygosity_type', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Zygosity Type', fontsize=12)
    axes[0, 1].set_ylabel('Absolute h Error', fontsize=12)
    axes[0, 1].set_title('Zygosity Error by Type', fontsize=14, fontweight='bold')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45)
    
    # s error distribution
    axes[1, 0].hist(success_df['s_error'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('s Error (Inferred - True)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Fitness Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # s absolute error by type
    success_df.boxplot(column='s_abs_error', by='zygosity_type', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Zygosity Type', fontsize=12)
    axes[1, 1].set_ylabel('Absolute s Error', fontsize=12)
    axes[1, 1].set_title('Fitness Error by Type', fontsize=14, fontweight='bold')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)
    
    plt.suptitle('Error Distributions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path / 'error_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: error_distributions.png")
    plt.close()
    
    # =========================================================================
    # 4. CI Coverage
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coverage_by_type = success_df.groupby('zygosity_type')['s_in_ci'].mean()
    
    bars = ax.bar(range(len(coverage_by_type)), coverage_by_type.values, 
                  color=['blue', 'red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(coverage_by_type)))
    ax.set_xticklabels([t.capitalize() for t in coverage_by_type.index], fontsize=12)
    ax.set_ylabel('95% CI Coverage', fontsize=14, fontweight='bold')
    ax.set_title('Fitness 95% CI Coverage by Zygosity Type', fontsize=16, fontweight='bold')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Expected (95%)')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ci_coverage.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: ci_coverage.png")
    plt.close()
    
    print("\n✅ All plots generated successfully!")


def plot_vaf_trajectories(cohort, ground_truth_df, results_df, output_dir, max_plots=30):
    """
    Plot VAF trajectories for each synthetic participant showing:
    - Observed VAF (with error bars)
    - True underlying VAF
    - Inferred parameters
    
    Parameters:
    -----------
    cohort : list of AnnData
        Synthetic participant data
    ground_truth_df : DataFrame
        Ground truth parameters
    results_df : DataFrame
        Inference results
    output_dir : str
        Output directory for plots
    max_plots : int
        Maximum number of individual plots to generate
    """
    
    print("\n" + "="*80)
    print("GENERATING VAF TRAJECTORY PLOTS")
    print("="*80)
    
    output_path = Path(output_dir) / 'vaf_trajectories'
    output_path.mkdir(exist_ok=True)
    
    # Merge ground truth and results
    merged = ground_truth_df.merge(results_df, on='participant_id', how='left')
    
    # Also create a summary figure with multiple participants
    n_summary = min(9, len(cohort))  # 3x3 grid
    fig_summary, axes_summary = plt.subplots(3, 3, figsize=(15, 12))
    axes_summary = axes_summary.flatten()
    
    for i, (part, gt_row) in enumerate(zip(cohort, merged.to_dict('records'))):
        participant_id = part.uns['participant_id']
        
        # Get data
        AO = part.layers['AO'][0]
        DP = part.layers['DP'][0]
        VAF_obs = AO / np.maximum(DP, 1.0)
        time_points = part.var.time_points.values
        
        # Get ground truth
        h_true = gt_row['h_true']
        s_true = gt_row['s_true']
        vaf_true_initial = gt_row['vaf_initial']
        vaf_true_final = gt_row['vaf_final']
        zygosity_type = gt_row['zygosity_type']
        
        # Get inferred values
        h_inferred = gt_row.get('h_inferred', np.nan)
        s_inferred = gt_row.get('s_inferred', np.nan)
        inference_success = gt_row.get('inference_success', False)
        
        # Compute binomial error bars (95% CI)
        vaf_std = np.sqrt(VAF_obs * (1 - VAF_obs) / np.maximum(DP, 1))
        
        # Individual plot (only for first max_plots)
        if i < max_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot observed VAF with error bars
            ax.errorbar(time_points, VAF_obs, yerr=1.96*vaf_std,
                       fmt='o', markersize=10, capsize=5, capthick=2,
                       color='blue', label='Observed VAF', linewidth=2, alpha=0.7)
            
            # Plot true VAF trajectory (interpolated)
            ax.plot([time_points[0], time_points[-1]], 
                   [vaf_true_initial, vaf_true_final],
                   'g--', linewidth=3, label='True VAF trajectory', alpha=0.8)
            
            # Add reference lines
            ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Het max (0.5)')
            ax.axhline(0.25, color='lightgray', linestyle=':', linewidth=1.5, alpha=0.5, label='Pure het (~0.25)')
            
            # Formatting
            ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Variant Allele Frequency', fontsize=12, fontweight='bold')
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            # Title with ground truth and inferred values
            title = f'{participant_id} - {zygosity_type.capitalize()}\n'
            title += f'TRUE: h={h_true:.3f}, s={s_true:.3f}'
            if inference_success:
                h_error = h_inferred - h_true
                s_error = s_inferred - s_true
                title += f'\nINFERRED: h={h_inferred:.3f} (Δ={h_error:+.3f}), s={s_inferred:.3f} (Δ={s_error:+.3f})'
                ax.set_title(title, fontsize=11, fontweight='bold', color='darkgreen')
            else:
                title += f'\nINFERRENCE: FAILED'
                ax.set_title(title, fontsize=11, fontweight='bold', color='red')
            
            plt.tight_layout()
            plt.savefig(output_path / f'{participant_id}_vaf_trajectory.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Add to summary plot (first 9 only)
        if i < n_summary:
            ax = axes_summary[i]
            
            # Plot observed VAF with error bars
            ax.errorbar(time_points, VAF_obs, yerr=1.96*vaf_std,
                       fmt='o', markersize=6, capsize=3, capthick=1.5,
                       color='blue', linewidth=1.5, alpha=0.7)
            
            # Plot true trajectory
            ax.plot([time_points[0], time_points[-1]], 
                   [vaf_true_initial, vaf_true_final],
                   'g--', linewidth=2, alpha=0.8)
            
            # Reference lines
            ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.4)
            ax.axhline(0.25, color='lightgray', linestyle=':', linewidth=1, alpha=0.4)
            
            # Title
            if inference_success:
                color = 'darkgreen'
                status = '✓'
            else:
                color = 'red'
                status = '✗'
            
            ax.set_title(f'{participant_id} {status}\n{zygosity_type[:3]}: h={h_true:.2f}, s={s_true:.2f}',
                        fontsize=8, color=color)
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3)
            
            if i >= 6:  # Bottom row
                ax.set_xlabel('Time (years)', fontsize=8)
            if i % 3 == 0:  # Left column
                ax.set_ylabel('VAF', fontsize=8)
    
    # Hide unused subplots
    for j in range(n_summary, 9):
        axes_summary[j].axis('off')
    
    fig_summary.suptitle('VAF Trajectories - Summary (First 9 Participants)', 
                        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path.parent / 'vaf_trajectories_summary.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Generated {min(max_plots, len(cohort))} individual VAF trajectory plots")
    print(f"  ✅ Generated summary plot with {n_summary} participants")
    print(f"  ✅ Saved to: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run validation pipeline."""
    
    print("="*80)
    print("SYNTHETIC DATA VALIDATION PIPELINE")
    print("="*80)
    
    # Load synthetic data
    print(f"\nLoading synthetic data from: {SYNTHETIC_DATA_FILE}")
    try:
        with open(SYNTHETIC_DATA_FILE, 'rb') as f:
            cohort = pk.load(f)
        print(f"✅ Loaded {len(cohort)} participants")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {SYNTHETIC_DATA_FILE}")
        print(f"   Run generate_synthetic_data.py first!")
        return
    
    # Load ground truth
    print(f"\nLoading ground truth from: {GROUND_TRUTH_FILE}")
    try:
        ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
        print(f"✅ Loaded ground truth for {len(ground_truth_df)} participants")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {GROUND_TRUTH_FILE}")
        return
    
    # Run inference
    results = run_inference_on_cohort(cohort)
    results_df = pd.DataFrame(results)
    
    # Save results
    results_file = Path(OUTPUT_DIR) / RESULTS_FILE
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Saved inference results to: {results_file}")
    
    # Compute metrics
    metrics, success_df = compute_accuracy_metrics(ground_truth_df, results_df)
    
    # Generate validation plots only if we have successful inferences
    if len(success_df) > 0:
        plot_validation_results(success_df, OUTPUT_DIR)
        
        # Save metrics
        metrics_file = Path(OUTPUT_DIR) / 'accuracy_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write("INFERENCE ACCURACY METRICS\n")
            f.write("="*80 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"\n✅ Saved metrics to: {metrics_file}")
    else:
        print("\n⚠️  No successful inferences - skipping validation plots")
    
    # Generate VAF trajectory plots (regardless of inference success)
    plot_vaf_trajectories(cohort, ground_truth_df, results_df, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()