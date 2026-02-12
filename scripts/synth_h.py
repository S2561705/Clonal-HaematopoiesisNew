"""
Test script for homozygous fraction (h) inference.
Creates synthetic patients with known h values and tests recovery accuracy.
UPDATED: Sampling starts when VAF reaches ~0.1 (detectable threshold)
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import anndata as ad
import os

# Import your inference functions
from src.KI_clonal_inference_3 import (
    compute_clonal_models_prob_vec_mixed,
    refine_optimal_model_posterior_vec
)

# ==============================================================================
# Synthetic Data Generation
# ==============================================================================

def generate_synthetic_patient(true_s, true_h, patient_id, 
                               n_timepoints=5, n_mutations=3,
                               N_w=1e5, DP_mean=2000, seed=42,
                               start_vaf=0.1, sampling_duration=4.0):
    """
    Generate synthetic clonal expansion data with known (s, h) parameters.
    
    Parameters:
    -----------
    true_s : float
        True fitness value
    true_h : float
        True homozygous fraction (0 = pure het, 1 = pure hom)
    patient_id : str
        Patient identifier
    n_timepoints : int
        Number of sampling timepoints
    n_mutations : int
        Number of mutations in the clone
    N_w : float
        Wild-type population size
    DP_mean : int
        Mean sequencing depth
    seed : int
        Random seed
    start_vaf : float
        VAF at which sampling begins (default 0.1 = 10%)
    sampling_duration : float
        Duration of sampling period in years after first sample
        
    Returns:
    --------
    synthetic_part : anndata object
        With .layers['AO'], .layers['DP'], .var.time_points
    ground_truth : dict
        True parameters for validation
    """
    
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"GENERATING SYNTHETIC PATIENT: {patient_id}")
    print(f"{'='*70}")
    print(f"True parameters:")
    print(f"  Fitness (s): {true_s:.3f}")
    print(f"  Homozygous fraction (h): {true_h:.3f}")
    if true_h < 0.1:
        zyg_type = "Pure Heterozygous"
    elif true_h > 0.9:
        zyg_type = "Pure Homozygous"
    else:
        zyg_type = "Mixed Zygosity"
    print(f"  Zygosity type: {zyg_type}")
    print(f"  Mutations: {n_mutations}")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Start VAF threshold: {start_vaf:.3f}")
    
    # Calculate initial clone size to achieve start_vaf
    # VAF = (N_het + 2*N_hom) / (2*N_w)
    # For heterozygous: VAF = N_het / (2*N_w) => N_het = 2*N_w*VAF
    # For homozygous: VAF = 2*N_hom / (2*N_w) = N_hom / N_w => N_hom = N_w*VAF
    # For mixed: N_total = N_het + N_hom, and (N_het + 2*N_hom) / (2*N_w) = VAF
    #            N_het = (1-h)*N_total, N_hom = h*N_total
    #            ((1-h)*N_total + 2*h*N_total) / (2*N_w) = VAF
    #            N_total*(1+h) / (2*N_w) = VAF
    #            N_total = 2*N_w*VAF / (1+h)
    
    N_mut_at_start = (2 * N_w * start_vaf) / (1 + true_h)
    
    # Calculate time to reach this starting size
    # N_mut_at_start = N_mut_0 * exp(s * t_start)
    # Assume very small initial size (1 cell)
    N_mut_0 = 1.0
    t_start = np.log(N_mut_at_start / N_mut_0) / true_s
    
    print(f"\n  Clone starts at size N=1 cell")
    print(f"  Grows to N={N_mut_at_start:.1f} (VAF={start_vaf:.3f}) at t={t_start:.2f} years")
    print(f"  Sampling from t={t_start:.2f} to t={t_start + sampling_duration:.2f} years")
    
    # Time points: start when VAF reaches threshold, sample over duration
    time_points = np.linspace(t_start, t_start + sampling_duration, n_timepoints)
    
    # Initialize output
    AO = np.zeros((n_mutations, n_timepoints))
    DP = np.zeros((n_mutations, n_timepoints))
    
    print(f"\n{'Time':<8} {'N_mut':<12} {'N_het':<12} {'N_hom':<12} {'True_VAF':<10}")
    print(f"{'-'*70}")
    
    for tp_idx, t in enumerate(time_points):
        # Exponential growth from t=0
        N_mut_t = N_mut_0 * np.exp(true_s * t)
        
        # Cap at 90% of wild-type population (biological constraint)
        N_mut_t = min(N_mut_t, N_w * 0.9)
        
        # Split into het/hom based on h
        N_hom = N_mut_t * true_h
        N_het = N_mut_t * (1 - true_h)
        
        # True VAF calculation
        # Heterozygous cells contribute 1 mutant allele per 2 total alleles
        # Homozygous cells contribute 2 mutant alleles per 2 total alleles
        true_vaf = (N_het + 2 * N_hom) / (2 * N_w)
        true_vaf = np.clip(true_vaf, 0, 1)
        
        print(f"{t:<8.2f} {N_mut_t:<12.1f} {N_het:<12.1f} {N_hom:<12.1f} {true_vaf:<10.4f}")
        
        # Generate observations for each mutation
        for mut_idx in range(n_mutations):
            # Sequencing depth (Poisson noise around mean)
            dp = np.random.poisson(DP_mean)
            
            # Observed reads (Binomial sampling from true VAF)
            ao = np.random.binomial(dp, true_vaf)
            
            # Add some mutation-specific noise (small variations)
            noise_factor = np.random.uniform(0.95, 1.05)
            ao = int(ao * noise_factor)
            ao = min(ao, dp)  # Can't exceed depth
            
            # Store
            AO[mut_idx, tp_idx] = ao
            DP[mut_idx, tp_idx] = dp
    
    # Create anndata structure
    # X should be (n_obs × n_vars) = (n_mutations × n_timepoints)
    synthetic_part = ad.AnnData(
        X=AO,  # Keep original orientation: mutations × timepoints
        layers={'AO': AO, 'DP': DP}
    )
    
    # Mutations are observations (rows)
    mut_names = [f'{patient_id}_mut_{i+1}' for i in range(n_mutations)]
    synthetic_part.obs_names = mut_names
    synthetic_part.obs['p_key'] = mut_names
    
    # Timepoints are variables (columns)
    # Use relative time (years since first sample)
    relative_time_points = time_points - time_points[0]
    tp_names = [f'tp_{i}' for i in range(n_timepoints)]
    synthetic_part.var_names = tp_names
    synthetic_part.var['time_points'] = relative_time_points
    
    # Store patient ID
    synthetic_part.uns['participant_id'] = patient_id
    
    ground_truth = {
        'true_s': true_s,
        'true_h': true_h,
        'zygosity_type': zyg_type,
        'time_points': relative_time_points,
        'absolute_time_points': time_points,
        'N_w': N_w,
        'n_mutations': n_mutations,
        'patient_id': patient_id,
        'start_vaf': start_vaf
    }
    
    # Show observed VAFs
    observed_vaf = AO / np.maximum(DP, 1)
    print(f"\n{'Mutation':<15} {'VAF trajectory':<50}")
    print(f"{'-'*70}")
    for mut_idx in range(n_mutations):
        vaf_str = ' → '.join([f'{v:.3f}' for v in observed_vaf[mut_idx]])
        print(f"{mut_names[mut_idx]:<15} {vaf_str}")
    
    print(f"\nRelative time points (years since first sample):")
    print(f"  {relative_time_points}")
    
    print(f"\n{'='*70}\n")
    
    return synthetic_part, ground_truth


# ==============================================================================
# Testing Function
# ==============================================================================

def test_h_inference(save_plots=True, output_dir='../exports/test_h_inference/'):
    """
    Test h inference on three synthetic patients:
    1. Pure heterozygous (h = 0.0)
    2. Pure homozygous (h = 1.0)
    3. Mixed zygosity (h = 0.5)
    
    All with moderate fitness (s ≈ 0.5)
    Sampling starts when VAF reaches 10% (detectable threshold)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("TESTING HOMOZYGOUS FRACTION (h) INFERENCE")
    print("="*80)
    print("\nGenerating 3 synthetic patients:")
    print("  1. Pure Heterozygous (h = 0.0)")
    print("  2. Pure Homozygous (h = 1.0)")
    print("  3. Mixed Zygosity (h = 0.5)")
    print(f"\nAll with fitness s ≈ 0.5")
    print(f"Sampling starts when VAF reaches ~10% (detectable)")
    print("="*80)
    
    # Test cases
    test_cases = [
        {'patient_id': 'SYN_HET', 'true_s': 0.5, 'true_h': 0.0, 'seed': 42},
        {'patient_id': 'SYN_HOM', 'true_s': 0.5, 'true_h': 1.0, 'seed': 43},
        {'patient_id': 'SYN_MIX', 'true_s': 0.5, 'true_h': 0.5, 'seed': 44},
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'#'*80}")
        print(f"TEST CASE {i+1}/3: {case['patient_id']}")
        print(f"{'#'*80}")
        
        # Generate synthetic data
        synth_part, ground_truth = generate_synthetic_patient(
            true_s=case['true_s'],
            true_h=case['true_h'],
            patient_id=case['patient_id'],
            n_timepoints=5,
            n_mutations=3,
            DP_mean=2000,
            seed=case['seed'],
            start_vaf=0.1,  # Start sampling at 10% VAF
            sampling_duration=4.0  # Sample over 4 years
        )
        
        # Run inference
        print(f"\n{'─'*70}")
        print("RUNNING INFERENCE...")
        print(f"{'─'*70}")
        
        try:
            # Step 1: Compute clonal models
            synth_part = compute_clonal_models_prob_vec_mixed(
                synth_part, 
                s_resolution=50,  # Reasonable resolution
                min_s=0.01,
                max_s=2.0,
                resolution=600,
                filter_invalid=False,  # Single clone, no need to filter
                master_key_seed=case['seed']
            )
            
            # Step 2: Refine optimal model with joint (s, h) inference
            synth_part = refine_optimal_model_posterior_vec(
                synth_part,
                s_resolution=201,  # High resolution for accurate MAP
                h_resolution=21    # 21 points: [0.0, 0.05, 0.1, ..., 1.0]
            )
            
            # Extract results
            joint_results = synth_part.uns['optimal_model']['joint_inference']
            result = joint_results[0]  # Single clone
            
            inferred_s = result['s_map']
            inferred_h = result['h_map']
            s_ci = result['s_ci']
            h_ci = result['h_ci']
            
            # Calculate errors
            s_error = abs(inferred_s - ground_truth['true_s'])
            h_error = abs(inferred_h - ground_truth['true_h'])
            
            # Check if truth is in credible interval
            s_in_ci = s_ci[0] <= ground_truth['true_s'] <= s_ci[1]
            h_in_ci = h_ci[0] <= ground_truth['true_h'] <= h_ci[1]
            
            # CI widths
            s_ci_width = s_ci[1] - s_ci[0]
            h_ci_width = h_ci[1] - h_ci[0]
            
            # Store results
            test_result = {
                'patient_id': case['patient_id'],
                'true_s': ground_truth['true_s'],
                'true_h': ground_truth['true_h'],
                'inferred_s': inferred_s,
                'inferred_h': inferred_h,
                's_error': s_error,
                'h_error': h_error,
                's_ci': s_ci,
                'h_ci': h_ci,
                's_ci_width': s_ci_width,
                'h_ci_width': h_ci_width,
                's_in_ci': s_in_ci,
                'h_in_ci': h_in_ci,
                's_posterior': result['s_posterior'],
                'h_posterior': result['h_posterior'],
                's_range': result['s_range'],
                'h_range': result['h_range'],
                'success': True
            }
            
            results.append(test_result)
            
            # Print results
            print(f"\n{'='*70}")
            print("RESULTS:")
            print(f"{'='*70}")
            print(f"\nFitness (s):")
            print(f"  True:      {ground_truth['true_s']:.3f}")
            print(f"  Inferred:  {inferred_s:.3f}")
            print(f"  Error:     {s_error:.3f}")
            print(f"  90% CI:    [{s_ci[0]:.3f}, {s_ci[1]:.3f}]")
            print(f"  CI Width:  {s_ci_width:.3f}")
            print(f"  In CI:     {s_in_ci} {'✓' if s_in_ci else '✗'}")
            
            print(f"\nHomozygous fraction (h):")
            print(f"  True:      {ground_truth['true_h']:.3f}")
            print(f"  Inferred:  {inferred_h:.3f}")
            print(f"  Error:     {h_error:.3f}")
            print(f"  90% CI:    [{h_ci[0]:.3f}, {h_ci[1]:.3f}]")
            print(f"  CI Width:  {h_ci_width:.3f}")
            print(f"  In CI:     {h_in_ci} {'✓' if h_in_ci else '✗'}")
            
            # Interpretation
            print(f"\nInterpretation:")
            if s_error < 0.1:
                print(f"  ✓ Fitness recovered accurately (error < 0.1)")
            else:
                print(f"  ✗ Fitness error is large (error = {s_error:.3f})")
            
            if h_error < 0.15:
                print(f"  ✓ Zygosity recovered accurately (error < 0.15)")
            else:
                print(f"  ✗ Zygosity error is large (error = {h_error:.3f})")
            
            if h_ci_width < 0.5:
                print(f"  ✓ Zygosity is well-constrained (CI width < 0.5)")
            else:
                print(f"  ⚠ Zygosity is uncertain (CI width = {h_ci_width:.3f})")
            
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"ERROR during inference:")
            print(f"{'='*70}")
            print(f"{e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'patient_id': case['patient_id'],
                'true_s': ground_truth['true_s'],
                'true_h': ground_truth['true_h'],
                'success': False,
                'error': str(e)
            })
    
    # ===========================================================================
    # Summary and Visualization
    # ===========================================================================
    
    print(f"\n{'#'*80}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'#'*80}\n")
    
    successful = [r for r in results if r.get('success', False)]
    
    if len(successful) > 0:
        print(f"{'Patient':<12} {'True s':<8} {'Inf s':<8} {'s Err':<8} {'True h':<8} {'Inf h':<8} {'h Err':<8}")
        print(f"{'-'*80}")
        for r in successful:
            print(f"{r['patient_id']:<12} {r['true_s']:<8.3f} {r['inferred_s']:<8.3f} "
                  f"{r['s_error']:<8.3f} {r['true_h']:<8.3f} {r['inferred_h']:<8.3f} "
                  f"{r['h_error']:<8.3f}")
        
        # Overall statistics
        s_errors = [r['s_error'] for r in successful]
        h_errors = [r['h_error'] for r in successful]
        
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS:")
        print(f"{'='*80}")
        print(f"Fitness (s) inference:")
        print(f"  Mean absolute error:   {np.mean(s_errors):.3f}")
        print(f"  Max absolute error:    {np.max(s_errors):.3f}")
        print(f"  All in 90% CI:         {all(r['s_in_ci'] for r in successful)}")
        
        print(f"\nZygosity (h) inference:")
        print(f"  Mean absolute error:   {np.mean(h_errors):.3f}")
        print(f"  Max absolute error:    {np.max(h_errors):.3f}")
        print(f"  All in 90% CI:         {all(r['h_in_ci'] for r in successful)}")
        
        # Success criteria
        print(f"\n{'='*80}")
        print("SUCCESS CRITERIA:")
        print(f"{'='*80}")
        s_success = all(e < 0.1 for e in s_errors)
        h_success = all(e < 0.15 for e in h_errors)
        ci_success = all(r['s_in_ci'] and r['h_in_ci'] for r in successful)
        
        print(f"  Fitness accuracy (all errors < 0.1):      {s_success} {'✓' if s_success else '✗'}")
        print(f"  Zygosity accuracy (all errors < 0.15):    {h_success} {'✓' if h_success else '✗'}")
        print(f"  Credible intervals (all contain truth):   {ci_success} {'✓' if ci_success else '✗'}")
        
        overall_success = s_success and h_success and ci_success
        print(f"\n  OVERALL: {['FAILED ✗', 'PASSED ✓'][overall_success]}")
        
    else:
        print("⚠ No successful inferences to summarize")
    
    # ===========================================================================
    # Plotting
    # ===========================================================================
    
    if save_plots and len(successful) > 0:
        print(f"\n{'='*80}")
        print("GENERATING PLOTS...")
        print(f"{'='*80}")
        
        fig, axes = plt.subplots(len(successful), 3, figsize=(15, 5*len(successful)))
        if len(successful) == 1:
            axes = axes[np.newaxis, :]
        
        for i, r in enumerate(successful):
            # Panel 1: Fitness posterior
            ax1 = axes[i, 0]
            ax1.plot(r['s_range'], r['s_posterior'], 'b-', linewidth=2)
            ax1.axvline(r['true_s'], color='green', linestyle='--', linewidth=2, label='True')
            ax1.axvline(r['inferred_s'], color='red', linestyle='-', linewidth=2, label='Inferred')
            ax1.axvspan(r['s_ci'][0], r['s_ci'][1], alpha=0.2, color='red', label='90% CI')
            ax1.set_xlabel('Fitness (s)', fontsize=12)
            ax1.set_ylabel('Posterior Density', fontsize=12)
            ax1.set_title(f"{r['patient_id']}: Fitness Posterior", fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Zygosity posterior
            ax2 = axes[i, 1]
            ax2.plot(r['h_range'], r['h_posterior'], 'b-', linewidth=2)
            ax2.axvline(r['true_h'], color='green', linestyle='--', linewidth=2, label='True')
            ax2.axvline(r['inferred_h'], color='red', linestyle='-', linewidth=2, label='Inferred')
            ax2.axvspan(r['h_ci'][0], r['h_ci'][1], alpha=0.2, color='red', label='90% CI')
            ax2.set_xlabel('Homozygous Fraction (h)', fontsize=12)
            ax2.set_ylabel('Posterior Density', fontsize=12)
            ax2.set_title(f"{r['patient_id']}: Zygosity Posterior", fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Panel 3: Summary
            ax3 = axes[i, 2]
            ax3.axis('off')
            
            summary_text = f"""
True Parameters:
  s = {r['true_s']:.3f}
  h = {r['true_h']:.3f}

Inferred (MAP):
  s = {r['inferred_s']:.3f}
  h = {r['inferred_h']:.3f}

Errors:
  Δs = {r['s_error']:.3f}
  Δh = {r['h_error']:.3f}

90% Credible Intervals:
  s: [{r['s_ci'][0]:.3f}, {r['s_ci'][1]:.3f}]
  h: [{r['h_ci'][0]:.3f}, {r['h_ci'][1]:.3f}]

CI Widths:
  s: {r['s_ci_width']:.3f}
  h: {r['h_ci_width']:.3f}

Truth in CI:
  s: {['✗', '✓'][r['s_in_ci']]}
  h: {['✗', '✓'][r['h_in_ci']]}
            """
            
            ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'h_inference_test_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {plot_path}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    results = test_h_inference(save_plots=True)

