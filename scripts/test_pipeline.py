"""
Complete Inference Pipeline Test
=================================

This script tests your inference code with synthetic data and compares
the results against known ground truth.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

# Import your inference functions
from src.KI_clonal_inference import (
    compute_clonal_models_prob_vec_mixed,
    refine_optimal_model_posterior_vec
)

# Import synthetic data generator
from generate_synthetic_data import (
    generate_synthetic_clonal_data, 
    create_anndata_from_synthetic,
    plot_synthetic_data
)


def run_inference_test(n_clones=2, n_mutations_per_clone=2, fitness_values=None,
                       n_timepoints=5, s_resolution=50, seed=42):
    """
    Run a complete inference test with synthetic data
    """
    
    print("="*70)
    print("INFERENCE PIPELINE TEST")
    print("="*70)
    
    # ==========================================================================
    # STEP 1: Generate synthetic data with known ground truth
    # ==========================================================================
    
    print("\n📊 Step 1: Generating synthetic data...")
    
    if fitness_values is None:
        fitness_values = np.random.uniform(0.2, 0.8, n_clones)
    
    data = generate_synthetic_clonal_data(
        n_clones=n_clones,
        n_mutations_per_clone=n_mutations_per_clone,
        n_timepoints=n_timepoints,
        fitness_values=fitness_values,
        depth_mean=8000,  # High-depth sequencing (8000 ± 1000)
        depth_std=1000,
        seed=seed
    )
    
    gt = data['ground_truth']
    
    print("✓ Synthetic data generated!")
    print(f"  Ground truth structure: {gt['clonal_structure']}")
    print(f"  Ground truth fitness: {gt['fitness_values']}")
    print(f"  Mutations: {gt['mutation_names']}")
    print(f"  Timepoints: {data['time_points']}")
    
    # Visualize the data
    plot_synthetic_data(data, save_path='test_synthetic_data.png')
    print("  📈 Saved: test_synthetic_data.png")
    
    # ==========================================================================
    # STEP 2: Convert to inference format
    # ==========================================================================
    
    print("\n🔄 Step 2: Converting to inference format...")
    part = create_anndata_from_synthetic(data)
    
    print("✓ Data converted!")
    print(f"  part.X shape: {part.X.shape} (mutations × timepoints)")
    print(f"  part.layers['AO'] shape: {part.layers['AO'].shape}")
    print(f"  part.layers['DP'] shape: {part.layers['DP'].shape}")
    
    # ==========================================================================
    # STEP 3: Run inference - Model comparison
    # ==========================================================================
    
    print("\n🔬 Step 3: Running inference (model comparison)...")
    print(f"  s_resolution: {s_resolution}")
    print("  This will evaluate all possible clonal structures...")
    
    part = compute_clonal_models_prob_vec_mixed(
        part, 
        s_resolution=s_resolution,
        min_s=0.01,
        max_s=1.0,
        filter_invalid=True,
        resolution=600
    )
    
    print("✓ Model comparison complete!")
    print(f"  Evaluated {len(part.uns['model_dict'])} possible structures")
    
    # Get the best model from model_dict
    best_model_key = list(part.uns['model_dict'].keys())[0]
    best_structure, best_prob = part.uns['model_dict'][best_model_key]
    print(f"  Best structure: {best_structure} (probability: {best_prob:.3e})")
    
    # ==========================================================================
    # STEP 4: Refine optimal model
    # ==========================================================================
    
    print("\n🎯 Step 4: Refining optimal model (fitness estimation)...")
    
    # Debug: Check what we have before refinement
    print(f"  Debug: part.uns keys = {list(part.uns.keys())}")
    if 'model_dict' in part.uns:
        best_key = list(part.uns['model_dict'].keys())[0]
        best_structure, best_prob = part.uns['model_dict'][best_key]
        print(f"  Debug: Best structure = {best_structure}, prob = {best_prob:.3e}")
    
    part = refine_optimal_model_posterior_vec(part, s_resolution=100)
    
    print("✓ Refinement complete!")
    
    # Check if fitness column was created
    if 'fitness' not in part.obs.columns:
        print("  ⚠️  WARNING: 'fitness' column not created!")
        print(f"  Available columns: {list(part.obs.columns)}")
        
        # Check for warning
        if 'warning' in part.uns:
            print(f"  ⚠️  Warning from inference: {part.uns['warning']}")
        
        # Check if optimal_model exists
        if 'optimal_model' not in part.uns:
            print("  ✗ ERROR: optimal_model not created")
            print("  This suggests the refinement function failed")
            return None, data
        
        print("  ✗ ERROR: Cannot proceed without fitness estimates")
        return None, data
    
    print(f"  Fitness estimates available in part.obs['fitness']")
    
    # ==========================================================================
    # STEP 5: Compare results with ground truth
    # ==========================================================================
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    inferred_structure = part.uns['optimal_model']['clonal_structure']
    inferred_fitness = part.obs['fitness'].values
    
    print("\n🎯 INFERRED:")
    print(f"  Structure: {inferred_structure}")
    print(f"  Fitness values: {inferred_fitness.round(3)}")
    print("\n  Per-mutation breakdown:")
    for i, mut in enumerate(part.obs.index):
        print(f"    {mut}: fitness={part.obs.iloc[i]['fitness']:.3f} "
              f"[{part.obs.iloc[i]['fitness_5']:.3f}, {part.obs.iloc[i]['fitness_95']:.3f}]")
    
    print("\n✅ GROUND TRUTH:")
    print(f"  Structure: {gt['clonal_structure']}")
    print(f"  Fitness values: {gt['fitness_values']}")
    
    # ==========================================================================
    # STEP 6: Evaluate accuracy
    # ==========================================================================
    
    print("\n" + "="*70)
    print("ACCURACY ASSESSMENT")
    print("="*70)
    
    # Check if structures match
    structures_match = (str(sorted([sorted(c) for c in inferred_structure])) == 
                       str(sorted([sorted(c) for c in gt['clonal_structure']])))
    
    print(f"\n📊 Structure correct: {'✓ YES' if structures_match else '✗ NO'}")
    
    if structures_match:
        print("  ✓ Clonal structure perfectly recovered!")
        
        # Calculate fitness errors
        print("\n📈 Fitness accuracy:")
        errors = []
        for i, clone in enumerate(inferred_structure):
            mut_idx = clone[0]  # Use first mutation in clone
            inferred_fit = part.obs.iloc[mut_idx]['fitness']
            true_fit = gt['fitness_values'][i]
            error = abs(inferred_fit - true_fit)
            errors.append(error)
            
            status = "✓" if error < 0.1 else "⚠" if error < 0.2 else "✗"
            print(f"  {status} Clone {i}: error = {error:.3f} "
                  f"(true={true_fit:.3f}, inferred={inferred_fit:.3f})")
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\n  Mean absolute error: {mean_error:.3f}")
        print(f"  Max absolute error: {max_error:.3f}")
        
        if mean_error < 0.1:
            print("  🎉 EXCELLENT accuracy!")
        elif mean_error < 0.2:
            print("  👍 GOOD accuracy")
        else:
            print("  ⚠️  Moderate accuracy - consider higher resolution or more data")
    
    else:
        print("  ✗ Clonal structure NOT recovered correctly")
        print(f"    Expected {len(gt['clonal_structure'])} clones, "
              f"got {len(inferred_structure)} clones")
        print("\n  Possible reasons:")
        print("    - s_resolution too low (try 100 instead of {})".format(s_resolution))
        print("    - Fitness values too similar (need >0.2 difference)")
        print("    - Not enough timepoints (try 7-10 instead of {})".format(n_timepoints))
        print("    - Sequencing depth too low")
    
    # ==========================================================================
    # STEP 7: Create visualizations
    # ==========================================================================
    
    print("\n📊 Creating comparison plots...")
    plot_comparison(part, data)
    print("✓ Saved: inference_results_comparison.png")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    
    return part, data


def plot_comparison(part, data):
    """Create comprehensive comparison plots"""
    
    gt = data['ground_truth']
    inferred_structure = part.uns['optimal_model']['clonal_structure']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ==========================================================================
    # Plot 1: VAF trajectories with true clone sizes
    # ==========================================================================
    
    ax = axes[0, 0]
    
    # Plot VAF trajectories
    time_points = data['time_points']
    colors = plt.cm.tab10(np.linspace(0, 1, len(gt['mutation_names'])))
    
    for i, mut_name in enumerate(gt['mutation_names']):
        vaf = part.X[i, :]
        ax.plot(time_points, vaf, 'o-', label=mut_name, 
                color=colors[i], alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('VAF', fontsize=12)
    ax.set_title('VAF Trajectories Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Plot 2: Fitness comparison
    # ==========================================================================
    
    ax = axes[0, 1]
    
    n_true = len(gt['clonal_structure'])
    n_inferred = len(inferred_structure)
    
    x_true = np.arange(n_true)
    x_inferred = np.arange(n_inferred)
    
    # Ground truth
    ax.bar(x_true - 0.2, gt['fitness_values'], 0.4, 
           label='Ground Truth', alpha=0.7, color='green')
    
    # Inferred
    inferred_fitness = []
    inferred_errors_low = []
    inferred_errors_high = []
    
    for clone in inferred_structure:
        fit = part.obs.iloc[clone[0]]['fitness']
        fit_5 = part.obs.iloc[clone[0]]['fitness_5']
        fit_95 = part.obs.iloc[clone[0]]['fitness_95']
        
        inferred_fitness.append(fit)
        inferred_errors_low.append(fit - fit_5)
        inferred_errors_high.append(fit_95 - fit)
    
    ax.bar(x_inferred + 0.2, inferred_fitness, 0.4, 
           label='Inferred', alpha=0.7, color='blue')
    ax.errorbar(x_inferred + 0.2, inferred_fitness,
                yerr=[inferred_errors_low, inferred_errors_high],
                fmt='none', color='black', capsize=5, capthick=2)
    
    ax.set_xlabel('Clone', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    title = 'Fitness Estimates vs Ground Truth'
    if n_true != n_inferred:
        title += f'\n(True: {n_true} clones, Inferred: {n_inferred} clones)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(max(n_true, n_inferred)))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ==========================================================================
    # Plot 3: Posterior distributions
    # ==========================================================================
    
    ax = axes[1, 0]
    
    s_range = part.uns['optimal_model']['s_range']
    posterior = part.uns['optimal_model']['posterior']
    
    for c in range(n_inferred):
        p = posterior[:, c]
        if p.sum() > 0:
            p = p / p.sum()
            ax.plot(s_range, p, label=f'Inferred Clone {c}', linewidth=2)
    
    # Mark true fitness values
    for c, true_fit in enumerate(gt['fitness_values']):
        ax.axvline(true_fit, color=f'C{c}', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'True Clone {c}')
    
    ax.set_xlabel('Fitness (s)', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title('Fitness Posterior Distributions\n(dashed = ground truth)', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Plot 4: Model probabilities
    # ==========================================================================
    
    ax = axes[1, 1]
    
    # Get top 10 models
    models = list(part.uns['model_dict'].items())[:10]
    model_names = [f"Model {i}" for i in range(len(models))]
    model_probs = [v[1] for k, v in models]
    
    # Highlight the optimal model
    colors_bars = ['green' if i == 0 else 'gray' for i in range(len(models))]
    
    ax.barh(model_names, model_probs, color=colors_bars, alpha=0.7)
    ax.set_xlabel('Model Probability', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(f'Top {len(models)} Model Probabilities\n(green = optimal)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add structure annotations
    for i, (k, (structure, prob)) in enumerate(models):
        ax.text(prob, i, f'  {structure}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('inference_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    print("\n" + "█"*70)
    print("RUNNING INFERENCE PIPELINE TESTS")
    print("█"*70)
    
    # ==========================================================================
    # Test 1: Simple 2-clone system
    # ==========================================================================
    
    print("\n\n" + "="*70)
    print("TEST 1: Simple 2-clone system")
    print("="*70)
    
    part1, data1 = run_inference_test(
        n_clones=2,
        n_mutations_per_clone=1,  # Just 1 mutation per clone
        fitness_values=[0.3, 0.7],  # Clear separation
        n_timepoints=6,
        s_resolution=50,
        seed=42
    )
    
    if part1 is None:
        print("\n✗ Test 1 failed - skipping Test 2")
        import sys
        sys.exit(1)
    
    # ==========================================================================
    # Test 2: More challenging scenario
    # ==========================================================================
    
    print("\n\n" + "="*70)
    print("TEST 2: Challenging 3-clone system")
    print("="*70)
    
    part2, data2 = run_inference_test(
        n_clones=3,
        n_mutations_per_clone=1,  # Just 1 mutation per clone
        fitness_values=[0.2, 0.5, 0.8],
        n_timepoints=7,
        s_resolution=50,
        seed=123
    )
    
    print("\n\n" + "█"*70)
    print("ALL TESTS COMPLETE!")
    print("█"*70)
    print("\nGenerated files:")
    print("  - test_synthetic_data.png")
    print("  - inference_results_comparison.png")