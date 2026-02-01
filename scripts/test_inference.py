"""
Test script for clonal inference with synthetic data
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import inference functions
sys.path.insert(0, os.path.abspath('..'))

# Import synthetic data generator
from generate_synthetic_data import (
    generate_synthetic_clonal_data, 
    create_anndata_from_synthetic,
    plot_synthetic_data
)

# Import inference functions - adjust the module name as needed
try:
    # Try importing from the parent directory
    from src.KI_clonal_inference import (
        compute_clonal_models_prob_vec_mixed,
        refine_optimal_model_posterior_vec
    )
    INFERENCE_AVAILABLE = True
except ImportError:
    print("Warning: Could not import inference functions.")
    print("Please update the import statement with the correct module path.")
    print("Continuing with data generation only...\n")
    INFERENCE_AVAILABLE = False


def test_inference_simple():
    """Test inference on a simple 2-clone scenario"""
    
    print("\n" + "="*70)
    print("TEST 1: Simple 2-clone scenario")
    print("="*70)
    
    # Generate synthetic data with known ground truth
    synthetic_data = generate_synthetic_clonal_data(
        n_clones=2,
        n_mutations_per_clone=2,
        n_timepoints=5,
        fitness_values=[0.3, 0.6],
        seed=42
    )
    
    # Create AnnData-like object
    part = create_anndata_from_synthetic(synthetic_data)
    
    # Print ground truth
    gt = synthetic_data['ground_truth']
    print("\nGROUND TRUTH:")
    print(f"  Clonal structure: {gt['clonal_structure']}")
    print(f"  Fitness values: {gt['fitness_values']}")
    print(f"  Mutation names: {gt['mutation_names']}")
    
    # Run inference (you'll uncomment these when ready)
    # Note: These are the vectorized versions which should be faster
    if INFERENCE_AVAILABLE:
        print("\nRunning inference...")
        part = compute_clonal_models_prob_vec_mixed(
            part, 
            s_resolution=30,  # Lower resolution for faster testing
            min_s=0.01,
            max_s=1.0,
            filter_invalid=True,
            resolution=400  # Grid resolution for integration
        )
        
        print("\nINFERRED RESULTS (after model comparison):")
        print(f"  Optimal clonal structure: {part.uns['optimal_model']['clonal_structure']}")
        
        # Refine optimal model to get fitness estimates
        print("\nRefining optimal model...")
        part = refine_optimal_model_posterior_vec(part, s_resolution=50)
        
        # Now we can access fitness values
        print("\nFINAL INFERRED RESULTS:")
        print(f"  Optimal clonal structure: {part.uns['optimal_model']['clonal_structure']}")
        print(f"  Inferred fitness: {part.obs['fitness'].values}")
        print(f"  Fitness 95% CI: {list(zip(part.obs['fitness_5'].values, part.obs['fitness_95'].values))}")
        
        # Plot comparison
        plot_inference_results(part, synthetic_data)
    else:
        print("\nSkipping inference - functions not available.")
        print("To run inference:")
        print("1. Update the import statement at the top of test_inference.py")
        print("2. Import your inference functions from the correct module")
        print("3. Re-run this script")
    
    return part, synthetic_data


def test_inference_complex():
    """Test inference on a more complex 3-clone scenario"""
    
    print("\n" + "="*70)
    print("TEST 2: Complex 3-clone scenario")
    print("="*70)
    
    # Generate synthetic data with known ground truth
    synthetic_data = generate_synthetic_clonal_data(
        n_clones=3,
        n_mutations_per_clone=[2, 1, 2],
        n_timepoints=6,
        time_points=np.array([0, 2, 4, 6, 8, 10]),
        fitness_values=[0.2, 0.5, 0.8],
        depth_mean=150,
        seed=123
    )
    
    # Create AnnData-like object
    part = create_anndata_from_synthetic(synthetic_data)
    
    # Print ground truth
    gt = synthetic_data['ground_truth']
    print("\nGROUND TRUTH:")
    print(f"  Clonal structure: {gt['clonal_structure']}")
    print(f"  Fitness values: {gt['fitness_values']}")
    print(f"  Mutation names: {gt['mutation_names']}")
    
    # Run inference (uncomment when ready)
    if INFERENCE_AVAILABLE:
        print("\nRunning inference...")
        part = compute_clonal_models_prob_vec_mixed(
            part, 
            s_resolution=30,
            min_s=0.01,
            max_s=1.0,
            filter_invalid=True,
            resolution=400
        )
        
        print("\nRefining optimal model...")
        part = refine_optimal_model_posterior_vec(part, s_resolution=50)
        
        print("\nINFERRED RESULTS:")
        print(f"  Optimal clonal structure: {part.uns['optimal_model']['clonal_structure']}")
        print(f"  Inferred fitness: {part.obs['fitness'].values}")
        
        plot_inference_results(part, synthetic_data)
    else:
        print("\nSkipping inference - functions not available.")
    
    return part, synthetic_data


def plot_inference_results(part, synthetic_data):
    """Compare inference results with ground truth"""
    
    gt = synthetic_data['ground_truth']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitness comparison
    ax = axes[0]
    
    inferred_structure = part.uns['optimal_model']['clonal_structure']
    n_clones_true = len(gt['clonal_structure'])
    n_clones_inferred = len(inferred_structure)
    
    # Ground truth
    x_true = np.arange(n_clones_true)
    ax.bar(x_true - 0.2, gt['fitness_values'], 0.4, label='Ground Truth', alpha=0.7)
    
    # Inferred (may have different number of clones)
    inferred_fitness = []
    fitness_lower = []
    fitness_upper = []
    
    for clone_idx, clone in enumerate(inferred_structure):
        # Get fitness from first mutation in clone
        fit = part.obs.iloc[clone[0]]['fitness']
        fit_5 = part.obs.iloc[clone[0]]['fitness_5']
        fit_95 = part.obs.iloc[clone[0]]['fitness_95']
        
        inferred_fitness.append(fit)
        fitness_lower.append(fit - fit_5)
        fitness_upper.append(fit_95 - fit)
    
    x_inferred = np.arange(n_clones_inferred)
    ax.bar(x_inferred + 0.2, inferred_fitness, 0.4, label='Inferred', alpha=0.7)
    ax.errorbar(x_inferred + 0.2, inferred_fitness, 
                yerr=[fitness_lower, fitness_upper],
                fmt='none', color='black', capsize=5)
    
    ax.set_xlabel('Clone', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    title = 'Fitness Comparison'
    if n_clones_true != n_clones_inferred:
        title += f'\n(True: {n_clones_true} clones, Inferred: {n_clones_inferred} clones)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(max(n_clones_true, n_clones_inferred)))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Posterior distributions
    ax = axes[1]
    s_range = part.uns['optimal_model']['s_range']
    posterior = part.uns['optimal_model']['posterior']
    
    for c in range(n_clones_inferred):
        # Normalize posterior
        p = posterior[:, c]
        if p.sum() > 0:
            p = p / p.sum()
            ax.plot(s_range, p, label=f'Inferred Clone {c}',
                   linewidth=2)
    
    # Mark true fitness values
    for c, true_fit in enumerate(gt['fitness_values']):
        ax.axvline(true_fit, color=f'C{c}', 
                  linestyle='--', alpha=0.5, label=f'True Clone {c}')
    
    ax.set_xlabel('Fitness (s)', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title('Posterior Distributions\n(dashed = ground truth)', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inference_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to inference_comparison.png")
    
    return fig


def quick_data_check(part):
    """Quick sanity check on the data format"""
    
    print("\nDATA FORMAT CHECK:")
    print("="*50)
    print(f"✓ part.X shape: {part.X.shape}")
    print(f"✓ part.obs shape: {part.obs.shape}")
    print(f"✓ part.var shape: {part.var.shape}")
    print(f"✓ AO layer shape: {part.layers['AO'].shape}")
    print(f"✓ DP layer shape: {part.layers['DP'].shape}")
    print(f"✓ Time points: {part.var.time_points.values}")
    print(f"✓ Mutation names: {list(part.obs.index)}")
    
    # Check for potential issues
    AO = part.layers['AO'].values
    DP = part.layers['DP'].values
    VAF = AO / DP
    
    print(f"\nDATA QUALITY CHECK:")
    print(f"✓ VAF range: [{VAF.min():.4f}, {VAF.max():.4f}]")
    print(f"✓ Depth range: [{DP.min():.0f}, {DP.max():.0f}]")
    print(f"✓ Zero VAFs: {np.sum(VAF == 0)} / {VAF.size}")
    print(f"✓ Missing data: {np.sum(np.isnan(VAF))}")
    
    if np.any(VAF > 0.5):
        print("⚠ Warning: Some VAFs > 0.5 (may indicate homozygous mutations)")
    
    if np.any(DP < 20):
        print("⚠ Warning: Some depths < 20 (low coverage)")
        
    print("="*50)


if __name__ == "__main__":
    
    # Test 1: Simple scenario
    print("\n" + "█"*70)
    print("TESTING SYNTHETIC DATA GENERATION")
    print("█"*70)
    
    part1, synth1 = test_inference_simple()
    quick_data_check(part1)
    
    # Visualize the synthetic data
    fig1 = plot_synthetic_data(synth1, save_path='test1_data_viz.png')
    
    # Test 2: Complex scenario
    part2, synth2 = test_inference_complex()
    quick_data_check(part2)
    
    fig2 = plot_synthetic_data(synth2, save_path='test2_data_viz.png')
    
    print("\n" + "█"*70)
    print("SYNTHETIC DATA READY FOR TESTING!")
    print("█"*70)
    print("\nTo run inference, uncomment the inference function calls in the test functions")
    print("and ensure your inference module is properly imported.")
    print("\nGenerated files:")
    print("  - test1_data_viz.png (simple scenario visualization)")
    print("  - test2_data_viz.png (complex scenario visualization)")
    print("  - AO_example1.csv, DP_example1.csv")
    print("  - AO_example2.csv, DP_example2.csv")