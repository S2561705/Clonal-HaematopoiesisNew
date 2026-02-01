"""
Minimal test with extremely simple data to isolate the numerical issue
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from src.KI_clonal_inference import (
    compute_clonal_models_prob_vec_mixed,
    refine_optimal_model_posterior_vec
)
from generate_synthetic_data import create_anndata_from_synthetic

print("="*70)
print("MINIMAL INFERENCE TEST")
print("="*70)

# Create VERY simple synthetic data manually
# 1 mutation, 3 timepoints, clear growth
print("\nCreating minimal test data...")
print("  1 mutation")
print("  3 timepoints: [0, 5, 10]")
print("  VAFs: [0.1, 0.2, 0.4] (clear growth)")
print("  Depth: 8000 everywhere")

# Manually construct the data
AO = np.array([
    [800],   # t=0: VAF=0.1
    [1600],  # t=5: VAF=0.2
    [3200]   # t=10: VAF=0.4
], dtype=float)

DP = np.array([
    [8000],
    [8000],
    [8000]
], dtype=float)

time_points = np.array([0., 5., 10.])

# Create the mock data structure
data = {
    'AO': AO,
    'DP': DP,
    'time_points': time_points,
    'ground_truth': {
        'clonal_structure': [[0]],
        'fitness_values': np.array([0.5]),
        'mutation_names': ['TestMut'],
        'clone_sizes': np.array([[1000], [4000], [16000]]),
        'mutation_sizes_het': np.zeros((3, 1)),
        'mutation_sizes_hom': np.zeros((3, 1)),
        'mutation_zygosity': np.array(['het']),
        'total_cells': np.array([100000, 100000, 100000])
    }
}

# Convert to AnnData format
part = create_anndata_from_synthetic(data)

print("\nData structure:")
print(f"  part.X shape: {part.X.shape}")
print(f"  part.layers['AO'].T shape: {part.layers['AO'].T.shape}")
print(f"  part.layers['DP'].T shape: {part.layers['DP'].T.shape}")

print("\nVAFs:")
print(f"  {(AO / DP).flatten()}")

print("\n" + "="*70)
print("RUNNING INFERENCE")
print("="*70)

try:
    print("\nStep 1: Model comparison with LOW resolution...")
    part = compute_clonal_models_prob_vec_mixed(
        part,
        s_resolution=10,  # Very low for speed
        min_s=0.01,
        max_s=1.0,
        filter_invalid=False,  # No filtering
        resolution=100  # Low grid resolution
    )
    
    print("\n✓ Model comparison completed")
    print(f"  Models evaluated: {len(part.uns['model_dict'])}")
    
    # Check probabilities
    for key, (structure, prob) in list(part.uns['model_dict'].items())[:3]:
        print(f"  {key}: structure={structure}, prob={prob:.6e}")
    
    if np.isnan(list(part.uns['model_dict'].values())[0][1]):
        print("\n✗ PROBLEM: Probabilities are NaN")
        print("  This indicates numerical overflow/underflow in likelihood calculation")
        
        # Check if there's more detail
        if 'warning' in part.uns:
            print(f"  Warning: {part.uns['warning']}")
    else:
        print("\n✓ Got valid probabilities!")
        
        print("\nStep 2: Refining model...")
        part = refine_optimal_model_posterior_vec(part, s_resolution=20)
        
        if 'fitness' in part.obs.columns:
            print(f"\n✓ SUCCESS! Fitness estimated: {part.obs['fitness'].values[0]:.3f}")
            print(f"  True fitness was: 0.5")
            print(f"  Error: {abs(part.obs['fitness'].values[0] - 0.5):.3f}")
        else:
            print("\n✗ Refinement did not create fitness column")
            if 'warning' in part.uns:
                print(f"  Warning: {part.uns['warning']}")
    
except Exception as e:
    print(f"\n✗ ERROR during inference:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("""
If you see NaN probabilities, the issue is in your inference code's
numerical stability. Possible fixes:

1. Use log-space arithmetic in likelihood calculations
2. Normalize probabilities at each step to prevent underflow
3. Check the integration grid - it might be too coarse
4. Add numerical safeguards (clip values, add small epsilon)

The synthetic data is CORRECT. The issue is in how your inference
handles the numerical likelihood calculations.
""")