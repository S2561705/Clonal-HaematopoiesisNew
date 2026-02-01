"""
Diagnostic script to inspect data before sending to inference
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from generate_synthetic_data import generate_synthetic_clonal_data, create_anndata_from_synthetic

print("="*70)
print("DIAGNOSTIC: INSPECTING DATA FOR INFERENCE")
print("="*70)

# Generate data
data = generate_synthetic_clonal_data(
    n_clones=2,
    n_mutations_per_clone=1,
    n_timepoints=6,
    fitness_values=[0.3, 0.7],
    depth_mean=8000,
    depth_std=1000,
    seed=42,
    plot_during_generation=False
)

# Convert to inference format
part = create_anndata_from_synthetic(data)

print("\n" + "="*70)
print("DATA STRUCTURE INSPECTION")
print("="*70)

print("\n1. PART.X (VAF matrix - mutations × timepoints):")
print(f"   Shape: {part.X.shape}")
print(f"   Type: {type(part.X)}")
print(f"   Dtype: {part.X.dtype}")
print("\n   Values:")
vaf_df = pd.DataFrame(
    part.X,
    index=part.obs.index,
    columns=[f'T{i}' for i in range(part.X.shape[1])]
)
print(vaf_df)

print("\n2. PART.LAYERS['AO'] (Alternate allele counts - mutations × timepoints):")
print(f"   Shape: {part.layers['AO'].shape}")
print(f"   Type: {type(part.layers['AO'])}")
print(f"   Dtype: {part.layers['AO'].values.dtype}")
print("\n   Values:")
print(part.layers['AO'])

print("\n3. PART.LAYERS['DP'] (Depth - mutations × timepoints):")
print(f"   Shape: {part.layers['DP'].shape}")
print(f"   Type: {type(part.layers['DP'])}")
print(f"   Dtype: {part.layers['DP'].values.dtype}")
print("\n   Values:")
print(part.layers['DP'])

print("\n4. PART.VAR.TIME_POINTS:")
print(f"   Type: {type(part.var.time_points)}")
print(f"   Values: {part.var.time_points.values}")
print(f"   Dtype: {part.var.time_points.dtype}")

print("\n5. PART.OBS (Mutation metadata):")
print(part.obs)

print("\n6. PART.SHAPE:")
print(f"   Value: {part.shape}")
print(f"   Type: {type(part.shape)}")

print("\n" + "="*70)
print("CHECKING WHAT INFERENCE WILL SEE")
print("="*70)

# Test what your inference code will extract
print("\n1. When inference does: part.layers['AO'].T")
AO_T = part.layers['AO'].T
print(f"   Shape: {AO_T.shape} (should be timepoints × mutations)")
print(f"   Type: {type(AO_T)}")
print(f"   Values:")
print(AO_T)

print("\n2. When inference does: part.layers['DP'].T")
DP_T = part.layers['DP'].T
print(f"   Shape: {DP_T.shape} (should be timepoints × mutations)")
print(f"   Values:")
print(DP_T)

print("\n3. VAF calculation: AO/DP")
VAF_calc = AO_T.values / DP_T.values
print(f"   Shape: {VAF_calc.shape}")
print(f"   Values:")
print(VAF_calc)

print("\n4. Check for any NaN or Inf values:")
print(f"   NaN in AO: {np.isnan(AO_T.values).any()}")
print(f"   NaN in DP: {np.isnan(DP_T.values).any()}")
print(f"   NaN in VAF: {np.isnan(VAF_calc).any()}")
print(f"   Inf in VAF: {np.isinf(VAF_calc).any()}")
print(f"   Zero in DP: {(DP_T.values == 0).any()}")

print("\n5. Data ranges:")
print(f"   AO range: [{AO_T.values.min():.0f}, {AO_T.values.max():.0f}]")
print(f"   DP range: [{DP_T.values.min():.0f}, {DP_T.values.max():.0f}]")
print(f"   VAF range: [{VAF_calc.min():.4f}, {VAF_calc.max():.4f}]")

print("\n" + "="*70)
print("GROUND TRUTH COMPARISON")
print("="*70)

gt = data['ground_truth']
print(f"\nTrue clonal structure: {gt['clonal_structure']}")
print(f"True fitness values: {gt['fitness_values']}")
print(f"\nTrue clone sizes over time:")
for t in range(len(data['time_points'])):
    print(f"  T{t} (t={data['time_points'][t]:.1f}): {gt['clone_sizes'][t]}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if not np.isnan(VAF_calc).any() and not np.isinf(VAF_calc).any():
    print("✓ Data looks clean (no NaN or Inf values)")
else:
    print("✗ Data has NaN or Inf values!")

if VAF_calc.max() > 0.05:
    print(f"✓ VAFs are detectable (max = {VAF_calc.max():.3f})")
else:
    print(f"⚠️  VAFs might be too low (max = {VAF_calc.max():.3f})")

if DP_T.values.min() > 1000:
    print(f"✓ Sequencing depth is high (min = {DP_T.values.min():.0f})")
else:
    print(f"⚠️  Sequencing depth might be low (min = {DP_T.values.min():.0f})")

print("\nThis data is ready to be passed to:")
print("  compute_clonal_models_prob_vec_mixed(part, ...)")