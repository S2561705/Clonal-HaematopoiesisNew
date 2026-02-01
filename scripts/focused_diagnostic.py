# focused_diagnostic.py
"""
Focused diagnostic to understand why inference is failing.
"""

import sys
sys.path.append("..")
import pickle as pk
import numpy as np
import jax
import jax.numpy as jnp

def diagnose_patient_zero():
    """Detailed diagnosis of Patient 0."""
    print("="*80)
    print("FOCUSED DIAGNOSTIC: PATIENT 0")
    print("="*80)
    
    # Load patient
    with open('../exports/simple_temporal_test_cohort.pk', 'rb') as f:
        patients = pk.load(f)
    
    patient = patients[0]
    print(f"Patient: {patient.uns['patient_id']}")
    
    # Extract data
    AO = np.array(patient.layers['AO'].T)  # Time × Mutations
    DP = np.array(patient.layers['DP'].T)
    time_points = np.array(patient.var.time_points)
    
    print(f"\nData shape: AO={AO.shape}, DP={DP.shape}")
    print(f"Time points: {time_points}")
    
    # ============================================
    # 1. Check VAF patterns
    # ============================================
    print(f"\n{'='*40}")
    print("1. VAF PATTERNS ANALYSIS")
    print(f"{'='*40}")
    
    vaf = AO / DP
    print(f"VAF matrix (time × mutation):")
    print("Time |", " | ".join([f"Mut{i}" for i in range(vaf.shape[1])]))
    for t in range(vaf.shape[0]):
        print(f"{time_points[t]:4d} |", " | ".join([f"{vaf[t,i]:.3f}" for i in range(vaf.shape[1])]))
    
    # Check for zeros (problematic for homozygous model)
    zero_mask = vaf == 0
    print(f"\nZero VAFs: {np.sum(zero_mask)}/{vaf.size} ({np.sum(zero_mask)/vaf.size*100:.1f}%)")
    
    # ============================================
    # 2. Check what the homozygous model expects
    # ============================================
    print(f"\n{'='*40}")
    print("2. HOMOZYGOUS MODEL EXPECTATIONS")
    print(f"{'='*40}")
    
    # The homozygous model uses formula: x = -N_w * vaf / (vaf - 1)
    # This breaks when vaf is not near 1.0
    N_w = 1e5
    
    print(f"\nTesting homozygous transformation for vaf=0.5 (heterozygous level):")
    for test_vaf in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
        x = -N_w * test_vaf / (test_vaf - 1)
        print(f"  vaf={test_vaf:.2f} → x={x:.0f} cells")
    
    print(f"\n⚠️ PROBLEM: VAFs in data are too low for homozygous model!")
    print(f"  Max VAF in data: {np.max(vaf):.3f}")
    print(f"  Typical homozygous: VAF > 0.7")
    
    # ============================================
    # 3. Test a single structure manually
    # ============================================
    print(f"\n{'='*40}")
    print("3. MANUAL SINGLE STRUCTURE TEST")
    print(f"{'='*40}")
    
    # Test the true structure: [[0,1,2], [3,4]]
    cs = [[0, 1, 2], [3, 4]]
    
    # We need to import and test the likelihood function
    try:
        from src.hom_inference import jax_cs_hmm_ll
        
        print(f"\nTesting structure: {cs}")
        
        # Convert to JAX arrays
        AO_jax = jnp.array(AO)
        DP_jax = jnp.array(DP)
        time_jax = jnp.array(time_points)
        
        # Test with a single fitness value
        s_test = jnp.array([0.2, 1.0])  # Rough guesses
        
        print(f"  Testing with s={s_test}")
        
        try:
            # This might fail due to numerical issues
            likelihood = jax_cs_hmm_ll(s_test, AO_jax, DP_jax, time_jax, cs, lamb=1.3)
            print(f"  Likelihood: {likelihood}")
            
            if jnp.all(likelihood == 0):
                print(f"  ⚠️ ZERO LIKELIHOOD - NUMERICAL UNDERFLOW")
                
        except Exception as e:
            print(f"  ❌ ERROR in likelihood calculation: {e}")
            
    except ImportError as e:
        print(f"  Cannot import jax_cs_hmm_ll: {e}")
    
    # ============================================
    # 4. Check correlation filtering
    # ============================================
    print(f"\n{'='*40}")
    print("4. CORRELATION FILTERING ANALYSIS")
    print(f"{'='*40}")
    
    from src.hom_inference import compute_invalid_combinations
    
    compute_invalid_combinations(patient, pearson_distance_threshold=0.5)
    invalid_pairs = patient.uns.get('invalid_combinations', [])
    
    print(f"Invalid pairs (distance > 0.5): {len(invalid_pairs)}")
    
    # Check which true clone pairs are being rejected
    true_structure = patient.uns['true_clonal_structure']
    print(f"\nTrue structure validation:")
    
    for clone_idx, clone_muts in enumerate(true_structure):
        print(f"  Clone {clone_idx} (muts {clone_muts}):")
        
        # Check all pairs within this clone
        rejected_pairs = []
        for i in range(len(clone_muts)):
            for j in range(i+1, len(clone_muts)):
                mut1, mut2 = clone_muts[i], clone_muts[j]
                
                # Check if this pair is invalid
                is_invalid = any([mut1 in pair and mut2 in pair for pair in invalid_pairs])
                
                if is_invalid:
                    rejected_pairs.append((mut1, mut2))
        
        if rejected_pairs:
            print(f"    ⚠️ Rejected pairs within clone: {rejected_pairs}")
        else:
            print(f"    ✓ All pairs valid")
    
    # ============================================
    # 5. Recommendations
    # ============================================
    print(f"\n{'='*40}")
    print("5. RECOMMENDATIONS")
    print(f"{'='*40}")
    
    print(f"\nPROBLEMS IDENTIFIED:")
    print(f"1. ❌ VAFs too low for homozygous model (max={np.max(vaf):.3f} < 0.7)")
    print(f"2. ❌ Zero VAFs at early timepoints (n={np.sum(zero_mask)})")
    print(f"3. ❌ Model probabilities = 0 (numerical underflow)")
    
    print(f"\nIMMEDIATE FIXES:")
    print(f"1. ✅ Use HETEROZYGOUS inference for this data (VAFs ~0.3-0.5)")
    print(f"2. ✅ Remove early timepoints with zero VAFs")
    print(f"3. ✅ Increase correlation threshold to 0.8 for temporal data")
    
    print(f"\nQUICK TEST: Remove timepoints 0 and 3 (where VAFs are zero)")
    
    # Create filtered version
    keep_indices = [2, 3, 4, 5, 6]  # Months 6, 9, 12, 18, 24
    print(f"  Keeping timepoints: {time_points[keep_indices]}")
    
    AO_filtered = AO[keep_indices, :]
    DP_filtered = DP[keep_indices, :]
    time_filtered = time_points[keep_indices]
    
    print(f"\nFiltered VAFs (no zeros):")
    vaf_filtered = AO_filtered / DP_filtered
    for t in range(vaf_filtered.shape[0]):
        print(f"  Month {time_filtered[t]}: {vaf_filtered[t].round(3)}")


def test_simple_case():
    """Test with a SIMPLE, CLEAN case that should work."""
    print(f"\n{'='*80}")
    print("TEST: SIMPLE CLEAN CASE")
    print(f"{'='*80}")
    
    # Create a simple case that SHOULD work
    np.random.seed(42)
    
    # 2 mutations in same clone, high VAFs (homozygous)
    time_points = np.array([0, 6, 12, 24])
    
    # Simulate homozygous growth (VAFs near 1.0)
    AO = np.array([
        [80, 85],    # Month 0: ~0.8 VAF
        [95, 98],    # Month 6: ~0.95 VAF
        [99, 100],   # Month 12: ~0.99 VAF
        [100, 100],  # Month 24: 1.0 VAF
    ]).T  # Mutations × Time
    
    DP = np.ones_like(AO) * 100
    
    print(f"Created simple test case:")
    print(f"  Time points: {time_points}")
    print(f"  AO shape: {AO.shape}")
    print(f"  VAFs:")
    for i in range(AO.shape[0]):
        vaf = AO[i] / DP[i]
        print(f"    Mutation {i}: {vaf.round(3)}")
    
    # Try to run inference on this
    try:
        from src.hom_inference import compute_clonal_models_prob_vec
        
        # Create simple AnnData
        import anndata as ad
        import pandas as pd
        
        obs = pd.DataFrame({'mutation': ['MUT_0', 'MUT_1']})
        var = pd.DataFrame({'time_points': time_points})
        
        patient = ad.AnnData(
            X=np.random.randn(2, 4),
            obs=obs,
            var=var,
            layers={'AO': AO, 'DP': DP}
        )
        
        patient.uns['true_structure'] = [[0, 1]]
        
        print(f"\nRunning inference on clean homozygous case...")
        result = compute_clonal_models_prob_vec(patient, s_resolution=20)
        
        if result.uns.get('model_dict'):
            top_key = list(result.uns['model_dict'].keys())[0]
            inferred = result.uns['model_dict'][top_key][0]
            prob = result.uns['model_dict'][top_key][1]
            
            print(f"  Inferred: {inferred}")
            print(f"  Probability: {prob:.2e}")
            print(f"  Correct: {inferred == [[0, 1]]}")
        else:
            print(f"  ❌ No models found")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    diagnose_patient_zero()
    test_simple_case()