# test_temporal_pipeline.py
"""
Test the homozygous inference pipeline on temporal emergence data.
"""

import sys
sys.path.append("..")
import pickle as pk
import numpy as np
from src.hom_inference import compute_clonal_models_prob_vec, refine_optimal_model_posterior_vec
import time

def test_temporal_pipeline():
    """Test pipeline on temporal emergence data."""
    print("Testing homozygous inference on temporal emergence data...")
    print("="*60)
    
    # Load the test data
    with open('../exports/simple_temporal_test_cohort.pk', 'rb') as f:
        patients = pk.load(f)
    
    results = []
    
    for patient in patients:
        print(f"\nPatient: {patient.uns['patient_id']}")
        print(f"  Mutations: {patient.shape[0]}, Clones: {len(patient.uns['true_clonal_structure'])}")
        print(f"  True structure: {patient.uns['true_clonal_structure']}")
        
        # Show clone parameters
        print(f"  Clone parameters:")
        for i, params in enumerate(patient.uns['clone_params']):
            print(f"    Clone {i}: fitness={params['fitness']:.2f}, "
                  f"emerges={params['emergence']}mo, init={params['initial']} cells")
        
        # Show VAF patterns
        print(f"\n  VAF patterns (first mutation of each clone):")
        for clone_idx, clone_muts in enumerate(patient.uns['true_clonal_structure']):
            if clone_muts:  # Check if clone has mutations
                mut_idx = clone_muts[0]
                vaf = patient.layers['AO'][mut_idx] / patient.layers['DP'][mut_idx]
                print(f"    Clone {clone_idx}, Mut {mut_idx}: {vaf.round(3)}")
        
        # Run inference with SAFE settings
        try:
            patient_inf = patient.copy()
            
            # Safety: reduce resolution for larger cases
            if patient.shape[0] >= 8:
                s_res = 20
                print(f"  WARNING: {patient.shape[0]} mutations - using s_resolution=20")
            else:
                s_res = 30
            
            # Time the inference
            start_time = time.time()
            
            print(f"  Running compute_clonal_models_prob_vec (s_res={s_res})...")
            patient_inf = compute_clonal_models_prob_vec(
                patient_inf, 
                s_resolution=s_res,
                filter_invalid=True
            )
            
            print(f"  Running refine_optimal_model_posterior_vec...")
            patient_inf = refine_optimal_model_posterior_vec(
                patient_inf,
                s_resolution=min(60, s_res * 2)
            )
            
            elapsed = time.time() - start_time
            print(f"  Inference time: {elapsed:.1f}s")
            
            # Get results
            if patient_inf.uns.get('model_dict'):
                # Get top model
                top_key = list(patient_inf.uns['model_dict'].keys())[0]
                inferred_struct = patient_inf.uns['model_dict'][top_key][0]
                model_prob = patient_inf.uns['model_dict'][top_key][1]
                
                print(f"  Inferred structure: {inferred_struct}")
                print(f"  Model probability: {model_prob:.2e}")
                print(f"  Correct: {inferred_struct == patient.uns['true_clonal_structure']}")
                
                # Show top 3 models
                print(f"  Top 3 models:")
                for i, key in enumerate(list(patient_inf.uns['model_dict'].keys())[:3]):
                    cs, prob = patient_inf.uns['model_dict'][key]
                    print(f"    {i+1}. {cs} (prob={prob:.2e})")
                
                # Get fitness estimates if available
                if 'fitness' in patient_inf.obs.columns:
                    print(f"  Fitness estimates:")
                    for clone_idx, clone_muts in enumerate(inferred_struct):
                        if clone_muts:  # Check if clone has mutations
                            mut_idx = clone_muts[0]
                            fitness = patient_inf.obs.iloc[mut_idx]['fitness']
                            fitness_5 = patient_inf.obs.iloc[mut_idx].get('fitness_5', np.nan)
                            fitness_95 = patient_inf.obs.iloc[mut_idx].get('fitness_95', np.nan)
                            print(f"    Clone {clone_idx}: {fitness:.3f} "
                                  f"({fitness_5:.3f}-{fitness_95:.3f})")
            
            else:
                print(f"  ERROR: No models in model_dict")
                if 'warning' in patient_inf.uns:
                    print(f"  Warning: {patient_inf.uns['warning']}")
            
            results.append({
                'patient': patient.uns['patient_id'],
                'true_structure': patient.uns['true_clonal_structure'],
                'inferred_structure': inferred_struct if 'inferred_struct' in locals() else None,
                'correct': 'inferred_struct' in locals() and 
                          inferred_struct == patient.uns['true_clonal_structure']
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        print(f"Correct inferences: {correct_count}/{len(results)}")
        
        for r in results:
            status = "✓" if r['correct'] else "✗"
            print(f"{status} {r['patient']}:")
            print(f"  True: {r['true_structure']}")
            print(f"  Inferred: {r['inferred_structure']}")
    
    return results


def debug_individual_mutations(patient_idx=0):
    """Debug individual mutation patterns."""
    print(f"\n{'='*60}")
    print("DEBUG: INDIVIDUAL MUTATION ANALYSIS")
    print(f"{'='*60}")
    
    with open('../exports/simple_temporal_test_cohort.pk', 'rb') as f:
        patients = pk.load(f)
    
    patient = patients[patient_idx]
    print(f"Patient: {patient.uns['patient_id']}")
    
    # Calculate correlation distances
    from src.hom_inference import compute_invalid_combinations
    compute_invalid_combinations(patient, pearson_distance_threshold=0.5)
    
    invalid_pairs = patient.uns.get('invalid_combinations', [])
    print(f"\nInvalid pairs (correlation distance > 0.5): {len(invalid_pairs)}")
    
    if invalid_pairs:
        print("First 10 invalid pairs:")
        for i, pair in enumerate(invalid_pairs[:10]):
            print(f"  {pair}")
    
    # Check VAF correlations
    print(f"\nVAF correlations between mutations:")
    n_muts = patient.shape[0]
    for i in range(n_muts):
        for j in range(i+1, n_muts):
            vaf_i = patient.layers['AO'][i] / patient.layers['DP'][i]
            vaf_j = patient.layers['AO'][j] / patient.layers['DP'][j]
            correlation = np.corrcoef(vaf_i, vaf_j)[0, 1]
            
            # Check if they should be in same clone
            in_same_clone = False
            for clone in patient.uns['true_clonal_structure']:
                if i in clone and j in clone:
                    in_same_clone = True
                    break
            
            warning = "⚠️ SAME CLONE!" if in_same_clone and abs(1 - correlation) > 0.5 else ""
            print(f"  Mut {i} & {j}: corr={correlation:.3f} {warning}")


if __name__ == "__main__":
    # Run the main test
    results = test_temporal_pipeline()
    
    # Optional: debug first patient
    print(f"\n{'='*60}")
    debug = input("Debug individual mutation analysis for Patient 0? (y/n): ")
    if debug.lower() == 'y':
        debug_individual_mutations(0)