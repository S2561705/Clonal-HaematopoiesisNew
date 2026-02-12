"""
Run Inference on Oval Posterior Synthetic Data

Runs the clonal inference pipeline on synthetic data optimized for
producing clear, oval-shaped joint posteriors.
"""

import sys
sys.path.append("..")

import pickle as pk
import pandas as pd
from pathlib import Path

# Import inference functions
from src.KI_clonal_inference_2 import (
    compute_clonal_models_prob_vec_mixed,
    refine_optimal_model_posterior_vec
)

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path('../exports/synthetic_oval/')
COHORT_FILE = DATA_DIR / 'synthetic_oval_cohort.pk'
GROUND_TRUTH_FILE = DATA_DIR / 'synthetic_oval_ground_truth.csv'
OUTPUT_FILE = DATA_DIR / 'cohort_with_inference.pk'

# Inference parameters
S_RESOLUTION = 50  # Grid resolution for fitness
H_RESOLUTION = 30  # Grid resolution for zygosity (for joint inference)
MIN_S = 0.01
MAX_S = 1.0
RESOLUTION = 600  # Grid resolution for het/hom sampling (HMM)

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("="*80)
    print("RUNNING INFERENCE ON OVAL POSTERIOR SYNTHETIC DATA")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {COHORT_FILE}")
    try:
        with open(COHORT_FILE, 'rb') as f:
            cohort = pk.load(f)
        print(f"✅ Loaded {len(cohort)} participants")
    except FileNotFoundError:
        print(f"❌ File not found: {COHORT_FILE}")
        print(f"   Run generate_synthetic_oval.py first!")
        return
    
    try:
        ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
        print(f"✅ Loaded ground truth")
    except FileNotFoundError:
        print(f"❌ File not found: {GROUND_TRUTH_FILE}")
        return
    
    # Run inference on each participant
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print("="*80)
    
    n_success = 0
    n_fail = 0
    
    for i, part in enumerate(cohort):
        participant_id = part.uns['participant_id']
        gt = ground_truth_df[ground_truth_df['participant_id'] == participant_id].iloc[0]
        
        print(f"\n[{i+1}/{len(cohort)}] {participant_id}")
        print(f"  Ground truth: h={gt['h_true']:.3f}, s={gt['s_true']:.3f} ({gt['scenario']})")
        
        try:
            # Step 1: Model selection
            print(f"  Computing clonal models...")
            part = compute_clonal_models_prob_vec_mixed(
                part,
                s_resolution=S_RESOLUTION,
                min_s=MIN_S,
                max_s=MAX_S,
                filter_invalid=False,  # Single mutation
                disable_progressbar=True,
                resolution=RESOLUTION
            )
            
            # Step 2: Refine with joint (s, h) inference
            print(f"  Refining with joint (s, h) inference...")
            part = refine_optimal_model_posterior_vec(
                part,
                s_resolution=S_RESOLUTION,
                h_resolution=H_RESOLUTION
            )
            
            # Extract results
            if 'optimal_model' in part.uns and 'joint_inference' in part.uns['optimal_model']:
                joint_results = part.uns['optimal_model']['joint_inference'][0]
                h_inferred = joint_results['h_map']
                s_inferred = joint_results['s_map']
                
                h_error = abs(h_inferred - gt['h_true'])
                s_error = abs(s_inferred - gt['s_true'])
                
                print(f"  ✅ Inferred: h={h_inferred:.3f} (Δ={h_error:.3f}), "
                      f"s={s_inferred:.3f} (Δ={s_error:.3f})")
                
                n_success += 1
            else:
                print(f"  ❌ Failed: No joint inference results")
                n_fail += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            n_fail += 1
    
    # Save cohort with inference results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    with open(OUTPUT_FILE, 'wb') as f:
        pk.dump(cohort, f)
    print(f"✅ Saved: {OUTPUT_FILE}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"  ✅ Success: {n_success}/{len(cohort)}")
    print(f"  ❌ Failed: {n_fail}/{len(cohort)}")
    print(f"\nNext step: python plot_joint_posteriors_oval.py")


if __name__ == '__main__':
    main()