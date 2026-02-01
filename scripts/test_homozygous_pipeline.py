# test_homozygous_pipeline.py
import sys
sys.path.append("..")
from src.hom_inference import compute_clonal_models_prob_vec, refine_optimal_model_posterior_vec
import pickle as pk

print("Testing homozygous inference on homozygous data...")
print("="*60)

with open('../exports/homozygous_test_cohort.pk', 'rb') as f:
    hom_patients = pk.load(f)

for i, patient in enumerate(hom_patients[:3]):  # Test first 3
    print(f"\nPatient {i}: {patient.uns['patient_id']}")
    print(f"  True structure: {patient.uns['true_clonal_structure']}")
    
    # Check VAFs (should be high)
    vaf0 = patient.layers['AO'][0] / patient.layers['DP'][0]
    print(f"  Mutation 0 VAFs: {vaf0.round(3)}")
    
    # Run inference
    patient_inf = patient.copy()
    try:
        patient_inf = compute_clonal_models_prob_vec(patient_inf, s_resolution=30)
        patient_inf = refine_optimal_model_posterior_vec(patient_inf, s_resolution=60)
        
        # Get results
        if patient_inf.uns['model_dict']:
            top_key = list(patient_inf.uns['model_dict'].keys())[0]
            inferred_struct = patient_inf.uns['model_dict'][top_key][0]
            print(f"  Inferred structure: {inferred_struct}")
            print(f"  Correct: {inferred_struct == patient.uns['true_clonal_structure']}")
    except Exception as e:
        print(f"  ERROR: {e}")