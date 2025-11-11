import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
import pickle as pk
import numpy as np

# Load processed MDS cohort
print("Loading MDS cohort data...")
with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    MDS_cohort = pk.load(f)

print(f"Loaded {len(MDS_cohort)} participants")

processed_part_list = []
success_count = 0
error_count = 0

for i, part in enumerate(MDS_cohort):
    print(f"\n--- Processing Participant {i+1}/{len(MDS_cohort)} ---")
    print(f"Participant ID: {part.uns.get('participant_id', 'Unknown')}")
    print(f"Number of mutations: {part.n_obs}")
    
    # Detailed layer debugging
    print(f"Available layers: {list(part.layers.keys())}")
    
    if 'AO' not in part.layers:
        print(f"❌ ERROR: 'AO' layer missing! Available layers: {list(part.layers.keys())}")
        error_count += 1
        continue
    
    # Debug the AO layer structure
    ao_layer = part.layers['AO']
    print(f"AO layer type: {type(ao_layer)}")
    print(f"AO layer shape: {ao_layer.shape}")
    print(f"AO layer dtype: {ao_layer.dtype}")
    print(f"AO layer sample values: {ao_layer[:5] if ao_layer.size > 0 else 'Empty'}")
    
    # Check if it's the expected shape (n_variants x 1)
    if len(ao_layer.shape) != 2 or ao_layer.shape[1] != 1:
        print(f"❌ ERROR: AO layer has unexpected shape {ao_layer.shape}, expected (n_variants, 1)")
        error_count += 1
        continue
    
    if ao_layer.shape[0] != part.n_obs:
        print(f"❌ ERROR: AO layer shape {ao_layer.shape} doesn't match n_obs {part.n_obs}")
        error_count += 1
        continue

    try:
        # Test if we can convert to jnp array (what the function will try to do)
        import jax.numpy as jnp
        test_ao = jnp.array(ao_layer.T)
        print(f"✅ Successfully converted AO to jnp array with shape: {test_ao.shape}")
        
        # Vectorised clonal inference
        print("Running clonal inference...")
        part = compute_clonal_models_prob_vec(part)
        part = refine_optimal_model_posterior_vec(part, 201)
        processed_part_list.append(part)
        success_count += 1
        print(f"✅ Successfully processed participant {i+1}")
        
    except Exception as e:
        print(f"❌ ERROR during clonal inference: {e}")
        import traceback
        traceback.print_exc()
        error_count += 1
        continue

# Export processed participant data
print(f"\n--- Summary ---")
print(f"Successfully processed: {success_count}/{len(MDS_cohort)} participants")
print(f"Errors: {error_count}")

if processed_part_list:
    with open('../exports/MDS/MDS_cohort_fitted.pk', 'wb') as f:
        pk.dump(processed_part_list, f)
    print(f"Results saved to ../exports/MDS/MDS_cohort_fitted.pk")
else:
    print("❌ No participants were successfully processed. Output file not created.")