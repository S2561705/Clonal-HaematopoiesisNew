import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
import pickle as pk

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
    print(f"Shape: {part.shape}")
    print(f"Layers: {list(part.layers.keys())}")
    
    if 'AO' not in part.layers:
        print(f"❌ ERROR: 'AO' layer missing!")
        error_count += 1
        continue

    # Check AO layer shape and content
    ao_layer = part.layers['AO']
    print(f"AO layer shape: {ao_layer.shape}")
    print(f"AO sample data: {ao_layer[:2] if ao_layer.size > 0 else 'Empty'}")

    try:
        # Vectorised clonal inference
        print("Running compute_clonal_models_prob_vec...")
        part = compute_clonal_models_prob_vec(part)
        print("Running refine_optimal_model_posterior_vec...")
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