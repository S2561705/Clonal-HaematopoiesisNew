import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pickle

with open('../exports/LBC/cohort_1/LBC_cohort_1_ns_LiFT_updated.pk', 'rb') as f:
    cohort_1 = pk.load(f)

with open('../exports/LBC/cohort_2/LBC_cohort_2_ns_LiFT_updated.pk', 'rb') as f:
    cohort_2 = pk.load(f)

# merge_cohorts
cohort = [part[part.obs.LiFT_value>0.9] for part in cohort_1 + cohort_2]
cohort = [part for part in cohort if part.shape[0]>0]

processed_part_list = []
for i, part in enumerate(cohort):
    print(f'Participant {i} of {len(cohort)}')
    
    # Vectorised clonal inference
    part = compute_clonal_models_prob_vec(part)
    part = refine_optimal_model_posterior_vec(part, 201)
    
    # Append processed results
    processed_part_list.append(part)

# Export results
with open('../exports/LBC/merged_cohort_fitted.pk', 'wb') as f:
    pk.dump(processed_part_list, f)

