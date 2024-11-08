import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from multiprocessing import Pool

with open('../exports/WHI/WHI_qc.pk', 'rb') as f:
    WHI_cohort = pk.load(f)

with open('../exports/sardiNIA/sardiNIA_qc.pk', 'rb') as f:
    sardinia_cohort = pk.load(f)

with open('../exports/LBC/cohort_1/LBC_cohort_1_ns_LiFT_updated.pk', 'rb') as f:
    LBC_1_cohort = pk.load(f)

with open('../exports/LBC/cohort_2/LBC_cohort_2_ns_LiFT_updated.pk', 'rb') as f:
    LBC_2_cohort = pk.load(f)

# merge LBC cohorts
LBC_cohort = [part[part.obs.LiFT_value>1] for part in LBC_1_cohort + LBC_2_cohort]
LBC_cohort = [part for part in LBC_cohort if part.shape[0]>0]

# merge cohorts
cohort = WHI_cohort + sardinia_cohort + LBC_cohort

processed_part_list = []
for i, part in enumerate(cohort):
    print(f'Participant {i} of {len(cohort)}')
    
    # Vectorised clonal inference
    part = compute_clonal_models_prob_vec(part)
    part = refine_optimal_model_posterior_vec(part, 201)
    
    # Append processed results
    processed_part_list.append(part)


# export participatn_data
with open('../exports/all_cohorts_fitted.pk', 'wb') as f:
    pk.dump(processed_part_list, f)