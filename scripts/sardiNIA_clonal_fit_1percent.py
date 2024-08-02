import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pickle

with open('../exports/sardiNIA/sardiNIA_1percent.pk', 'rb') as f:
    cohort = pk.load(f)

processed_part_list = []
for i, part in enumerate(cohort):
    print(f'Participant {i} of {len(cohort)}')
    
    # Vectorised clonal inference
    part = compute_clonal_models_prob_vec(part)
    part = refine_optimal_model_posterior_vec(part, 201)
    
    # Append processed results
    processed_part_list.append(part)

# Export results
with open('../exports/sardiNIA/sardiNIA_1percent_fitted.pk', 'wb') as f:
    pk.dump(processed_part_list, f)