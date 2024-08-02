import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pickle

with open('../exports/sardiNIA/sardiNIA.pk', 'rb') asa f:
    cohort = pk.load(f)

processed_part_list = []
for i, part in enumerate(cohort[113:]):
    print(f'Participant {i} of {len(cohort[113:])}')
    
    # Vectorised clonal inference
    part = compute_clonal_models_prob_vec(part)
    part = refine_optimal_model_posterior_vec(part, 201)
    
    # Append processed results
    processed_part_list.append(part)

# Export results
with open('../exports/sardiNIA/sardiNIA_fitted.pk', 'wb') as f:
    pk.dump(processed_part_list, f)