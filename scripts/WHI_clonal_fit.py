import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pickle

with open('../exports/WHI/WHI.pk', 'rb') as f:
    participant_list = pk.load(f)

processed_part_list = []
for i, part in enumerate(participant_list):
    print(f'Participant {i} of {len(participant_list)}')
    
    # Vectorised clonal inference
    part = compute_clonal_models_prob_vec(part)
    part = refine_optimal_model_posterior_vec(part, 201)
    
    # Append processed results
    processed_part_list.append(part)

# Export results
with open('../exports/WHI/WHI_fitted.pk', 'wb') as f:
    pk.dump(processed_part_list, f)

# %%
