# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

with open('../exports/fabre.pk', 'rb') as f:
    participant_list = pk.load(f)

# Plot sample participant
part = participant_list[4]
plot_part(part)

# %%
%%time
part = compute_clonal_models_prob(part)
part = refine_optimal_model_posterior(part, 201)

plot_part(part)
plot_optimal_model(part)

# %%

part.uns['model_dict']