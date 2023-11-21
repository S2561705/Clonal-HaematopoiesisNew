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
part = participant_list[2]
plot_part(part)

# %%
# Example of single clonal structure posterior

# specify clonal structure
cs = [[0], [1], [2]]

# Compute posterior
output, s_range = single_cs_posterior(part, cs=cs, s_resolution=50)

# Plot posterior
for i, out in enumerate(output.T):
    sns.lineplot(x=s_range, y=out/np.max(out),
                 label=part.obs.iloc[cs[i]]['PreferredSymbol'].values)

# %%
part = compute_clonal_models_prob(part)
part = refine_optimal_model_posterior(part, 201)

plot_part(part)
plot_optimal_model(part)

# %%
