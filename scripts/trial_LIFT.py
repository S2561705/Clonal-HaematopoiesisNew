%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

with open('../exports/fabre.pk', 'rb') as f:
    participant_list = pk.load(f)

for part in participant_list:
    print(part)


part = participant_list[0]

cs = [[i] for i in range(part.shape[0])]
n_mutations = part.shape[0]

# Extract participant features
AO = jnp.array(part.layers['AO'].T)
DP = jnp.array(part.layers['DP'].T)
time_points = jnp.array(part.var.time_points)

# compute deterministic clone sizes
deterministic_size, total_cells = compute_deterministic_size(cs, AO, DP, AO.shape[1])

# Create refined s_range
s_vec = jnp.linspace(0.001, 1, 50)

# compute clonal posteriors
output = jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, deterministic_size, total_cells)
