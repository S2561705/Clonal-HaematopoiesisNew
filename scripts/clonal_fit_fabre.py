# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

with open('../exports/fabre.pk', 'rb') as f:
    participant_list = pk.load(f)

sns.histplot(x=[part.shape[0] for part in participant_list])


# %%
total=0
len(participant_list)
for part in participant_list:
    if part.shape[0]==4:
        total+=1
        plot_part(part)
        print(part.obs)
        print(part.layers['DP'])

participant_list[0].obs
part = participant_list[4]
part.obs.PreferredSymbol.values
np.corrcoef(part.X)
sns.scatterplot(x=part.X[1], y=part.X[2])
compute_invalid_combinations(part)

# %%
s_resolution = 50
 # Extract participant features
AO = jnp.array(part.layers['AO'].T)
DP = jnp.array(part.layers['DP'].T)
time_points = jnp.array(part.var.time_points)
s_vec = jnp.linspace(0.01, 1, s_resolution)

# Compute all possible clonal structures (or partition sets) 
n_mutations = part.shape[0]
a = partition(list(range(n_mutations)))
part.uns['model_dict'] = {}    

total=0
for cs in a:
    if len([i for i in cs if i in not_valid_comb]) >0:
        continue
    total += 1
# %%
s= 0
x = np.arange(10)
y = np.exp(s*x)
sns.lineplot(x=x, y=y)
# %%
