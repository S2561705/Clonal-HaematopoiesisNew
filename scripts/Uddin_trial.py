# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

with open('../exports/Uddin.pk', 'rb') as f:
    participant_list = pk.load(f)

sns.histplot(x=[part.shape[0] for part in participant_list])
plt.show()
plt.clf()

part_less_5 = [part for part in participant_list if part.shape[0]<5]
part_more_5 = [part for part in participant_list if part.shape[0]>=5]

# Valid clonal models
sns.histplot(x=[len(find_valid_clonal_structures(part, p_distance_threshold=0.5)) for part in participant_list])
plt.show()


# %%

part = part_more_5[6]

for part in part_more_5:
    plot_part(part)

    slopes = np.diff(part.X)/np.diff(part.var.time_points)
    AF = part.X[:, -1]

    fitness_proxy = (slopes.T/AF).mean(axis=0)






    plt.clf()

# %%

# label invalid combinations if pearson correlation is too different
not_valid_comb = np.argwhere(distance_matrix
                                >0.5)

not_valid_comb = [list(i) for i in not_valid_comb]

# Extract unique tuples from list(Order Irrespective)
# using list comprehension + set()
res = []
for i in not_valid_comb:
    if [i[0], i[1]] and [i[1], i[0]] not in res:
        res.append(i) 

# %% 
part = participant_list[0]
plot_part(part)

# %%
# Vectorised
part = compute_clonal_models_prob_vec(part)
part = refine_optimal_model_posterior_vec(part, 201)

plot_optimal_model(part)
# %%

part = part_less_5[1]
plot_part(part)

part.layers['DP']

# Find valid clonal structures
find_valid_clonal_structures(part)
# %%
# Vectorised
part = compute_clonal_models_prob_vec(part)
part = refine_optimal_model_posterior_vec(part, 201)
plot_optimal_model(part)

# %%
