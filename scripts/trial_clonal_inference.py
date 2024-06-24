# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

with open('../exports/Uddin.pk', 'rb') as f:
    participant_list = pk.load(f)

# %%
# Plot sample participant
part = participant_list[3]
plot_part(part)

sns.histplot(x=[part.shape[0] for part in participant_list])
plt.clf()

for part in participant_list:
    if part.shape[0]==3:
        plot_part(part)
        break

np.mean([part.X.mean() for part in participant_list])

# %%
%%time

# Non-vectorised code
part = compute_clonal_models_prob(part)
part = refine_optimal_model_posterior(part, 201)

plot_part(part)
plot_optimal_model(part)

part.uns['model_dict']

# %%
%%time

# Vectorised
part = compute_clonal_models_prob_vec(part)
part = refine_optimal_model_posterior_vec(part, 201)

plot_part(part)
plot_optimal_model(part)

# part.uns['model_dict']
# %%


# Compute all possible clonal structures (or partition sets) 
n_mutations = part.shape[0]
a = partition(list(range(n_mutations)))
part.uns['model_dict'] = {}    

cs = [[0,1], [2]]
i = 1 
s_resolution = 50
s_range = jnp.linspace(0.01,1, s_resolution)
multi_s_range =  jnp.broadcast_to(s_range, (len(cs), s_resolution)).T

AO = part.layers['AO'].T
DP = part.layers['DP'].T

N_w = 1e5
n_mutations = AO.shape[1]

# determine leading mutation for each clone
leading_mutation_in_cs_idx = []         
clonal_map = jnp.zeros(n_mutations, dtype=int)


for i, cs_idx in enumerate(cs):

    # Sum over time points to find biggest overall mutation (summing over time points)
    # argamx detects leading mutation
    max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
    leading_mutation_in_cs_idx.append(cs_idx[max_idx])
    clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

mutation_likelihood = jnp.zeros(n_mutations)

deterministic_clone_size = jnp.array(-N_w*jnp.sum((AO/DP)[:, leading_mutation_in_cs_idx], axis=1) 
                                        / (jnp.sum((AO/DP)[:, leading_mutation_in_cs_idx], axis=1)-0.5))

deterministic_clone_size = jnp.ceil(deterministic_clone_size)
total_cells = N_w + deterministic_clone_size


deterministic_size = AO/DP*2*total_cells[:, None]

j = 0

s = multi_s_range
s_clone = s[clonal_map[j]]


beta_p_rvs = jrnd.beta(key=key, a=AO[:,j][:, None]+1,
                    b=DP[:,j][:, None]- AO[:,j][:, None] + 1,
                    shape=(AO.shape[0], 1_000))


# sort sampled vafs  (for integration purposes)
beta_p_rvs = jnp.sort(beta_p_rvs)

# Wild type cells plus other clones
N_w_cond = (total_cells - deterministic_size[:, j])[:, None]
x_range = jnp.array(-N_w_cond*beta_p_rvs / (beta_p_rvs-0.5))
x = jnp.ceil(x_range)
delta_t = jnp.diff(jnp.array(part.var.time_points))
true_vaf = x/(2*(N_w_cond+x))

# 
x_weight = jnp.repeat(1/x[0].shape[0], x[0].shape[0])
recursive_term = jsp_stats.binom.pmf(AO[0, j], n=DP[0,j], p=true_vaf[0])*x_weight

for i in range(1, x.shape[0]):
    init_size = x[i-1]
    next_size = x[i]
    p_y_cond_x = jsp_stats.binom.pmf(AO[i, j], n=DP[i, j], p=true_vaf[i])

    # Predict pmf of BD process
    exp_term = jnp.exp(delta_t[i-1]*s_clone)
    mean = init_size*exp_term
    variance = init_size*(2*1.3 + s_clone)*exp_term*(exp_term-1)/s_clone
    
    # Neg Binom parametrization
    p = mean / variance
    n = jnp.power(mean, 2) / (variance-mean)  

    bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)

    inner_sum = bd_pmf*recursive_term
    recursive_term = p_y_cond_x*jsp.integrate.trapezoid(x=x[i-1], y=inner_sum)


likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

clonal_likelihood = jnp.zeros(len(cs))
for i, c_idx in enumerate(cs):
    clonal_likelihood = clonal_likelihood.at[i].set(
        np.prod(mutation_likelihood[jnp.array(c_idx)]))