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
part = participant_list[3]
plot_part(part)

# %%
# Example of single clonal structure posterior

cs = [[0, 1], [2]]
s = jnp.array([0.2, 0.01])
AO = jnp.array(part.layers['AO'].T)
DP = jnp.array(part.layers['DP'].T)
time_points=jnp.array(part.var.time_points)

def compute_deterministic_size(cs, AO, DP, n_mutations):
    # Deterministic size of clones
    # s independent, but cs dependent
    N_w = 1e5

    # determine leading mutation for each clone
    lm = jnp.zeros(len(cs), dtype=int)         
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        lm = lm.at[i].set(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    deterministic_clone_size = jnp.array(-N_w*jnp.sum((AO/DP)[:, lm], axis=1) 
                                            / (jnp.sum((AO/DP)[:, lm], axis=1)-0.5))
   
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO/DP*2*total_cells[:, None]

    return deterministic_size, total_cells

s_clone=0.2
def jax_parallel_hmm_ll(s, AO, DP, time_points,
                  cs,
                  lamb=1.3):
    
    n_mutations = AO.shape[1]
    mutation_likelihood = jnp.zeros(n_mutations)
    
    deterministic_size, total_cells = compute_deterministic_size(cs, AO, DP, n_mutations)
j=0
    for j in range(n_mutations):
        beta_p_rvs = jrnd.beta(key=key, a=AO[:,j][:, None]+1,
                            b=DP[:,j][:, None]- AO[:,j][:, None] + 1,
                            shape=(AO.shape[0], 1_000))

        beta_p_rvs = jnp.sort(beta_p_rvs)

        N_w_cond = (total_cells - deterministic_size[:, j])[:, None]
        x_range = jnp.array(-N_w_cond*beta_p_rvs / (beta_p_rvs-0.5))
        x = jnp.ceil(x_range)
        # add endpoints right endpoint is set to 2*max_x
        # x = jnp.c_[jnp.ones(x.shape[0]), x, x[:,-1]*2] 
        true_vaf = x/(2*(N_w_cond+x))

        # # computation of BD process 
        delta_t = jnp.diff(jnp.array(time_points))
        # exp_term = jnp.exp(delta_t*s_clone[j])

        x_weight = jnp.repeat(1/x[0].shape[0], x[0].shape[0])
        recursive_term = jsp_stats.binom.pmf(AO[0, j], n=DP[0,j], p=true_vaf[0])*x_weight

i=1
        for i in range(1, x.shape[0]):
            init_size = x[i-1]
            next_size = x[i]
            p_y_cond_x = jsp_stats.binom.pmf(AO[i, j], n=DP[i, j], p=true_vaf[i])


            # Predict pmf of BD process
            exp_term = jnp.exp(delta_t[i-1]*s_clone)
            mean = init_size*exp_term
            variance = init_size*(2*lamb + s_clone)*exp_term*(exp_term-1)/s_clone
            # Neg Binom parametrization
            p = mean / variance
            n = jnp.power(mean, 2) / (variance-mean)  

            bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)

            sns.lineplot(x=next_size, y=bd_pmf[0,:])
            inner_sum = bd_pmf*recursive_term
            recursive_term = p_y_cond_x*jsp.integrate.trapezoid(x=x[i-1], y=inner_sum)

        likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
        mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)]))

    return clonal_likelihood


def compute_deterministic_size_app(cs, AO, DP, n_mutations):
    # Deterministic size of clones
    # s independent, but cs dependent
    N_w = 1e5

    # determine leading mutation for each clone
    lm = []        
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    deterministic_clone_size = jnp.array(-N_w*jnp.sum((AO/DP)[:, lm], axis=1) 
                                            / (jnp.sum((AO/DP)[:, lm], axis=1)-0.5))
   
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO/DP*2*total_cells[:, None]

    return deterministic_size, total_cells