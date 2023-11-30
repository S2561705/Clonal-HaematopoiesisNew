# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from functools import partial

with open('../exports/fabre.pk', 'rb') as f:
    participant_list = pk.load(f)

# Plot sample participant
part = participant_list[4]
plot_part(part)

# Global computations
# %%

def jax_cs_hmm_ll_vec(s_vec, AO, DP, 
                      time_points, cs, 
                      deterministic_size,
                      total_cells):

    global_variables = compute_global_variables(s_vec, AO, DP, total_cells, deterministic_size, time_points)
    x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mutations = global_variables
    
    # create s indexes for vectorisation
    s_idx = jnp.arange(s_vec.shape[0])
    
    clonal_likelihood = jax.vmap(fitness_specific_computations,
                                 in_axes=(0, None, None, None, None, None, None, None, None))(
                                 s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec,
                                 p_y_cond_x_vec, time_points, n_mutations, cs)
    
    return clonal_likelihood

@jit
def compute_global_variables(s_vec, AO, DP, 
                             total_cells, deterministic_size,
                             time_points,
                             resolution=1_000):
    
    # number of mutations in the participant
    n_mutations = AO.shape[1]

    # BD process exponential term for pmf
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t*s_vec[:, None])
    exp_term_vec_s = np.reshape(exp_term_vec_s, (*exp_term_vec_s.shape,1, 1))

    beta_p_rvs_vec = jrnd.beta(key=key, a=(AO+1)[:, :, None],
                    b=DP[:, :, None]- AO[:, :, None] + 1,
                    shape=(AO.shape[0], AO.shape[1], resolution))

    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    x_range_vec = jnp.array(-N_w_cond_vec*beta_p_rvs_vec / (beta_p_rvs_vec-0.5))
    x_vec = jnp.ceil(x_range_vec)

    # add endpoints right endpoint is set to 2*max_x
    true_vaf_vec = x_vec/(2*(N_w_cond_vec+x_vec))

    # compute conditional observation probabilities over all time points
    p_y_cond_x_vec = jsp_stats.binom.pmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)

    # Compute probability of first timepoint
    recursive_term_vec = p_y_cond_x_vec[0, :, :]*1/resolution

    return (x_vec, exp_term_vec_s, recursive_term_vec, 
            p_y_cond_x_vec, n_mutations)

@jit
def BD_process_dynamics (s, x_vec, exp_term_vec):
    lamb=1.3
    # Vectorised computation of BD process dynamics
    mean_vec = x_vec[:-1, : , :]*exp_term_vec
    variance_vec = x_vec[:-1]*(2*lamb + s)*exp_term_vec*(exp_term_vec-1)/s

    # Neg Binom parametrization
    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec-mean_vec) 

    return p_vec, n_vec

def fitness_specific_computations (s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, time_points, n_mutations, cs):
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]

    # Vectorised computation of BD process dynamics
    p_vec, n_vec = BD_process_dynamics(s,x_vec, exp_term_vec)    

    mutation_likelihood = jax.vmap(mutation_specific_ll,
                                in_axes=(0, None, None, None, None, None, None))(
                                        jnp.arange(n_mutations, dtype=int),
                                        recursive_term_vec, x_vec, p_vec,
                                        n_vec, p_y_cond_x_vec, time_points.shape[0])

    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)]))

    return clonal_likelihood

def mutation_specific_ll(i, recursive_term_vec, x_vec, p_vec,
                         n_vec, p_y_cond_x_vec, n_tps):
    
    # locate mutation specific data
    recursive_term_i = recursive_term_vec[i]
    x_i = x_vec[:, i]
    p_i = p_vec[:, i]
    n_i = n_vec[:, i]
    p_y_cond_x_i = p_y_cond_x_vec[:, i]

    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i)

    return jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)

@jit
def recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i):
    """Update the recursive term associated with mutation i for data point j"""

    bd_pmf_i = jsp_stats.nbinom.pmf(x_i[j][:, None], p=p_i[j-1], n=n_i[j-1])
    inner_sum_i = bd_pmf_i*recursive_term_i
    recursive_term_i = p_y_cond_x_i[j]*jsp.integrate.trapezoid(x=x_i[j-1], y=inner_sum_i)

    return recursive_term_i

def compute_clonal_models_prob_vec (part, s_resolution=50):
    """Compute model probabilities for each clonal structure.
    Append results in unstructued dictionary."""
    
    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)
    jnp.diff(time_points)
    s_vec = jnp.linspace(0.01, 1, s_resolution)

    # Compute all possible clonal structures (or partition sets) 
    n_mutations = part.shape[0]
    a = partition(list(range(n_mutations)))
    part.uns['model_dict'] = {}    

    cs_list = [cs for cs in a]
    # Compute clonal structure probability
    for i, cs in tqdm(enumerate(cs_list), total=len(cs_list)):  
        deterministic_size, total_cells = compute_deterministic_size(cs, AO, DP, AO.shape[1])

        # compute clonal posteriors
        output = jax_cs_hmm_ll_vec(s_vec, AO, DP, 
                                   time_points, cs, 
                                   deterministic_size,
                                   total_cells)

        # compute clonal structure probability
        model_prob = compute_model_likelihood(output, cs, s_vec)

        # save model probability
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob) 

    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), key=lambda item: item[1][1], reverse=True)}

    return part

def refine_optimal_model_posterior_vec(part, s_resolution=100):
    # Compute finer posterior for optimal model
    # retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]
    n_mutations = part.shape[0]

    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    # compute deterministic clone sizes
    deterministic_size, total_cells = compute_deterministic_size(cs, AO, DP, AO.shape[1])

    # Create refined s_range
    s_vec = jnp.linspace(0.01, 1, s_resolution)

   # compute clonal posteriors
    output = jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, deterministic_size, total_cells)

    part.uns['optimal_model'] = {'clonal_structure': cs,
                                'mutation_structure': [list(part.obs.iloc[cs_idx]['PreferredSymbol']) for cs_idx in cs],
                                'posterior': output,
                                's_range': s_vec}

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):

        p = np.array(output[:, i])
        # normalise output
        p/=p.sum()     
        sample_range = np.random.choice(s_vec, p=p, size=1_000)
        fitness_map = np.argmax(p)
        cfd_int = np.quantile(sample_range, [0.05, 0.95])

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = cfd_int[0]
        fitness_95[c_idx] = cfd_int[1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index
    return part

# %%
%%time
part = compute_clonal_models_prob_vec(part)
part = refine_optimal_model_posterior_vec(part, 201)

plot_part(part)
plot_optimal_model(part)

part.uns['model_dict']
# %%