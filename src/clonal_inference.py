import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd

key = jrnd.PRNGKey(758493)  # Random seed is explicit in JAX
def jax_cs_hmm_ll(s, AO, DP, time_points,
                  cs,
                  lamb=1.3):
    N_w = 1e5
    n_mutations = AO.shape[1]

    # determine leading mutation for each clone
    lm = []         
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    mutation_likelihood = jnp.zeros(n_mutations)

    deterministic_clone_size = jnp.array(-N_w*jnp.sum((AO/DP)[:, lm], axis=1) 
                                            / (jnp.sum((AO/DP)[:, lm], axis=1)-0.5))
   
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO/DP*2*total_cells[:, None]

    for j in range(n_mutations):
        s_clone = s[clonal_map[j]]
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

            inner_sum = bd_pmf*recursive_term
            recursive_term = p_y_cond_x*jsp.integrate.trapezoid(x=x[i-1], y=inner_sum)

        likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
        mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)]))

    return clonal_likelihood


def partition(collection):
    """Module computing an iterable over all partitions of a set"""
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def single_cs_posterior(part, cs, s_resolution=100):
    """Compute posterior for a single clonal structure"""
    
    # create finer s array
    s_range = jnp.linspace(0.01,1, s_resolution)
    multi_s_range =  jnp.broadcast_to(s_range, (len(cs), s_resolution)).T
    
    output = jax.vmap(jax_cs_hmm_ll, in_axes=(0, None, None, None, None, None))(multi_s_range, part.layers['AO'].T, part.layers['DP'].T, part.var.time_points,cs, 1.3)
    
    return output, s_range

def compute_clonal_models_prob (part, s_resolution=50):
    """Compute model probabilities for each clonal structure.
    Append results in unstructued dictionary."""
    
    # Compute all possible clonal structures (or partition sets) 
    n_mutations = part.shape[0]
    a = partition(list(range(n_mutations)))
    part.uns['model_dict'] = {}    

    for i, cs in tqdm(enumerate(a)):    

        output, s_range = single_cs_posterior(part, cs, s_resolution)
        model_prob = compute_model_likelihood(output, cs, s_range)

        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob) 

    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), key=lambda item: item[1][1], reverse=True)}

    return part


def compute_model_likelihood(output, cs, s_range):
    model_prob = np.zeros(len(cs))
    for i, out in enumerate(output.T):
        model_prob[i] = np.trapz(x=s_range, y=out)

    return np.prod(model_prob)


def refine_optimal_model_posterior(part, s_resolution=100):
    # Compute finer posterior for optimal model
    # retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]
    output, s_range = single_cs_posterior(part, cs, s_resolution)
    part.uns['optimal_model'] = {'clonal_structure': cs,
                                'mutation_structure': [list(part.obs.iloc[cs_idx]['PreferredSymbol']) for cs_idx in cs],
                                'posterior': output,
                                's_range': s_range}
    return part


def plot_optimal_model(part):
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    ms = model['mutation_structure']
    s_range = model['s_range']

    # normalisation constant
    norm_max = np.max(output, axis=0)
    # Plot
    for i in range(len(cs)):
        sns.lineplot(x=s_range, y=output[:, i]/ norm_max[i], label=f'clone {ms[i]}')

    plt.show()