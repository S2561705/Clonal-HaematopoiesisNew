import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd
from itertools import combinations


key = jrnd.PRNGKey(758493)  # Random seed is explicit in JAX

#region Non vectorised functions with DEBUG
# Non vectorised 
def compute_deterministic_size_mixed(cs, AO, DP, n_mutations, N_w=1e5):
    """
    Produce a crude deterministic estimate of TOTAL mutant cells per timepoint & mutation,
    plus a recommended max_total per mutation for sampling (used as upper bound).
    We do NOT enforce het/hom split here — that will be sampled explicitly.
    """
    # print(f"\n=== compute_deterministic_size_mixed ===")
    # print(f"Input: n_mutations={n_mutations}, N_w={N_w}")
    # print(f"AO shape: {AO.shape}, DP shape: {DP.shape}")
    # print(f"AO values:\n{AO}")
    # print(f"DP values:\n{DP}")
    
    # Calculate VAFs and check for issues
    vaf_ratio = AO / DP
    # print(f"VAFs (AO/DP):\n{vaf_ratio}")
    # print(f"VAF range: [{vaf_ratio.min():.3f}, {vaf_ratio.max():.3f}]")
    
    # Find leading mutation per clone (same as before but ensure int index)
    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        # print(f"Processing clone {i} with mutations: {cs_idx}")
        clone_vafs = vaf_ratio[:, cs_idx]
        vaf_sums = clone_vafs.sum(axis=0)
        # print(f"  VAF sums for mutations in clone: {vaf_sums}")
        max_idx = int(jnp.argmax(vaf_sums))
        lm.append(cs_idx[max_idx])
        # print(f"  Leading mutation: {cs_idx[max_idx]} (index {max_idx})")
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    # print(f"Leading mutations: {lm}")
    # print(f"Clonal map: {clonal_map}")

    # As before, use heterozygous rearrangement to estimate total mutant cells,
    # but this is only a heuristic upper bound.
    vafs_lead = vaf_ratio[:, lm]                # (n_timepoints, n_clones)
    # print(f"Leading mutation VAFs:\n{vafs_lead}")
    
    sum_vafs_lead = jnp.sum(vafs_lead, axis=1)
    denominator = jnp.sum(vafs_lead, axis=1) - 0.5
    # print(f"Sum of leading VAFs per timepoint: {sum_vafs_lead}")
    # print(f"Denominator (sum_vafs - 0.5): {denominator}")
    
    # Check for problematic denominators
    problematic_denom = jnp.abs(denominator) < 1e-8
    if jnp.any(problematic_denom):
        # print(f"WARNING: Small denominators detected at indices: {jnp.where(problematic_denom)[0]}")
        denominator = jnp.where(problematic_denom, jnp.sign(denominator) * 0.1, denominator)
    
    deterministic_clone_size = -N_w * sum_vafs_lead / denominator
    # print(f"Raw deterministic_clone_size: {deterministic_clone_size}")
    
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)       # (n_clones,)
    # print(f"Ceiled deterministic_clone_size: {deterministic_clone_size}")

    total_cells = N_w + deterministic_clone_size
    # print(f"Total cells (N_w + clone_size): {total_cells}")

    # Deterministic SIZE grid for each mutation (heuristic)
    # Use observed VAFs scaled to the total_cells as a centre estimate
    deterministic_size = vaf_ratio * 2 * total_cells[:, None]   # (n_timepoints, n_mutations)
    # print(f"Deterministic size shape: {deterministic_size.shape}")
    # print(f"Deterministic size range: [{deterministic_size.min():.1f}, {deterministic_size.max():.1f}]")

    # Compute an upper bound for sampling total mutant cells per mutation:
    # e.g. 3x the deterministic estimate (safe margin)
    max_total_per_mutation = jnp.maximum(deterministic_size.max(axis=0) * 3, 10.0)
    # print(f"Max total per mutation: {max_total_per_mutation}")

    return deterministic_size.astype(jnp.float32), total_cells.astype(jnp.float32), max_total_per_mutation.astype(jnp.float32), clonal_map


def jax_cs_hmm_ll(s, AO, DP, time_points, cs, lamb=1.3):
    # print(f"\n=== jax_cs_hmm_ll ===")
    # print(f"s: {s}")
    # print(f"AO shape: {AO.shape}, DP shape: {DP.shape}")
    # print(f"time_points: {time_points}")
    # print(f"clonal structure: {cs}")
    # print(f"lambda: {lamb}")
    
    N_w = 1e5
    n_mutations = AO.shape[1]
    # print(f"N_w: {N_w}, n_mutations: {n_mutations}")

    # determine leading mutation for each clone
    leading_mutation_in_cs_idx = []         
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    vaf_ratio = AO / DP
    # print(f"VAF ratios shape: {vaf_ratio.shape}")

    for i, cs_idx in enumerate(cs):
        # print(f"Processing clone {i} with mutations: {cs_idx}")
        clone_vafs = vaf_ratio[:, cs_idx]
        vaf_sums = clone_vafs.sum(axis=0)
        # print(f"  VAF sums: {vaf_sums}")
        max_idx = jnp.argmax(vaf_sums)
        leading_mutation_in_cs_idx.append(cs_idx[max_idx])
        # print(f"  Leading mutation: {cs_idx[max_idx]}")
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    # print(f"Leading mutations: {leading_mutation_in_cs_idx}")
    # print(f"Clonal map: {clonal_map}")

    mutation_likelihood = jnp.zeros(n_mutations)
    # print(f"Initialized mutation_likelihood: {mutation_likelihood}")

    # Calculate deterministic clone sizes
    lead_vafs = vaf_ratio[:, leading_mutation_in_cs_idx]
    sum_lead_vafs = jnp.sum(lead_vafs, axis=1)
    denominator = jnp.sum(lead_vafs, axis=1) - 0.5
    
    # print(f"Leading VAFs shape: {lead_vafs.shape}")
    # print(f"Sum lead VAFs: {sum_lead_vafs}")
    # print(f"Denominator: {denominator}")
    
    # Fix division issues
    denominator = jnp.where(jnp.abs(denominator) < 0.1, jnp.sign(denominator) * 0.1, denominator)
    
    deterministic_clone_size = jnp.array(-N_w * sum_lead_vafs / denominator)
    # print(f"Raw deterministic_clone_size: {deterministic_clone_size}")
   
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    # print(f"Ceiled deterministic_clone_size: {deterministic_clone_size}")
    
    total_cells = N_w + deterministic_clone_size
    # print(f"Total cells: {total_cells}")

    deterministic_size = vaf_ratio * 2 * total_cells[:, None]
    # print(f"Deterministic size shape: {deterministic_size.shape}")

    # Loop through each mutation
    for j in range(n_mutations):
        # print(f"\n--- Processing mutation {j} ---")
        s_clone = s[clonal_map[j]]
        # print(f"Clone fitness s: {s_clone}")

        # Sample hidden Vaf sizes for all time points
        a_params = AO[:, j][:, None] + 1
        b_params = DP[:, j][:, None] - AO[:, j][:, None] + 1
        # print(f"Beta params - a: {a_params.flatten()}, b: {b_params.flatten()}")
        
        beta_p_rvs = jrnd.beta(key=key, a=a_params, b=b_params, shape=(AO.shape[0], 1_000))
        # print(f"Beta samples shape: {beta_p_rvs.shape}")
        # print(f"Beta samples range: [{beta_p_rvs.min():.3f}, {beta_p_rvs.max():.3f}]")

        # sort sampled vafs (for integration purposes)
        beta_p_rvs = jnp.sort(beta_p_rvs)
        # print(f"Sorted beta samples range: [{beta_p_rvs.min():.3f}, {beta_p_rvs.max():.3f}]")

        # Transform VAFs to clone sizes
        N_w_cond = (total_cells - deterministic_size[:, j])[:, None]
        # print(f"N_w_cond shape: {N_w_cond.shape}, range: [{N_w_cond.min():.1f}, {N_w_cond.max():.1f}]")

        # VAF to Size transformation - CHECK FOR DIVISION BY ZERO
        denominator_vaf = beta_p_rvs - 0.5
        zero_denom_mask = jnp.abs(denominator_vaf) < 1e-8
        if jnp.any(zero_denom_mask):
            # print(f"WARNING: {jnp.sum(zero_denom_mask)} near-zero denominators in VAF transformation")
            denominator_vaf = jnp.where(zero_denom_mask, jnp.sign(denominator_vaf) * 0.1, denominator_vaf)
        
        x_range = jnp.array(-N_w_cond * beta_p_rvs / denominator_vaf)
        x = jnp.ceil(x_range)
        # print(f"x shape: {x.shape}, range: [{x.min():.1f}, {x.max():.1f}]")

        # Calculate true VAF - CHECK FOR DIVISION BY ZERO
        denominator_vaf2 = 2 * (N_w_cond + x)
        zero_denom_mask2 = denominator_vaf2 == 0
        if jnp.any(zero_denom_mask2):
            # print(f"WARNING: {jnp.sum(zero_denom_mask2)} zero denominators in true VAF calculation")
            denominator_vaf2 = jnp.where(zero_denom_mask2, 1.0, denominator_vaf2)
            
        true_vaf = x / denominator_vaf2
        # print(f"true_vaf shape: {true_vaf.shape}, range: [{true_vaf.min():.3f}, {true_vaf.max():.3f}]")

        # Computation of BD process 
        delta_t = jnp.diff(jnp.array(time_points))
        # print(f"delta_t: {delta_t}")

        x_weight = jnp.repeat(1/x[0].shape[0], x[0].shape[0])
        # print(f"x_weight: {x_weight.shape}, sum: {x_weight.sum()}")

        # Initial recursive term
        recursive_term = jsp_stats.binom.pmf(AO[0, j], n=DP[0, j], p=true_vaf[0]) * x_weight
        # print(f"Initial recursive_term shape: {recursive_term.shape}")
        # print(f"Initial recursive_term sum: {recursive_term.sum():.3e}")

        for i in range(1, x.shape[0]):
            # print(f"  Timepoint transition {i-1} -> {i}")
            init_size = x[i-1]
            next_size = x[i]
            p_y_cond_x = jsp_stats.binom.pmf(AO[i, j], n=DP[i, j], p=true_vaf[i])
            # print(f"    p_y_cond_x shape: {p_y_cond_x.shape}, sum: {p_y_cond_x.sum():.3e}")

            # Predict pmf of BD process
            exp_term = jnp.exp(delta_t[i-1] * s_clone)
            mean = init_size * exp_term
            variance = init_size * (2*lamb + s_clone) * exp_term * (exp_term-1) / jnp.maximum(s_clone, 1e-8)
            
            # print(f"    exp_term: {exp_term:.3f}")
            # print(f"    mean range: [{mean.min():.1f}, {mean.max():.1f}]")
            # print(f"    variance range: [{variance.min():.1f}, {variance.max():.1f}]")

            # Check for invalid negative binomial parameters
            invalid_variance = variance <= mean
            if jnp.any(invalid_variance):
                # print(f"    WARNING: {jnp.sum(invalid_variance)} invalid variances (variance <= mean)")
                variance = jnp.where(invalid_variance, mean * 1.1, variance)
            
            # Neg Binom parametrization
            p = mean / variance
            n = jnp.power(mean, 2) / jnp.maximum(variance - mean, 1e-8)
            
            # print(f"    p range: [{p.min():.3f}, {p.max():.3f}]")
            # print(f"    n range: [{n.min():.1f}, {n.max():.1f}]")

            bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)
            # print(f"    bd_pmf shape: {bd_pmf.shape}, sum: {bd_pmf.sum():.3e}")

            inner_sum = bd_pmf * recursive_term
            # print(f"    inner_sum shape: {inner_sum.shape}, sum: {inner_sum.sum():.3e}")

            recursive_term = p_y_cond_x * jsp.integrate.trapezoid(x=x[i-1], y=inner_sum)
            # print(f"    New recursive_term sum: {recursive_term.sum():.3e}")

        likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
        # print(f"Final likelihood for mutation {j}: {likelihood:.3e}")
        mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

    # print(f"\nAll mutation likelihoods: {mutation_likelihood}")
    
    # Compute clonal likelihoods
    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clone_mutations = jnp.array(c_idx)
        clone_likelihood = jnp.prod(mutation_likelihood[clone_mutations])
        clonal_likelihood = clonal_likelihood.at[i].set(clone_likelihood)
        # print(f"Clone {i} (mutations {c_idx}) likelihood: {clone_likelihood:.3e}")

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
    # print(f"\n=== single_cs_posterior ===")
    # print(f"Clonal structure: {cs}")
    # print(f"s_resolution: {s_resolution}")
    
    # create finer s array
    s_range = jnp.linspace(0.01, 1, s_resolution)
    multi_s_range = jnp.broadcast_to(s_range, (len(cs), s_resolution)).T
    # print(f"s_range: {s_range}")
    # print(f"multi_s_range shape: {multi_s_range.shape}")
    
    print("Computing likelihoods for all s values...")
    output = jax.vmap(jax_cs_hmm_ll, in_axes=(0, None, None, None, None, None))(
        multi_s_range, part.layers['AO'].T, part.layers['DP'].T, part.var.time_points, cs, 1.3)
    
    # print(f"Output shape: {output.shape}")
    # print(f"Output range: [{output.min():.3e}, {output.max():.3e}]")
    
    return output, s_range

def compute_clonal_models_prob(part, s_resolution=50):
    """Compute model probabilities for each clonal structure.
    Append results in unstructued dictionary."""
    # print(f"\n" + "="*60)
    print("COMPUTING CLONAL MODELS PROBABILITIES")
    print("="*60)
    
    # Compute all possible clonal structures (or partition sets) 
    n_mutations = part.shape[0]
    # print(f"Number of mutations: {n_mutations}")
    
    a = list(partition(list(range(n_mutations))))
    # print(f"Total number of possible clonal structures: {len(a)}")
    
    part.uns['model_dict'] = {}    

    for i, cs in enumerate(a):    
        # print(f"\n--- Evaluating model {i} ---")
        # print(f"Clonal structure: {cs}")
        
        output, s_range = single_cs_posterior(part, cs, s_resolution)
        model_prob = compute_model_likelihood(output, cs, s_range)

        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)
        # print(f"Model {i} probability: {model_prob:.3e}")

    # Sort models by probability
    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), key=lambda item: item[1][1], reverse=True)}
    
    # print(f"\n" + "="*60)
    print("MODEL RANKING:")
    print("="*60)
    for i, (k, v) in enumerate(part.uns['model_dict'].items()):
        cs, prob = v
        # print(f"Rank {i}: {k} - Probability: {prob:.3e}")
        # print(f"       Structure: {cs}")
    
    return part

def compute_model_likelihood(output, cs, s_range):
    # print(f"\n=== compute_model_likelihood ===")
    # print(f"Output shape: {output.shape}")
    # print(f"Clonal structure: {cs}")
    # print(f"s_range: {s_range}")
    
    # initialise clonal probability
    clonal_prob = np.zeros(len(cs))

    s_range_size = s_range.max() - s_range.min()
    s_prior = 1/s_range_size
    # print(f"s_range_size: {s_range_size}, s_prior: {s_prior}")
    
    # Marginalising fitness for every clone to get clonal probability
    for i, out in enumerate(output.T):
        # print(f"Processing clone {i}")
        # print(f"  Posterior shape: {out.shape}")
        # print(f"  Posterior range: [{out.min():.3e}, {out.max():.3e}]")
        
        integral = np.trapz(x=s_range, y=out)
        clonal_prob[i] = s_prior * integral
        # print(f"  Integral: {integral:.3e}, clonal_prob: {clonal_prob[i]:.3e}")

    # Model probability as product of clonal probabilities
    # because of independence
    model_probability = np.prod(clonal_prob)
    # print(f"Final model probability: {model_probability:.3e}")
    # print(f"Individual clonal probabilities: {clonal_prob}")

    return model_probability

def refine_optimal_model_posterior(part, s_resolution=100):
    # print(f"\n=== refine_optimal_model_posterior ===")
    
    # Compute finer posterior for optimal model
    # retrieve optimal clonal structure
    best_key = list(part.uns['model_dict'].keys())[0]
    cs, best_prob = part.uns['model_dict'][best_key]
    # print(f"Optimal model: {best_key}")
    # print(f"Clonal structure: {cs}")
    # print(f"Probability: {best_prob:.3e}")
    
    output, s_range = single_cs_posterior(part, cs, s_resolution)
    part.uns['optimal_model'] = {'clonal_structure': cs,
                                'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
                                'posterior': output,
                                's_range': s_range}

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    # print(f"\nComputing fitness estimates:")
    for i, c_idx in enumerate(cs):
        # print(f"Clone {i} (mutations {c_idx}):")
        
        p = np.array(output[:, i])
        # print(f"  Raw posterior shape: {p.shape}, sum: {p.sum():.3e}")
        
        # Check for zero posterior
        if p.sum() == 0:
            # print(f"  WARNING: Zero posterior sum for clone {i}!")
            continue
            
        # normalise output
        p_norm = p / p.sum()
        # print(f"  Normalized posterior sum: {p_norm.sum()}")
        
        sample_range = np.random.choice(s_range, p=p_norm, size=1_000)
        fitness_map = s_range[np.argmax(p_norm)]
        cfd_int = np.quantile(sample_range, [0.05, 0.95])
        
        # print(f"  MAP fitness: {fitness_map:.3f}")
        # print(f"  95% CI: [{cfd_int[0]:.3f}, {cfd_int[1]:.3f}]")

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = cfd_int[0]
        fitness_95[c_idx] = cfd_int[1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index

    # print(f"\nFinal fitness assignments:")
    print(part.obs[['fitness', 'fitness_5', 'fitness_95', 'clonal_index']])

    return part

def plot_optimal_model(part):
    if part.uns['warning'] is not None:
        print('WARNING: ' + part.uns['warning'])
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    ms = model['mutation_structure']
    s_range = model['s_range']

    # normalisation constant
    norm_max = np.max(output, axis=0)

    # Plot
    for i in range(len(cs)):
        p_key_str = f''
        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f'{part[j].obs.p_key.values[0]}'
            if k>0:
                p_key_str += f'\n{part[j].obs.p_key.values[0]}'

        sns.lineplot(x=s_range,
                    y=output[:, i]/ norm_max[i],
                    label=p_key_str)
#endregion

#region vectorised and jit optimised functions with DEBUG

def jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs,
                            deterministic_size, total_cells, max_total_per_mutation, key, resolution=600):
    """
    s_vec: (n_s,)
    AO, DP: (n_tps, n_mut)
    cs: clonal structure (list of lists)
    key: PRNGKey for sampling (will be split inside compute_global_variables_mixed)
    """
    # print(f"\n=== jax_cs_hmm_ll_vec_mixed ===")
    # print(f"s_vec shape: {s_vec.shape}, range: [{s_vec.min():.3f}, {s_vec.max():.3f}]")
    # print(f"AO shape: {AO.shape}, DP shape: {DP.shape}")
    # print(f"time_points: {time_points}")
    # print(f"clonal structure: {cs}")
    # print(f"deterministic_size shape: {deterministic_size.shape}")
    # print(f"total_cells: {total_cells}")
    # print(f"max_total_per_mutation: {max_total_per_mutation}")
    # print(f"resolution: {resolution}")
    
    # compute global variables (samples) using the provided key
    x_het_vec, x_hom_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut = \
        compute_global_variables_mixed(s_vec, AO, DP, total_cells, deterministic_size, max_total_per_mutation, time_points, key, resolution=resolution)

    # s index vector
    s_idx = jnp.arange(s_vec.shape[0])
    # print(f"s_idx: {s_idx}")

    # fitness_specific_computations mapped over s index
    def fitness_specific_computations_mapped(s_idx):
        s = s_vec[s_idx]
        exp_term_vec = exp_term_vec_s[s_idx]   # (n_intervals,1,1)
        ## print(f"  Processing s_idx {s_idx}: s={s:.3f}")
        
        # BD dynamics on total
        p_vec, n_vec = BD_process_dynamics_mixed(s, x_het_vec + x_hom_vec, exp_term_vec)

        # compute mutation-likelihoods across all mutations (vectorised)
        mutation_likelihood = jax.vmap(lambda ii: mutation_specific_ll_mixed_grid(ii,
                                                                                recursive_term_vec[:, :],
                                                                                x_het_vec, x_hom_vec,
                                                                                p_vec, n_vec, p_y_cond_x_vec,
                                                                                time_points.shape[0]))(jnp.arange(n_mut))
        
        # print(f"  Mutation likelihoods shape: {mutation_likelihood.shape}")
        # print(f"  Mutation likelihoods: {mutation_likelihood}")
        
        # now compute clonal likelihoods as product over mutations in each clone
        clonal_likelihood = jnp.zeros(len(cs))
        for idx_c, c_idx in enumerate(cs):
            clone_mutations = jnp.array(c_idx)
            clone_likelihood = jnp.prod(mutation_likelihood[clone_mutations])
            clonal_likelihood = clonal_likelihood.at[idx_c].set(clone_likelihood)
            # print(f"  Clone {idx_c} (mutations {c_idx}) likelihood: {clone_likelihood:.3e}")
            
        return clonal_likelihood

    # vmap over s
    print("Computing clonal likelihoods for all s values...")
    clonal_likelihood = jax.vmap(fitness_specific_computations_mapped)(s_idx)
    # print(f"Final clonal_likelihood shape: {clonal_likelihood.shape}")
    # print(f"Final clonal_likelihood range: [{clonal_likelihood.min():.3e}, {clonal_likelihood.max():.3e}]")
    return clonal_likelihood  # shape (n_s, n_clones)

def compute_global_variables_mixed(s_vec, AO, DP, total_cells, deterministic_size,
                                    max_total_per_mutation, time_points, key, resolution=600):
    """
    Sample x_het and x_hom directly with uniform prior on [0, max_total_per_mutation[m]]
    Shapes:
    AO: (n_tps, n_mut)
    returns x_het_vec, x_hom_vec shaped (n_tps, n_mut, resolution)
    """
    # print(f"\n=== compute_global_variables_mixed ===")
    # print(f"s_vec shape: {s_vec.shape}")
    # print(f"AO shape: {AO.shape}, DP shape: {DP.shape}")
    # print(f"total_cells shape: {total_cells.shape}")
    # print(f"deterministic_size shape: {deterministic_size.shape}")
    # print(f"max_total_per_mutation: {max_total_per_mutation}")
    # print(f"time_points: {time_points}")
    # print(f"resolution: {resolution}")
    
    n_tps, n_mut = AO.shape
    # print(f"n_timepoints: {n_tps}, n_mutations: {n_mut}")
    
    # BD exponential term for pmf (shape: (n_s, n_intervals))
    delta_t = jnp.diff(time_points)
    # print(f"delta_t: {delta_t}")
    
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])             # (n_s, n_intervals)
    # print(f"exp_term_vec_s shape: {exp_term_vec_s.shape}")
    # print(f"exp_term_vec_s sample (first 3 s, first 3 intervals):\n{exp_term_vec_s[:3, :3]}")
    
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))  # (...,1,1) for broadcasting

    # Create two subkeys for uniform sampling
    key_het, key_hom = jrnd.split(key, 2)

    # Uniform samples over [0, max_total_per_mutation] for each mutation & timepoint
    # We'll sample independently for het and hom (they will be integrated jointly).
    # Shape requested: (n_tps, n_mut, resolution)
    het_raw = jrnd.uniform(key_het, shape=(n_tps, n_mut, resolution), dtype=jnp.float32)
    hom_raw = jrnd.uniform(key_hom, shape=(n_tps, n_mut, resolution), dtype=jnp.float32)
    ## print(f"het_raw shape: {het_raw.shape}, range: [{het_raw.min():.3f}, {het_raw.max():.3f}]")
    ## print(f"hom_raw shape: {hom_raw.shape}, range: [{hom_raw.min():.3f}, {hom_raw.max():.3f}]")

    # Scale to [0, max_total_per_mutation[m]]
    max_totals = max_total_per_mutation[None, :, None]  # (1, n_mut, 1)
    ## print(f"max_totals shape: {max_totals.shape}")
    
    x_het_vec = het_raw * max_totals
    x_hom_vec = hom_raw * max_totals
    ## print(f"x_het_vec shape: {x_het_vec.shape}, range: [{x_het_vec.min():.1f}, {x_het_vec.max():.1f}]")
    ## print(f"x_hom_vec shape: {x_hom_vec.shape}, range: [{x_hom_vec.min():.1f}, {x_hom_vec.max():.1f}]")

    # Optionally enforce a tiny floor to avoid zeros everywhere
    eps = 1e-6
    x_het_vec = jnp.clip(x_het_vec, eps, max_totals)
    x_hom_vec = jnp.clip(x_hom_vec, eps, max_totals)
    ## print(f"After clipping - x_het_vec range: [{x_het_vec.min():.1f}, {x_het_vec.max():.1f}]")
    ## print(f"After clipping - x_hom_vec range: [{x_hom_vec.min():.1f}, {x_hom_vec.max():.1f}]")

    # Compute total_x and true_vaf for the sampled grid
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]  # (n_tps, n_mut, 1)
    ## print(f"N_w_cond_vec shape: {N_w_cond_vec.shape}, range: [{N_w_cond_vec.min():.1f}, {N_w_cond_vec.max():.1f}]")
    
    x_total_vec = x_het_vec + x_hom_vec
    ## print(f"x_total_vec shape: {x_total_vec.shape}, range: [{x_total_vec.min():.1f}, {x_total_vec.max():.1f}]")
    
    # CRITICAL: Check for division issues here
    denominator_vaf = 2.0 * (N_w_cond_vec + x_total_vec)
    zero_denom_mask = denominator_vaf == 0
    if jnp.any(zero_denom_mask):
        # print(f"WARNING: {jnp.sum(zero_denom_mask)} zero denominators in VAF calculation!")
        denominator_vaf = jnp.where(zero_denom_mask, 1.0, denominator_vaf)
    
    ## print(f"VAF denominator shape: {denominator_vaf.shape}, range: [{denominator_vaf.min():.1f}, {denominator_vaf.max():.1f}]")
    
    true_vaf_vec = (x_het_vec + 2.0 * x_hom_vec) / denominator_vaf
    ## print(f"true_vaf_vec shape: {true_vaf_vec.shape}, range: [{true_vaf_vec.min():.3f}, {true_vaf_vec.max():.3f}]")
    
    # Check for invalid VAFs
    invalid_vafs = (true_vaf_vec <= 0) | (true_vaf_vec >= 1)
    if jnp.any(invalid_vafs):
        # print(f"WARNING: {jnp.sum(invalid_vafs)} invalid VAFs (<=0 or >=1)")
        true_vaf_vec = jnp.clip(true_vaf_vec, 1e-8, 1-1e-8)

    # compute conditional observation probabilities over all time points
    # AO: (n_tps, n_mut) -> expand to (n_tps, n_mut, resolution)
    p_y_cond_x_vec = jsp_stats.binom.pmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)
    p_y_cond_x_vec = jnp.maximum(p_y_cond_x_vec, 1e-300)
    # # print(f"p_y_cond_x_vec shape: {p_y_cond_x_vec.shape}")
    # # print(f"p_y_cond_x_vec range: [{p_y_cond_x_vec.min():.3e}, {p_y_cond_x_vec.max():.3e}]")
    # # print(f"Zero probabilities in p_y_cond_x_vec: {jnp.sum(p_y_cond_x_vec == 0)}")
    # # print(f"NaN probabilities in p_y_cond_x_vec: {jnp.sum(jnp.isnan(p_y_cond_x_vec))}")

    # initial recursive term (uniform prior on the grid => weight = 1/resolution^2)
    prior_weight = (1.0 / resolution) * (1.0 / resolution)  # joint uniform over het & hom samples
    recursive_term_vec = p_y_cond_x_vec[0, :, :] * prior_weight
    # # print(f"recursive_term_vec shape: {recursive_term_vec.shape}")
    # # print(f"recursive_term_vec range: [{recursive_term_vec.min():.3e}, {recursive_term_vec.max():.3e}]")
    # # print(f"Zero recursive terms: {jnp.sum(recursive_term_vec == 0)}")

    return x_het_vec, x_hom_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut

def BD_process_dynamics_mixed(s, x_total_vec, exp_term_vec):
    # # print(f"\n=== BD_process_dynamics_mixed ===")
    # # print(f"s: {s:.3f}")
    # # print(f"x_total_vec shape: {x_total_vec.shape}")
    # # print(f"exp_term_vec shape: {exp_term_vec.shape}")
    
    lamb = 1.3
    # x_total_vec shape: (n_tps, n_mut, resolution)
    mean_vec = x_total_vec[:-1, :, :] * exp_term_vec   # broadcasting
    ## print(f"mean_vec shape: {mean_vec.shape}, range: [{mean_vec.min():.1f}, {mean_vec.max():.1f}]")
    
    # CRITICAL: Check for division by zero in variance calculation
    s_safe = jnp.maximum(s, 1e-8)
    variance_vec = x_total_vec[:-1, :, :] * (2.0 * lamb + s) * exp_term_vec * (exp_term_vec - 1.0) / s_safe
    ## print(f"variance_vec shape: {variance_vec.shape}, range: [{variance_vec.min():.1f}, {variance_vec.max():.1f}]")
    
    # Check for variance <= mean (invalid for negative binomial)
    invalid_variance = variance_vec <= mean_vec
    variance_vec = jnp.where(invalid_variance, mean_vec * 1.1 + 1e-8, variance_vec)
    
    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / jnp.maximum(variance_vec - mean_vec, 1e-8)
    
    # # print(f"p_vec shape: {p_vec.shape}, range: [{p_vec.min():.3f}, {p_vec.max():.3f}]")
    # # print(f"n_vec shape: {n_vec.shape}, range: [{n_vec.min():.1f}, {n_vec.max():.1f}]")
    
    # Fix invalid parameters (always apply clipping)
    p_vec = jnp.clip(p_vec, 1e-8, 1-1e-8)
    n_vec = jnp.maximum(n_vec, 1e-8)
    
    return p_vec, n_vec

def mutation_specific_ll_mixed_grid(i, recursive_term_vec, x_het_vec, x_hom_vec, p_vec, n_vec, p_y_cond_x_vec, n_tps):
    """
    Integrate recursively over the 2D grid for mutation i.
    x_het_vec, x_hom_vec shapes: (n_tps, n_mut, resolution)
    We extract mutation i and run the same recursive trapezoid updates.
    """
    ## print(f"\n=== mutation_specific_ll_mixed_grid for mutation {i} ===")
    
    recursive_term_i = recursive_term_vec[i]        # shape (resolution,)
    x_het_i = x_het_vec[:, i, :]                    # (n_tps, resolution)
    x_hom_i = x_hom_vec[:, i, :]                    # (n_tps, resolution)
    p_i = p_vec[:, i, :]                            # (n_intervals, resolution)
    n_i = n_vec[:, i, :]                            # (n_intervals, resolution)
    p_y_cond_x_i = p_y_cond_x_vec[:, i, :]          # (n_tps, resolution)
    
    # # print(f"recursive_term_i shape: {recursive_term_i.shape}")
    # # print(f"x_het_i shape: {x_het_i.shape}, x_hom_i shape: {x_hom_i.shape}")
    # # print(f"p_i shape: {p_i.shape}, n_i shape: {n_i.shape}")
    # # print(f"p_y_cond_x_i shape: {p_y_cond_x_i.shape}")

    # iterate timepoints
    for j in range(1, n_tps):
        # print(f"  Timepoint {j}:")
        
        # total sizes at previous and current tp (vectors over resolution)
        init_total = x_het_i[j-1] + x_hom_i[j-1]   # (resolution,)
        next_total = x_het_i[j] + x_hom_i[j]       # (resolution,)
        
        # # print(f"    init_total range: [{init_total.min():.1f}, {init_total.max():.1f}]")
        # # print(f"    next_total range: [{next_total.min():.1f}, {next_total.max():.1f}]")

        # bd pmf: probability of transitioning from init_total grid to next_total grid
        # Use broadcasting with [:, None] to get (next_res, init_res)
        bd_pmf = jsp_stats.nbinom.pmf(next_total[:, None], p=p_i[j-1][None, :], n=n_i[j-1][None, :])
        # # print(f"    bd_pmf shape: {bd_pmf.shape}, range: [{bd_pmf.min():.3e}, {bd_pmf.max():.3e}]")
        # # print(f"    Zero bd_pmf values: {jnp.sum(bd_pmf == 0)}")

        # inner_sum: integrate over previous-res grid (trapezoid) weighted by recursive_term_i (shape matches init_res)
        inner_sum = bd_pmf * recursive_term_i  # (next_res, init_res)
        # # print(f"    inner_sum shape: {inner_sum.shape}, range: [{inner_sum.min():.3e}, {inner_sum.max():.3e}]")
        # # print(f"    Zero inner_sum values: {jnp.sum(inner_sum == 0)}")

        # integrate over init axis using trapezoid
        # x coordinates for trap integration should be init_total values
        inner_integrated = jsp.integrate.trapezoid(x=init_total, y=inner_sum, axis=1)  # returns shape (next_res,)
        # # print(f"    inner_integrated shape: {inner_integrated.shape}, range: [{inner_integrated.min():.3e}, {inner_integrated.max():.3e}]")
        # # print(f"    Zero inner_integrated values: {jnp.sum(inner_integrated == 0)}")

        # multiply by observation probability at current timepoint from the grid
        log_recursive_term_i = jnp.log(p_y_cond_x_i[j]) + jnp.log(inner_integrated)
        recursive_term_i = jnp.exp(log_recursive_term_i)
        # # print(f"    New recursive_term_i shape: {recursive_term_i.shape}, range: [{recursive_term_i.min():.3e}, {recursive_term_i.max():.3e}]")
        # # print(f"    Zero new recursive terms: {jnp.sum(recursive_term_i == 0)}")

    # final integrate over the last grid (total_x_final)
    total_x_final = (x_het_i[-1] + x_hom_i[-1])   # (resolution,)
    final_like = jsp.integrate.trapezoid(x=total_x_final, y=recursive_term_i)
    ## print(f"  Final likelihood for mutation {i}: {final_like:.3e}")
    
    return final_like

def compute_clonal_models_prob_vec_mixed(part, s_resolution=50, min_s=0.01, max_s=3,
                                        filter_invalid=True, disable_progressbar=False,
                                        resolution=600, master_key_seed=758493):
    # print(f"\n" + "="*60)
    print("COMPUTING CLONAL MODELS PROBABILITIES (VECTORIZED)")
    print("="*60)
    
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    n_mutations = part.shape[0]
    part.uns['model_dict'] = {}
    
    # print(f"Dataset info:")
    # print(f"  n_mutations: {n_mutations}")
    # print(f"  n_timepoints: {len(time_points)}")
    # print(f"  AO shape: {AO.shape}")
    # print(f"  DP shape: {DP.shape}")
    # print(f"  time_points: {time_points}")
    # print(f"  s_range: [{min_s}, {max_s}] with {s_resolution} points")
    # print(f"  resolution: {resolution}")

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)
    # print(f"Number of valid clonal structures: {len(cs_list)}")
    for i, cs in enumerate(cs_list):
        # print(f"  Structure {i}: {cs}")

        part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mutations)]]
        # print(f"WARNING: Too many structures, using single mutation per clone")

    # create master key and split it for each model
    master_key = jrnd.PRNGKey(master_key_seed)
    keys = jrnd.split(master_key, len(cs_list))

    for i, cs in enumerate(cs_list):
        # print(f"\n" + "-"*50)
        # print(f"PROCESSING MODEL {i}: {cs}")
        print("-"*50)
        
        # deterministic estimates + max_total bound
        deterministic_size, total_cells, max_total_per_mutation, clonal_map = compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

        # call vectorised mixed-zygosity likelihood with its own key
        key_i = keys[i]
        output = jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs,
                                        deterministic_size, total_cells, max_total_per_mutation, key_i,
                                        resolution=resolution)

        model_prob = compute_model_likelihood(output, cs, s_vec)
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)
        # print(f"MODEL {i} FINAL PROBABILITY: {model_prob:.3e}")

    # Sort models by probability
    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), key=lambda item: item[1][1], reverse=True)}
    
    # print(f"\n" + "="*60)
    print("MODEL RANKING (sorted by probability):")
    print("="*60)
    for i, (k, v) in enumerate(part.uns['model_dict'].items()):
        cs, prob = v
        # print(f"Rank {i}: {k} - Probability: {prob:.3e}")
        # print(f"       Structure: {cs}")
    
    return part

def refine_optimal_model_posterior_vec(part, s_resolution=100):
    # Compute finer posterior for optimal model
    # retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]

    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    # compute deterministic clone sizes
    deterministic_size, total_cells, max_total_per_mutation, clonal_map = compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

    # Create refined s_range
    s_vec = jnp.linspace(0.01, 1, s_resolution)

    # compute clonal posteriors
    master_key = jrnd.PRNGKey(758493)
    output = jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs, deterministic_size, total_cells, max_total_per_mutation, master_key)

    part.uns['optimal_model'] = {'clonal_structure': cs,
                                'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
                                'posterior': output,
                                's_range': s_vec}

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):

        # Extract posterior for each clone
        p = np.array(output[:, i])
        if np.nansum(p).sum() == 0:
            part.uns['warning'] = 'Zero posterior'
            return part
        
        # normalise posterior and handle numerical issues
        p = np.maximum(p, 0)  # Ensure non-negative
        if p.sum() == 0:
            print(f"WARNING: Zero posterior for clone {i}, skipping")
            continue
        p = p / p.sum()

        # fitness map 
        fitness_map = s_vec[np.argmax(p)]

        # 95% CI for fitness using bootstrap
        sample_range = np.random.choice(s_vec, p=p, size=1_000)
        cfd_int = np.quantile(sample_range, [0.05, 0.95])

        # Append information
        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = cfd_int[0]
        fitness_95[c_idx] = cfd_int[1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index

    # Append mutational structure to each mutation
    mut_structure = part.uns['optimal_model']['mutation_structure']

    clonal_structure_list = []
    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)

    part.obs['clonal_structure'] = clonal_structure_list

    return part

#endregion

def compute_invalid_combinations(part, pearson_distance_threshold=0.5):
    # print(f"\n=== compute_invalid_combinations ===")
    # print(f"pearson_distance_threshold: {pearson_distance_threshold}")

    # Compute pearson of each mutation with time
    correlation_matrix = np.corrcoef(
        np.vstack([part.X, part.var.time_points]))
    correlation_vec = correlation_matrix[-1, :-1]
    # print(f"Correlation with time: {correlation_vec}")

    # Compute distance between pearsonr's
    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])
    # print(f"Distance matrix shape: {distance_matrix.shape}")
    # print(f"Distance matrix:\n{distance_matrix}")

    # label invalid combinations if pearson correlation is too different
    not_valid_comb = np.argwhere(distance_matrix > pearson_distance_threshold)
    # print(f"Raw invalid combinations: {not_valid_comb.tolist()}")
    
    # Extract unique tuples from list(Order Irrespective)
    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] and [i[1], i[0]] not in res:
            res.append(i.tolist()) 
    
    # print(f"Final invalid combinations: {res}")
    part.uns['invalid_combinations'] = res

def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """
    Find all valid clonal structures using pearson correlation analysis"""
    # print(f"\n=== find_valid_clonal_structures ===")
    # print(f"filter_invalid: {filter_invalid}, p_distance_threshold: {p_distance_threshold}")
    
    n_mutations = part.shape[0]
    # print(f"n_mutations: {n_mutations}")

    if n_mutations == 1:
        valid_cs = [[[0]]]
        # print(f"Single mutation case, returning: {valid_cs}")
        return valid_cs

    else:
        if filter_invalid is True:
            # compute invalid clonal structures using correlation analysis
            compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)
            # print(f"Invalid combinations: {part.uns.get('invalid_combinations', [])}")
            
        # create list of all possible clonal structures
        a = list(partition(list(range(n_mutations))))
        cs_list = [cs for cs in a]
        # print(f"Total possible structures: {len(cs_list)}")

        if filter_invalid is False:
            # print(f"Returning all {len(cs_list)} structures (no filtering)")
            return cs_list
        
        else:
            # find all valid clonal structures
            valid_cs = []

            for cs in cs_list:
                invalid_combinations_in_cs = 0
                for clone in cs:
                    # compute all pairs of mutations inside clone
                    mut_comb = list(combinations(clone, 2))
                    # check if any pair is invalid
                    n_invalid_comb_in_clone = len(
                        [comb for comb in mut_comb 
                            if list(comb) in part.uns['invalid_combinations']])
                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                # append valid clonal structure
                if invalid_combinations_in_cs == 0:
                    valid_cs.append(cs)
                    
            # print(f"Valid structures after filtering: {len(valid_cs)}")
            return valid_cs