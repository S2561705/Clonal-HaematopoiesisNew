"""
HYBRID BEST VERSION: Combines the best of both approaches

From Document 3 (h-param):
- ✅ Numerical stability fixes (log-space, clipping, bounds checking)
- ✅ Robust BD parameter calculation
- ✅ Careful error handling

From Fixed A-param:
- ✅ True A-parameterization (A ∈ [0.5, 1.0])
- ✅ No het/hom splitting (simpler, faster)
- ✅ Early marginalization over A
- ✅ Data-independent uniform prior

Result: Best of both worlds!
"""

import sys
sys.path.append("..")
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd
from itertools import combinations
from scipy.stats import binom
import numpy as np


key = jrnd.PRNGKey(758493)

#region Non-vectorised functions (from Doc 3 - working well)

def compute_deterministic_size_mixed(cs, AO, DP, n_mutations, N_w=1e5):
    """
    Robust deterministic estimate with proper handling of edge cases.
    (From Document 3 - this works well)
    """
    
    # Calculate VAFs with safety checks
    vaf_ratio = AO / jnp.maximum(DP, 1.0)
    
    # Find leading mutation per clone
    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        clone_vafs = vaf_ratio[:, cs_idx]
        vaf_sums = clone_vafs.sum(axis=0)
        max_idx = int(jnp.argmax(vaf_sums))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    # Robust heterozygous rearrangement with proper bounds
    vafs_lead = vaf_ratio[:, lm]
    sum_vafs_lead = jnp.sum(vafs_lead, axis=1)
    
    # Handle singularity near 0.5
    denominator = sum_vafs_lead - 0.5
    far_from_singular = jnp.abs(denominator) > 0.15
    
    original_estimate = -N_w * sum_vafs_lead / jnp.maximum(jnp.abs(denominator), 0.15) * jnp.sign(denominator)
    linear_estimate = N_w * sum_vafs_lead / (1.0 - sum_vafs_lead + 1e-8)
    
    deterministic_clone_size = jnp.where(far_from_singular, original_estimate, linear_estimate)
    deterministic_clone_size = jnp.maximum(deterministic_clone_size, 0.0)
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    deterministic_clone_size = jnp.minimum(deterministic_clone_size, N_w * 10)
    
    total_cells = N_w + deterministic_clone_size
    deterministic_size = vaf_ratio * 2 * total_cells[:, None]
    
    max_per_tp = deterministic_size.max(axis=0)
    mean_per_mut = deterministic_size.mean(axis=0)
    max_total_per_mutation = jnp.maximum(
        jnp.maximum(max_per_tp * 3, mean_per_mut * 5),
        100.0
    )
    max_total_per_mutation = jnp.minimum(max_total_per_mutation, N_w * 5)

    return (deterministic_size.astype(jnp.float32), 
            total_cells.astype(jnp.float32), 
            max_total_per_mutation.astype(jnp.float32), 
            clonal_map)

#endregion

#region Vectorized functions - TRUE A-parameterization with numerical stability

def jax_cs_hmm_ll_vec_A_param(s_vec, AO, DP, time_points, cs,
                               deterministic_size, total_cells, max_total_per_mutation, 
                               key, resolution=600, A_resolution=20):
    """
    TRUE A-PARAMETERIZATION with numerical stability from Doc 3
    """
    
    x_total_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut = \
        compute_global_variables_A_param(s_vec, AO, DP, total_cells, deterministic_size, 
                                         max_total_per_mutation, time_points, key, 
                                         resolution=resolution, A_resolution=A_resolution)

    s_idx = jnp.arange(s_vec.shape[0])

    def fitness_specific_computations_mapped(s_idx):
        s = s_vec[s_idx]
        exp_term_vec = exp_term_vec_s[s_idx]
        
        # BD dynamics on total clone size (from Doc 3 - robust)
        p_vec, n_vec = BD_process_dynamics_robust(s, x_total_vec, exp_term_vec)

        mutation_likelihood = jax.vmap(
            lambda ii: mutation_specific_ll_A_param(
                ii, recursive_term_vec,
                x_total_vec,
                p_vec, n_vec, p_y_cond_x_vec,
                time_points.shape[0]
            )
        )(jnp.arange(n_mut))
        
        clonal_likelihood = jnp.zeros(len(cs))
        for idx_c, c_idx in enumerate(cs):
            clone_mutations = jnp.array(c_idx)
            log_mut_liks = jnp.log(jnp.maximum(mutation_likelihood[clone_mutations], 1e-300))
            clone_likelihood = jnp.exp(jnp.sum(log_mut_liks))
            clonal_likelihood = clonal_likelihood.at[idx_c].set(clone_likelihood)
            
        return clonal_likelihood

    clonal_likelihood = jax.vmap(fitness_specific_computations_mapped)(s_idx)
    return clonal_likelihood


def compute_global_variables_A_param(s_vec, AO, DP, total_cells, deterministic_size,
                                      max_total_per_mutation, time_points, key, 
                                      resolution=600, A_resolution=20):
    """
    TRUE A-PARAMETERIZATION: Combines efficiency with numerical stability
    
    Key improvements:
    - A ∈ [0.5, 1.0] sampled directly (not h)
    - Early marginalization (efficient)
    - Vectorized operations (JAX-compatible)
    - Log-space stability (from Doc 3)
    - Proper bounds checking (from Doc 3)
    """
    
    n_tps, n_mut = AO.shape
    
    # BD exponential term
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1))

    key_total, _ = jrnd.split(key, 2)

    # 1. Sample total mutant cells
    total_raw = jrnd.uniform(key_total, shape=(n_tps, n_mut, resolution), 
                             dtype=jnp.float32)
    max_totals = max_total_per_mutation[None, :, None]
    x_total_vec = total_raw * max_totals
    
    # 2. Marginalize over A using vectorized grid
    A_grid = jnp.linspace(0.5, 1.0, A_resolution)
    
    # Get conditional wild-type population (with safety from Doc 3)
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    N_w_cond_vec = jnp.maximum(N_w_cond_vec, 1e-6)  # Safety from Doc 3
    
    # Clone fraction
    clone_fraction = x_total_vec / (N_w_cond_vec + x_total_vec)
    
    # 3. Vectorized VAF computation for all A values
    clone_fraction_expanded = clone_fraction[:, :, :, None]  # (n_tps, n_mut, res, 1)
    A_grid_expanded = A_grid[None, None, None, :]  # (1, 1, 1, A_res)
    
    # VAF for all A values
    true_vaf_all_A = A_grid_expanded * clone_fraction_expanded
    true_vaf_all_A = jnp.clip(true_vaf_all_A, 1e-8, 1.0 - 1e-8)  # Bounds from Doc 3
    
    # 4. Observation likelihoods (log-space from Doc 3)
    AO_expanded = AO[:, :, None, None]
    DP_expanded = DP[:, :, None, None]
    
    log_p_y_cond_x_all_A = jsp_stats.binom.logpmf(
        AO_expanded, 
        n=DP_expanded, 
        p=true_vaf_all_A
    )
    
    # Floor to prevent underflow (from Doc 3)
    log_p_y_cond_x_all_A = jnp.maximum(log_p_y_cond_x_all_A, -300.0)
    p_y_cond_x_all_A = jnp.exp(log_p_y_cond_x_all_A)
    
    # 5. Marginalize over A (uniform prior)
    p_y_cond_x_vec = jnp.mean(p_y_cond_x_all_A, axis=3)
    
    # 6. Initial recursive term
    prior_weight = 1.0 / resolution
    recursive_term_vec = p_y_cond_x_vec[0, :, :] * prior_weight

    return x_total_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut


def BD_process_dynamics_robust(s, x_total_vec, exp_term_vec):
    """
    Robust BD dynamics from Document 3
    """
    
    lamb = 1.3
    
    mean_vec = x_total_vec[:-1, :, :] * exp_term_vec
    
    # Robust variance (from Doc 3)
    s_safe = jnp.maximum(jnp.abs(s), 1e-8)
    variance_vec = x_total_vec[:-1, :, :] * (2.0 * lamb + s) * exp_term_vec * (exp_term_vec - 1.0) / s_safe
    
    # Ensure variance > mean (from Doc 3)
    min_variance_ratio = 1.2
    min_variance = mean_vec * min_variance_ratio + 1e-6
    variance_vec = jnp.maximum(variance_vec, min_variance)
    
    # Strict parameter bounds (from Doc 3)
    p_vec = mean_vec / variance_vec
    p_vec = jnp.clip(p_vec, 1e-8, 1.0 - 1e-8)
    
    n_vec = jnp.power(mean_vec, 2) / jnp.maximum(variance_vec - mean_vec, 1e-8)
    n_vec = jnp.maximum(n_vec, 1e-8)
    
    return p_vec, n_vec


def mutation_specific_ll_A_param(i, recursive_term_vec, x_total_vec, 
                                  p_vec, n_vec, p_y_cond_x_vec, n_tps):
    """
    A-parameterized likelihood with log-space stability from Doc 3
    """
    
    recursive_term_i = recursive_term_vec[i]
    x_total_i = x_total_vec[:, i, :]
    p_i = p_vec[:, i, :]
    n_i = n_vec[:, i, :]
    p_y_cond_x_i = p_y_cond_x_vec[:, i, :]

    for j in range(1, n_tps):
        init_total = x_total_i[j-1]
        next_total = x_total_i[j]

        # Log-space BD PMF (from Doc 3)
        log_bd_pmf = jsp_stats.nbinom.logpmf(
            next_total[:, None], 
            p=p_i[j-1][None, :], 
            n=n_i[j-1][None, :]
        )
        log_bd_pmf = jnp.maximum(log_bd_pmf, -300.0)  # Floor from Doc 3
        bd_pmf = jnp.exp(log_bd_pmf)
        
        inner_sum = bd_pmf * recursive_term_i[None, :]
        
        inner_integrated = jsp.integrate.trapezoid(
            x=init_total, 
            y=inner_sum, 
            axis=1
        )
        inner_integrated = jnp.maximum(inner_integrated, 1e-300)  # Floor from Doc 3
        
        # Log-space multiplication (from Doc 3)
        log_p_y = jnp.log(jnp.maximum(p_y_cond_x_i[j], 1e-300))
        log_inner = jnp.log(inner_integrated)
        log_recursive_term_i = log_p_y + log_inner
        
        recursive_term_i = jnp.exp(jnp.maximum(log_recursive_term_i, -300.0))

    final_like = jsp.integrate.trapezoid(
        x=x_total_i[-1], 
        y=recursive_term_i, 
        axis=0
    )
    final_like = jnp.maximum(final_like, 1e-300)
    
    return final_like


def compute_clonal_models_prob_vec_hybrid(part, s_resolution=50, min_s=0.01, max_s=3,
                                          filter_invalid=True, 
                                          resolution=600, A_resolution=20, 
                                          master_key_seed=758493):
    """
    HYBRID VERSION: True A-parameterization + numerical stability
    """
    
    print("="*70)
    print("HYBRID A-PARAMETERIZATION (Best of Both Worlds)")
    print("="*70)
    print(f"✓ True A-parameterization (A ∈ [0.5, 1.0])")
    print(f"✓ Numerical stability from battle-tested h-version")
    print(f"✓ Efficient early marginalization")
    print(f"✓ Data-independent uniform prior")
    print("="*70)
    
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    n_mutations = part.shape[0]
    part.uns['model_dict'] = {}

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mutations)]]

    master_key = jrnd.PRNGKey(master_key_seed)
    keys = jrnd.split(master_key, len(cs_list))

    for i, cs in enumerate(cs_list):
        print(f"\nProcessing model {i+1}/{len(cs_list)}: {cs}")
        
        deterministic_size, total_cells, max_total_per_mutation, clonal_map = \
            compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

        key_i = keys[i]
        output = jax_cs_hmm_ll_vec_A_param(s_vec, AO, DP, time_points, cs,
                                           deterministic_size, total_cells, max_total_per_mutation, 
                                           key_i, resolution=resolution, A_resolution=A_resolution)

        model_prob = compute_model_likelihood(output, cs, s_vec)
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)
        print(f"  Model probability: {model_prob:.3e}")

    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), 
                                                       key=lambda item: item[1][1], reverse=True)}
    
    print("\n" + "="*70)
    print("MODEL RANKING:")
    print("="*70)
    for i, (k, v) in enumerate(part.uns['model_dict'].items()):
        cs, prob = v
        print(f"Rank {i+1}: {k} - Probability: {prob:.3e}, Structure: {cs}")
    
    return part


def infer_sA_jointly_from_dynamics(cs, AO, DP, time_points, 
                                   deterministic_size, total_cells, max_total_per_mutation,
                                   s_resolution=30, A_resolution=20, lamb=1.3, N_w=1e5):
    """
    Joint (s, A) inference with numerical stability
    """
    
    print("\n" + "="*70)
    print("JOINT (s, A) INFERENCE")
    print("="*70)
    print("A is the VAF asymptote ∈ [0.5, 1.0]")
    print("h = 2A - 1 is derived for interpretation")
    print("="*70)
    
    s_range = np.linspace(0.01, 1.0, s_resolution)
    A_range = np.linspace(0.5, 1.0, A_resolution)
    
    results = []
    
    for clone_idx, clone_muts in enumerate(cs):
        print(f"\nClone {clone_idx}: {clone_muts}")
        print("-" * 70)
        
        mut_idx = clone_muts[0]
        AO_clone = np.array(AO[:, mut_idx])
        DP_clone = np.array(DP[:, mut_idx])
        
        observed_vaf = AO_clone / np.maximum(DP_clone, 1.0)
        print(f"  VAF trajectory: {observed_vaf[0]:.3f} → {observed_vaf[-1]:.3f}")
        
        joint_log_likelihood = np.zeros((s_resolution, A_resolution))
        
        for s_idx, s in enumerate(s_range):
            for A_idx, A in enumerate(A_range):
                log_lik = 0
                N_mut = max(observed_vaf[0] * N_w, 100)
                
                for tp_idx in range(len(time_points)):
                    if tp_idx > 0:
                        dt = time_points[tp_idx] - time_points[tp_idx-1]
                        N_mut = N_mut * np.exp(s * dt)
                    
                    N_mut = min(N_mut, N_w * 0.95)
                    
                    # TRUE A-parameterized VAF
                    clone_fraction = N_mut / (N_w + N_mut)
                    vaf_expected = A * clone_fraction
                    vaf_expected = np.clip(vaf_expected, 1e-8, 1.0 - 1e-8)
                    
                    log_lik += binom.logpmf(int(AO_clone[tp_idx]), 
                                           int(DP_clone[tp_idx]), 
                                           vaf_expected)
                
                joint_log_likelihood[s_idx, A_idx] = log_lik
        
        # Normalize
        max_log_lik = joint_log_likelihood.max()
        joint_log_likelihood = joint_log_likelihood - max_log_lik
        joint_likelihood = np.exp(np.clip(joint_log_likelihood, -700, 0))
        
        # Uniform prior
        prior_s = np.ones_like(s_range) / s_range.shape[0]
        prior_A = np.ones_like(A_range) / A_range.shape[0]
        prior_joint = prior_s[:, None] * prior_A[None, :]
        
        joint_posterior = joint_likelihood * prior_joint
        Z = joint_posterior.sum()
        
        if Z == 0 or not np.isfinite(Z):
            print(f"  ⚠️  WARNING: Posterior normalization failed")
            joint_posterior = prior_joint
        else:
            joint_posterior = joint_posterior / Z
        
        # Marginalize
        s_posterior = joint_posterior.sum(axis=1)
        A_posterior = joint_posterior.sum(axis=0)
        
        s_posterior = s_posterior / (s_posterior.sum() + 1e-300)
        A_posterior = A_posterior / (A_posterior.sum() + 1e-300)
        
        # MAP estimates
        s_map_idx, A_map_idx = np.unravel_index(
            joint_posterior.argmax(), joint_posterior.shape
        )
        s_map = s_range[s_map_idx]
        A_map = A_range[A_map_idx]
        
        s_map_marginal = s_range[np.argmax(s_posterior)]
        A_map_marginal = A_range[np.argmax(A_posterior)]
        
        h_map = 2 * A_map_marginal - 1  # Derived
        
        # Credible intervals
        s_cumsum = np.cumsum(s_posterior)
        A_cumsum = np.cumsum(A_posterior)
        
        s_ci_low = s_range[np.searchsorted(s_cumsum, 0.05)]
        s_ci_high = s_range[np.searchsorted(s_cumsum, 0.95)]
        A_ci_low = A_range[np.searchsorted(A_cumsum, 0.05)]
        A_ci_high = A_range[np.searchsorted(A_cumsum, 0.95)]
        
        h_ci_low = 2 * A_ci_low - 1
        h_ci_high = 2 * A_ci_high - 1
        
        print(f"\n  Results:")
        print(f"    s = {s_map_marginal:.3f}  [90% CI: {s_ci_low:.3f} - {s_ci_high:.3f}]")
        print(f"    A = {A_map_marginal:.3f}  [90% CI: {A_ci_low:.3f} - {A_ci_high:.3f}]")
        print(f"    h = {h_map:.3f}  [90% CI: {h_ci_low:.3f} - {h_ci_high:.3f}] (derived)")
        
        results.append({
            's_map': s_map_marginal,
            'A_map': A_map_marginal,
            'h_map': h_map,
            's_posterior': s_posterior,
            'A_posterior': A_posterior,
            'joint_posterior': joint_posterior,
            's_range': s_range,
            'A_range': A_range,
            's_ci': (s_ci_low, s_ci_high),
            'A_ci': (A_ci_low, A_ci_high),
            'h_ci': (h_ci_low, h_ci_high),
        })
    
    print("\n" + "="*70)
    return results


def refine_optimal_model_posterior_vec(part, s_resolution=30, A_resolution=20):
    """
    HYBRID VERSION: Joint (s, A) inference
    """
    
    cs = list(part.uns['model_dict'].values())[0][0]

    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    deterministic_size, total_cells, max_total_per_mutation, clonal_map = \
        compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

    joint_results = infer_sA_jointly_from_dynamics(
        cs, AO, DP, time_points, 
        deterministic_size, total_cells, max_total_per_mutation,
        s_resolution=s_resolution, 
        A_resolution=A_resolution
    )

    A_vec = np.array([result['A_map'] for result in joint_results])
    h_vec = np.array([result['h_map'] for result in joint_results])
    A_posterior_list = [result['A_posterior'] for result in joint_results]
    s_vec = joint_results[0]['s_range']
    
    fitness_posterior = np.column_stack([result['s_posterior'] for result in joint_results])

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        'joint_inference': joint_results,
        'posterior': fitness_posterior,
        's_range': s_vec,
        'A_vec': A_vec,
        'h_vec': h_vec,
        'A_posterior': A_posterior_list,
    }

    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):
        result = joint_results[i]
        fitness[c_idx] = result['s_map']
        fitness_5[c_idx] = result['s_ci'][0]
        fitness_95[c_idx] = result['s_ci'][1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index

    mut_structure = part.uns['optimal_model']['mutation_structure']
    clonal_structure_list = []
    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)
                break

    part.obs['clonal_structure'] = clonal_structure_list

    return part


#endregion

#region Utility functions (from Doc 3 - working)

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller


def compute_model_likelihood(output, cs, s_range):
    clonal_prob = np.zeros(len(cs))
    s_range_size = s_range.max() - s_range.min()
    s_prior = 1/s_range_size
    
    for i, out in enumerate(output.T):
        out = np.array(out, copy=False)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.maximum(out, 0.0)
        integral = np.trapz(x=s_range, y=out)
        clonal_prob[i] = s_prior * max(integral, 0.0)

    model_probability = np.prod(clonal_prob)
    return model_probability


def compute_invalid_combinations(part, pearson_distance_threshold=0.5):
    correlation_matrix = np.corrcoef(np.vstack([part.X, part.var.time_points]))
    correlation_vec = correlation_matrix[-1, :-1]
    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])
    not_valid_comb = np.argwhere(distance_matrix > pearson_distance_threshold)
    
    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] and [i[1], i[0]] not in res:
            res.append(i.tolist()) 
    
    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    n_mutations = part.shape[0]

    if n_mutations == 1:
        valid_cs = [[[0]]]
        return valid_cs

    else:
        if filter_invalid is True:
            compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)
            
        a = list(partition(list(range(n_mutations))))
        cs_list = [cs for cs in a]

        if filter_invalid is False:
            return cs_list
        
        else:
            valid_cs = []
            for cs in cs_list:
                invalid_combinations_in_cs = 0
                for clone in cs:
                    mut_comb = list(combinations(clone, 2))
                    n_invalid_comb_in_clone = len(
                        [comb for comb in mut_comb 
                            if list(comb) in part.uns['invalid_combinations']])
                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                if invalid_combinations_in_cs == 0:
                    valid_cs.append(cs)
                    
            return valid_cs


def plot_optimal_model(part):
    if part.uns.get('warning') is not None:
        print('WARNING: ' + part.uns['warning'])
        
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    s_range = model['s_range']

    norm_max = np.max(output, axis=0)
    norm_max = np.where(norm_max == 0, 1.0, norm_max)

    import seaborn as sns
    for i in range(len(cs)):
        p_key_str = ''
        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f'{part[j].obs.p_key.values[0]}'
            if k > 0:
                p_key_str += f'\n{part[j].obs.p_key.values[0]}'

        sns.lineplot(x=s_range, y=output[:, i] / norm_max[i], label=p_key_str)

#endregion