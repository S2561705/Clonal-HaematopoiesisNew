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

# ==============================================================================
# UNIFIED ZYGOSITY MODEL
# ==============================================================================
# This script handles heterozygous, homozygous, and mixed zygosity mutations
# 
# Zygosity modes:
# - 'het': Heterozygous (h=0)
# - 'hom': Homozygous (h=1)  
# - 'mixed': Infer h from data (0 < h < 1)
# - 'auto': Automatically detect based on max VAF
# ==============================================================================

def compute_deterministic_size_unified(cs, AO, DP, n_mutations, zygosity_mode='auto', h_fixed=None):
    """
    Compute deterministic clone sizes with flexible zygosity handling.
    
    Uses PyMC-inspired approach: infers separate heterozygous and homozygous
    cell counts from VAF data.
    
    Key insight from PyMC model:
    - VAF = (H + 2*M) / (2*(N_w + H + M))
    - Where H = heterozygous cells, M = homozygous cells
    
    Parameters:
    -----------
    cs : list of lists
        Clonal structure (partition of mutations)
    AO : jnp.array
        Alternate allele counts (timepoints x mutations)
    DP : jnp.array
        Total depth (timepoints x mutations)
    n_mutations : int
        Number of mutations
    zygosity_mode : str or array-like
        - 'auto': Infer h from VAF patterns (default)
        - 'het': Force h=0 (heterozygous only)
        - 'hom': Force h=1 (homozygous only)
        - 'mixed': Infer h allowing mixed populations
        - array: Specify h per clone directly
    h_fixed : array-like, optional
        Fixed h values per clone (overrides zygosity_mode)
    
    Returns:
    --------
    deterministic_size : array
        Deterministic total mutation sizes (H+M) (timepoints x mutations)
    total_cells : array
        Total cells per timepoint (N_w + all mutants)
    h_vec : array
        Homozygous fraction per clone (M/(H+M))
    het_frac : array
        Heterozygous cell counts H (timepoints x mutations)
    hom_frac : array
        Homozygous cell counts M (timepoints x mutations)
    """
    N_w = 1e5
    n_clones = len(cs)
    
    # Create clonal map
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    lm = []  # leading mutations
    
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))
    
    vaf_all = AO / DP
    
    # Determine h (homozygous fraction) per clone
    if h_fixed is not None:
        h_vec = jnp.array(h_fixed)
    elif isinstance(zygosity_mode, (list, tuple, jnp.ndarray, np.ndarray)):
        h_vec = jnp.array(zygosity_mode)
    else:
        if zygosity_mode == 'het':
            h_vec = jnp.zeros(n_clones)
        elif zygosity_mode == 'hom':
            h_vec = jnp.ones(n_clones)
        elif zygosity_mode in ['mixed', 'auto']:
            # Improved estimation following PyMC approach:
            # For each clone, estimate h from the VAF pattern
            
            h_vec = jnp.zeros(n_clones)
            for i, cs_idx in enumerate(cs):
                # Get VAF for mutations in this clone
                vaf_clone = vaf_all[:, cs_idx]
                max_vaf = jnp.max(vaf_clone)
                
                # Key insight: 
                # - Pure het: VAF_max ≤ 0.5 (always, regardless of clone size)
                # - Pure hom: VAF_max → 1.0 as clone → 100% of population
                # - Mixed: 0.5 < VAF_max < 1.0
                
                # For mixed populations with homozygous fraction h:
                # VAF = (H + 2*M) / (2*(N_w + H + M))
                # If X = H + M (total mutant cells), and M = h*X, H = (1-h)*X:
                # VAF = ((1-h)*X + 2*h*X) / (2*(N_w + X))
                #     = X*(1+h) / (2*(N_w + X))
                
                # When clone is large (X >> N_w): VAF → (1+h)/2
                # So: h ≈ 2*VAF_max - 1
                
                # Conservative thresholds accounting for noise:
                if max_vaf <= 0.52:  # Allow 2% margin for sequencing noise
                    h_est = 0.0  # Pure heterozygous
                elif max_vaf >= 0.85:  # High VAF suggests pure homozygous
                    h_est = 1.0  # Pure homozygous
                else:
                    # Estimate h, but be conservative
                    # Use the maximum VAF as proxy for (1+h)/2
                    h_est = jnp.clip(2 * max_vaf - 1, 0, 1)
                
                h_vec = h_vec.at[i].set(h_est)
        else:
            raise ValueError(f"Invalid zygosity_mode: {zygosity_mode}")
    
    # Compute deterministic clone sizes
    # Following PyMC: VAF = (H + 2*M) / (2*(N_w + H + M))
    # Where total clone size X = H + M, and h = M/X
    # So: VAF = X*(1+h) / (2*(N_w + X))
    # Solving for X: X = 2*N_w*VAF / (1+h - 2*VAF)
    
    vaf_leading = vaf_all[:, lm]  # shape: (n_timepoints, n_clones)
    sum_vaf = jnp.sum(vaf_leading, axis=1)  # shape: (n_timepoints,)
    
    # Sum h terms across clones for the denominator
    sum_h_term = jnp.sum((1 + h_vec) / 2)  # scalar
    
    deterministic_clone_size = -N_w * sum_vaf / (sum_vaf - sum_h_term)
    deterministic_clone_size = jnp.ceil(jnp.maximum(deterministic_clone_size, 0))
    total_cells = N_w + deterministic_clone_size
    
    # Compute deterministic size per mutation (total X = H + M)
    h_per_mutation = h_vec[clonal_map]
    deterministic_size = vaf_all * 2 * total_cells[:, None] / (1 + h_per_mutation[None, :])
    
    # Decompose into heterozygous and homozygous components
    # Following PyMC: H = X*(1-h), M = X*h
    het_frac = deterministic_size * (1 - h_per_mutation[None, :])
    hom_frac = deterministic_size * h_per_mutation[None, :]
    
    return deterministic_size, total_cells, h_vec, het_frac, hom_frac, clonal_map


@jit
def compute_global_variables_unified(s_vec, AO, DP, total_cells, 
                                    deterministic_size, time_points, 
                                    h_per_mutation, resolution=1_000):
    """
    Compute global variables for unified zygosity model.
    
    The key transformation is:
    - VAF to clone size: x = 2*N_w*VAF / (1+h - 2*VAF)
    - Clone size to VAF: VAF = x*(1+h) / (2*(N_w + x))
    
    Where h is the homozygous fraction (0=het, 1=hom, 0<h<1=mixed)
    """
    n_mutations = AO.shape[1]
    
    # BD process exponential term
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))
    
    # Sample latent VAFs from Beta posterior
    beta_p_rvs_vec = jrnd.beta(
        key=key, 
        a=(AO+1)[:, :, None],
        b=(DP - AO + 1)[:, :, None],
        shape=(AO.shape[0], AO.shape[1], resolution)
    )
    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)
    
    # Compute N_w conditional (wild-type + other clones)
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    
    # Transform VAF to total clone size using unified formula
    # x = 2*N_w*VAF / (1+h - 2*VAF)
    h_broadcast = h_per_mutation[None, :, None]
    
    denominator = (1 + h_broadcast - 2 * beta_p_rvs_vec)
    # Avoid division by zero (happens when VAF ≈ (1+h)/2)
    denominator = jnp.where(jnp.abs(denominator) < 1e-10, 1e-10, denominator)
    
    x_range_vec = 2 * N_w_cond_vec * beta_p_rvs_vec / denominator
    x_vec = jnp.ceil(jnp.maximum(x_range_vec, 1))  # Ensure positive
    
    # Compute true VAF from clone size
    # VAF = x*(1+h) / (2*(N_w + x))
    true_vaf_vec = x_vec * (1 + h_broadcast) / (2 * (N_w_cond_vec + x_vec))
    
    # Observation likelihoods
    p_y_cond_x_vec = jsp_stats.binom.pmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)
    
    # Initialize recursive term with first timepoint
    recursive_term_vec = p_y_cond_x_vec[0, :, :] / resolution
    
    return (x_vec, exp_term_vec_s, recursive_term_vec, 
            p_y_cond_x_vec, n_mutations)


@jit
def BD_process_dynamics(s, x_vec, exp_term_vec, lamb=1.3):
    """Birth-death process dynamics (same for all zygosity modes)"""
    mean_vec = x_vec[:-1, :, :] * exp_term_vec
    variance_vec = x_vec[:-1] * (2*lamb + s) * exp_term_vec * (exp_term_vec - 1) / s
    
    # Negative binomial parametrization
    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec - mean_vec)
    
    return p_vec, n_vec


@jit
def recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i):
    """Update recursive term for HMM forward algorithm"""
    bd_pmf_i = jsp_stats.nbinom.pmf(x_i[j][:, None], p=p_i[j-1], n=n_i[j-1])
    inner_sum_i = bd_pmf_i * recursive_term_i
    recursive_term_i = p_y_cond_x_i[j] * jsp.integrate.trapezoid(x=x_i[j-1], y=inner_sum_i)
    return recursive_term_i


def mutation_specific_ll(i, recursive_term_vec, x_vec, p_vec, n_vec, 
                        p_y_cond_x_vec, n_tps):
    """Compute likelihood for a single mutation"""
    recursive_term_i = recursive_term_vec[i]
    x_i = x_vec[:, i]
    p_i = p_vec[:, i]
    n_i = n_vec[:, i]
    p_y_cond_x_i = p_y_cond_x_vec[:, i]
    
    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i)
    
    return jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)


def fitness_specific_computations(s_idx, s_vec, x_vec, exp_term_vec_s, 
                                 recursive_term_vec, p_y_cond_x_vec, 
                                 time_points, n_mutations, cs):
    """Compute likelihood for specific fitness value"""
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]
    
    p_vec, n_vec = BD_process_dynamics(s, x_vec, exp_term_vec)
    
    mutation_likelihood = jax.vmap(
        mutation_specific_ll,
        in_axes=(0, None, None, None, None, None, None)
    )(
        jnp.arange(n_mutations, dtype=int),
        recursive_term_vec, x_vec, p_vec, n_vec, p_y_cond_x_vec, time_points.shape[0]
    )
    
    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            jnp.prod(mutation_likelihood[jnp.array(c_idx)])
        )
    
    return clonal_likelihood


def jax_cs_hmm_ll_vec_unified(s_vec, AO, DP, time_points, cs, 
                              deterministic_size, total_cells, h_per_mutation):
    """
    Vectorized likelihood computation for unified zygosity model.
    
    Parameters:
    -----------
    h_per_mutation : array
        Homozygous fraction for each mutation (length: n_mutations)
    """
    global_variables = compute_global_variables_unified(
        s_vec, AO, DP, total_cells, deterministic_size, 
        time_points, h_per_mutation
    )
    x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mutations = global_variables
    
    s_idx = jnp.arange(s_vec.shape[0])
    
    clonal_likelihood = jax.vmap(
        fitness_specific_computations,
        in_axes=(0, None, None, None, None, None, None, None, None)
    )(
        s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec,
        p_y_cond_x_vec, time_points, n_mutations, cs
    )
    
    return clonal_likelihood


# ==============================================================================
# Model comparison and inference
# ==============================================================================

def compute_model_likelihood(output, cs, s_range):
    """Compute marginal likelihood of clonal structure"""
    clonal_prob = np.zeros(len(cs))
    s_range_size = s_range.max() - s_range.min()
    s_prior = 1 / s_range_size
    
    for i, out in enumerate(output.T):
        clonal_prob[i] = s_prior * np.trapz(x=s_range, y=out)
    
    model_probability = np.prod(clonal_prob)
    return model_probability


def compute_clonal_models_prob_vec_unified(part, s_resolution=50, min_s=0.01, 
                                          max_s=3, zygosity_mode='auto',
                                          h_resolution=5,  # NEW: resolution for h grid
                                          filter_invalid=True, 
                                          disable_progressbar=False):
    """
    Compute model probabilities for all valid clonal structures.
    
    Parameters:
    -----------
    part : AnnData-like object
        Participant data with layers['AO'], layers['DP'], var.time_points
    s_resolution : int
        Number of fitness values to evaluate
    min_s, max_s : float
        Fitness range
    zygosity_mode : str or array
        'auto': Grid search over h values (recommended)
        'het': Force h=0
        'hom': Force h=1
        'mixed': Grid search with expanded h range
        or array of h values per clone
    h_resolution : int
        Number of h values to test in grid (only used if zygosity_mode='auto' or 'mixed')
    filter_invalid : bool
        Filter clonal structures using correlation analysis
    """
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
    
    for i, cs in tqdm(enumerate(cs_list), disable=disable_progressbar, total=len(cs_list)):
        # Grid search over h values if in auto or mixed mode
        if zygosity_mode in ['auto', 'mixed']:
            best_h_vec = None
            best_prob = -np.inf
            
            # Create h grid - test multiple h values per clone
            n_clones = len(cs)
            h_values = jnp.linspace(0, 1, h_resolution)
            
            # For efficiency, test common scenarios first
            if h_resolution == 5:
                # Test: all-het, all-hom, and mixed combinations
                test_h_configs = [
                    jnp.zeros(n_clones),  # All het
                    jnp.ones(n_clones),   # All hom
                ]
                # Add some mixed configurations
                for h_val in [0.25, 0.5, 0.75]:
                    test_h_configs.append(jnp.ones(n_clones) * h_val)
            else:
                # Full grid if resolution is higher
                # For computational efficiency, test all clones with same h
                test_h_configs = [jnp.ones(n_clones) * h for h in h_values]
            
            # Test each h configuration
            for h_vec_test in test_h_configs:
                try:
                    deterministic_size, total_cells, _, het_frac, hom_frac, clonal_map = \
                        compute_deterministic_size_unified(cs, AO, DP, n_mutations, 
                                                          zygosity_mode=h_vec_test)
                    
                    h_per_mutation = h_vec_test[clonal_map]
                    
                    output = jax_cs_hmm_ll_vec_unified(
                        s_vec, AO, DP, time_points, cs, 
                        deterministic_size, total_cells, h_per_mutation
                    )
                    
                    model_prob = compute_model_likelihood(output, cs, s_vec)
                    
                    if model_prob > best_prob:
                        best_prob = model_prob
                        best_h_vec = h_vec_test
                except Exception as e:
                    # Skip invalid h configurations
                    continue
            
            if best_h_vec is None:
                print(f"  ⚠️ Warning: No valid h configuration found for model {i}")
                continue
            
            # Store best result
            part.uns['model_dict'][f'model_{i}'] = (cs, best_prob, best_h_vec)
            
        else:
            # Original behavior for fixed zygosity modes
            deterministic_size, total_cells, h_vec, het_frac, hom_frac, clonal_map = \
                compute_deterministic_size_unified(cs, AO, DP, n_mutations, zygosity_mode)
            
            h_per_mutation = h_vec[clonal_map]
            
            output = jax_cs_hmm_ll_vec_unified(
                s_vec, AO, DP, time_points, cs, 
                deterministic_size, total_cells, h_per_mutation
            )
            
            model_prob = compute_model_likelihood(output, cs, s_vec)
            part.uns['model_dict'][f'model_{i}'] = (cs, model_prob, h_vec)
    
    # Sort by probability
    part.uns['model_dict'] = {
        k: v for k, v in sorted(part.uns['model_dict'].items(), 
                               key=lambda item: item[1][1], reverse=True)
    }
    
    return part


def refine_optimal_model_posterior_vec_unified(part, s_resolution=100, 
                                               zygosity_mode='auto',
                                               h_resolution=11):
    """
    Refine posterior for optimal model with higher resolution.
    
    If zygosity_mode='auto' or 'mixed', performs grid search over h values.
    
    Stores results in part.obs and part.uns['optimal_model']
    """
    # Extract optimal clonal structure and h values if available
    optimal_data = list(part.uns['model_dict'].values())[0]
    
    if len(optimal_data) == 3:
        cs, _, h_vec_initial = optimal_data
    else:
        cs, _ = optimal_data
        h_vec_initial = None
    
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)
    n_mutations = AO.shape[1]
    n_clones = len(cs)
    
    # Refined fitness range
    s_vec = jnp.linspace(0.01, 3, s_resolution)
    
    # Grid search over h if in auto/mixed mode
    if zygosity_mode in ['auto', 'mixed'] and h_vec_initial is not None:
        # Use initial h as starting point, but refine with grid search
        best_h_vec = h_vec_initial
        best_output = None
        best_likelihood = -np.inf
        
        # Create finer h grid around initial values
        h_values = jnp.linspace(0, 1, h_resolution)
        
        # Test configurations
        test_h_configs = []
        
        # 1. Test initial h
        test_h_configs.append(h_vec_initial)
        
        # 2. Test uniform h values (all clones same h)
        for h_val in h_values:
            test_h_configs.append(jnp.ones(n_clones) * h_val)
        
        # 3. Test per-clone variations if computational budget allows
        if n_clones <= 3 and h_resolution <= 5:
            # Test some combinations with different h per clone
            for h1 in [0.0, 0.5, 1.0]:
                for h2 in [0.0, 0.5, 1.0]:
                    if n_clones == 2:
                        test_h_configs.append(jnp.array([h1, h2]))
                    elif n_clones >= 3:
                        for h3 in [0.0, 0.5, 1.0]:
                            test_h_configs.append(jnp.array([h1, h2, h3]))
        
        # Evaluate each h configuration
        for h_vec_test in test_h_configs:
            if len(h_vec_test) != n_clones:
                continue
                
            try:
                deterministic_size, total_cells, _, het_frac, hom_frac, clonal_map = \
                    compute_deterministic_size_unified(cs, AO, DP, n_mutations, 
                                                      zygosity_mode=h_vec_test)
                
                h_per_mutation = h_vec_test[clonal_map]
                
                output = jax_cs_hmm_ll_vec_unified(
                    s_vec, AO, DP, time_points, cs, 
                    deterministic_size, total_cells, h_per_mutation
                )
                
                # Compute total likelihood
                total_likelihood = np.sum([np.trapz(y=output[:, i], x=s_vec) 
                                          for i in range(n_clones)])
                
                if total_likelihood > best_likelihood:
                    best_likelihood = total_likelihood
                    best_h_vec = h_vec_test
                    best_output = output
            except Exception as e:
                continue
        
        if best_output is None:
            print(f"  ⚠️ Warning: Grid search over h failed, using initial h")
            h_vec = h_vec_initial
            deterministic_size, total_cells, _, het_frac, hom_frac, clonal_map = \
                compute_deterministic_size_unified(cs, AO, DP, n_mutations, 
                                                  zygosity_mode=h_vec_initial)
            h_per_mutation = h_vec[clonal_map]
            output = jax_cs_hmm_ll_vec_unified(
                s_vec, AO, DP, time_points, cs, 
                deterministic_size, total_cells, h_per_mutation
            )
        else:
            h_vec = best_h_vec
            output = best_output
            # Recompute deterministic sizes with best h
            deterministic_size, total_cells, _, het_frac, hom_frac, clonal_map = \
                compute_deterministic_size_unified(cs, AO, DP, n_mutations, 
                                                  zygosity_mode=h_vec)
    else:
        # Use fixed zygosity mode
        deterministic_size, total_cells, h_vec, het_frac, hom_frac, clonal_map = \
            compute_deterministic_size_unified(cs, AO, DP, n_mutations, zygosity_mode)
        
        h_per_mutation = h_vec[clonal_map]
        
        output = jax_cs_hmm_ll_vec_unified(
            s_vec, AO, DP, time_points, cs, 
            deterministic_size, total_cells, h_per_mutation
        )
    
    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        'posterior': output,
        's_range': s_vec,
        'h_vec': h_vec,
        'h_per_mutation': h_per_mutation,
        'het_frac': het_frac,
        'hom_frac': hom_frac,
        'clonal_map': clonal_map,
        'zygosity_mode': zygosity_mode
    }
    
    # Extract results per mutation
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])
    homozygous_fraction = np.zeros(part.shape[0])
    het_cells_mean = np.zeros(part.shape[0])
    hom_cells_mean = np.zeros(part.shape[0])
    
    for i, c_idx in enumerate(cs):
        p = np.array(output[:, i])
        
        # Check for invalid posterior
        if np.nansum(p) == 0 or np.any(~np.isfinite(p)):
            part.uns['warning'] = 'Zero or invalid posterior'
            # Set default values for this clone
            for mut_idx in c_idx:
                fitness[mut_idx] = np.nan
                fitness_5[mut_idx] = np.nan
                fitness_95[mut_idx] = np.nan
                clonal_index[mut_idx] = i
                homozygous_fraction[mut_idx] = h_vec[i]
                het_cells_mean[mut_idx] = np.nan
                hom_cells_mean[mut_idx] = np.nan
            continue
        
        # Clip negative values and normalize
        p = np.maximum(p, 0)
        p_sum = p.sum()
        
        if p_sum == 0 or not np.isfinite(p_sum):
            part.uns['warning'] = 'Zero or invalid posterior after clipping'
            for mut_idx in c_idx:
                fitness[mut_idx] = np.nan
                fitness_5[mut_idx] = np.nan
                fitness_95[mut_idx] = np.nan
                clonal_index[mut_idx] = i
                homozygous_fraction[mut_idx] = h_vec[i]
                het_cells_mean[mut_idx] = np.nan
                hom_cells_mean[mut_idx] = np.nan
            continue
            
        p /= p_sum
        
        # Additional check after normalization
        if np.any(p < 0) or not np.all(np.isfinite(p)):
            print(f"  ⚠️ Warning: Invalid probabilities for clone {i} after normalization")
            for mut_idx in c_idx:
                fitness[mut_idx] = np.nan
                fitness_5[mut_idx] = np.nan
                fitness_95[mut_idx] = np.nan
                clonal_index[mut_idx] = i
                homozygous_fraction[mut_idx] = h_vec[i]
                het_cells_mean[mut_idx] = np.nan
                hom_cells_mean[mut_idx] = np.nan
            continue
        
        fitness_map = s_vec[np.argmax(p)]
        
        # Bootstrap confidence intervals
        try:
            sample_range = np.random.choice(s_vec, p=p, size=1_000)
            cfd_int = np.quantile(sample_range, [0.05, 0.95])
        except Exception as e:
            print(f"  ⚠️ Warning: Could not compute CI for clone {i}: {e}")
            cfd_int = [fitness_map, fitness_map]
        
        for mut_idx in c_idx:
            fitness[mut_idx] = fitness_map
            fitness_5[mut_idx] = cfd_int[0]
            fitness_95[mut_idx] = cfd_int[1]
            clonal_index[mut_idx] = i
            homozygous_fraction[mut_idx] = h_vec[i]
            het_cells_mean[mut_idx] = np.mean(het_frac[:, mut_idx])
            hom_cells_mean[mut_idx] = np.mean(hom_frac[:, mut_idx])
    
    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index
    part.obs['homozygous_fraction'] = homozygous_fraction
    part.obs['het_cells_mean'] = het_cells_mean
    part.obs['hom_cells_mean'] = hom_cells_mean
    
    # Classify zygosity
    zygosity_type = []
    for h in homozygous_fraction:
        if np.isnan(h):
            zygosity_type.append('unknown')
        elif h < 0.1:
            zygosity_type.append('heterozygous')
        elif h > 0.9:
            zygosity_type.append('homozygous')
        else:
            zygosity_type.append('mixed')
    part.obs['zygosity_type'] = zygosity_type
    
    # Mutational structure
    mut_structure = part.uns['optimal_model']['mutation_structure']
    clonal_structure_list = []
    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)
                break
    
    part.obs['clonal_structure'] = clonal_structure_list
    
    return part


# ==============================================================================
# Utilities
# ==============================================================================

def partition(collection):
    """Generate all partitions of a set"""
    if len(collection) == 1:
        yield [collection]
        return
    
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller


def compute_invalid_combinations(part, pearson_distance_threshold=0.5):
    """Identify mutation pairs with incompatible temporal dynamics"""
    correlation_matrix = np.corrcoef(np.vstack([part.X, part.var.time_points]))
    correlation_vec = correlation_matrix[-1, :-1]
    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])
    
    not_valid_comb = np.argwhere(distance_matrix > pearson_distance_threshold)
    not_valid_comb = [list(i) for i in not_valid_comb]
    
    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] not in res and [i[1], i[0]] not in res:
            res.append(i)
    
    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """Find valid clonal structures using correlation filtering"""
    n_mutations = part.shape[0]
    
    if n_mutations == 1:
        return [[[0]]]
    
    if filter_invalid:
        compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)
    
    cs_list = list(partition(list(range(n_mutations))))
    
    if not filter_invalid:
        return cs_list
    
    valid_cs = []
    for cs in cs_list:
        invalid_count = 0
        for clone in cs:
            mut_comb = list(combinations(clone, 2))
            invalid_count += len([comb for comb in mut_comb 
                                 if list(comb) in part.uns['invalid_combinations']])
        
        if invalid_count == 0:
            valid_cs.append(cs)
    
    return valid_cs


def plot_optimal_model(part, show_zygosity=True):
    """Plot fitness posteriors for optimal model"""
    if part.uns.get('warning'):
        print('WARNING: ' + part.uns['warning'])
    
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    ms = model['mutation_structure']
    s_range = model['s_range']
    h_vec = model['h_vec']
    
    norm_max = np.max(output, axis=0)
    
    for i in range(len(cs)):
        label_parts = []
        for j in cs[i]:
            label_parts.append(part.obs.index[j])
        
        label = '\n'.join(label_parts)
        
        if show_zygosity:
            h = h_vec[i]
            if h < 0.1:
                zyg_label = 'het'
            elif h > 0.9:
                zyg_label = 'hom'
            else:
                zyg_label = f'mixed (h={h:.2f})'
            label += f'\n({zyg_label})'
        
        sns.lineplot(x=s_range, y=output[:, i] / norm_max[i], label=label)
    
    plt.xlabel('Fitness (s)')
    plt.ylabel('Normalized Posterior')
    plt.title('Clonal Fitness Posteriors')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


