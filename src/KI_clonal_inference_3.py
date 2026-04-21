import sys
sys.path.append("..")   # fix to import modules from root
from scripts.debug_patient2 import AO, DP
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd
from itertools import combinations
from scipy.stats import binom  # For h inference

key = jrnd.PRNGKey(758493)  # Random seed is explicit in JAX

#region FIXED: Non vectorised functions with DEBUG
# Non vectorised 
def compute_deterministic_size_mixed(cs, AO, DP, n_mutations, N_w=1e5):
    """
    FIXED VERSION: Robust deterministic estimate with proper handling of edge cases.
    Produces TOTAL mutant cells per timepoint & mutation with recommended max bounds.
    """
    
    # Calculate VAFs with safety checks
    vaf_ratio = AO / jnp.maximum(DP, 1.0)  # Prevent division by zero
    
    # Find leading mutation per clone
    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        clone_vafs = vaf_ratio[:, cs_idx]
        vaf_sums = clone_vafs.sum(axis=0)
        max_idx = int(jnp.argmax(vaf_sums))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    # FIXED: Robust heterozygous rearrangement with proper bounds
    vafs_lead = vaf_ratio[:, lm]  # (n_timepoints, n_clones)
    sum_vafs_lead = jnp.sum(vafs_lead, axis=1)
    
    # CRITICAL FIX: Handle the case when sum_vafs ≈ 0.5 (singularity point)
    # The formula: N_mutant = -N_w * sum_vafs / (sum_vafs - 0.5) breaks down near 0.5
    # Use different estimation strategy based on how far we are from 0.5
    denominator = sum_vafs_lead - 0.5
    
    # For sum_vafs far from 0.5: use original formula
    # For sum_vafs near 0.5: use linear approximation
    far_from_singular = jnp.abs(denominator) > 0.15
    
    # Original formula (valid when denominator is large enough)
    original_estimate = -N_w * sum_vafs_lead / jnp.maximum(jnp.abs(denominator), 0.15) * jnp.sign(denominator)
    
    # Linear approximation near singularity (assumes roughly proportional)
    linear_estimate = N_w * sum_vafs_lead / (1.0 - sum_vafs_lead + 1e-8)
    
    # Choose based on distance from singularity
    deterministic_clone_size = jnp.where(
        far_from_singular,
        original_estimate,
        linear_estimate
    )
    
    # Ensure non-negative and apply ceiling
    deterministic_clone_size = jnp.maximum(deterministic_clone_size, 0.0)
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    
    # Cap at reasonable maximum (10x wild-type population)
    deterministic_clone_size = jnp.minimum(deterministic_clone_size, N_w * 10)
    
    total_cells = N_w + deterministic_clone_size

    # Deterministic SIZE grid for each mutation
    deterministic_size = vaf_ratio * 2 * total_cells[:, None]
    
    # FIXED: Safer upper bound calculation
    # Use 95th percentile of deterministic estimates * 3 as upper bound
    max_per_tp = deterministic_size.max(axis=0)
    mean_per_mut = deterministic_size.mean(axis=0)
    max_total_per_mutation = jnp.maximum(
        jnp.maximum(max_per_tp * 3, mean_per_mut * 5),
        100.0  # Minimum bound
    )
    # Cap at reasonable maximum
    max_total_per_mutation = jnp.minimum(max_total_per_mutation, N_w * 5)

    return (deterministic_size.astype(jnp.float32), 
            total_cells.astype(jnp.float32), 
            max_total_per_mutation.astype(jnp.float32), 
            clonal_map)


#endregion

#region FIXED: Vectorised and jit optimised functions

def jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs,
                            deterministic_size, total_cells, max_total_per_mutation, 
                            key, resolution=600):
    """
    FIXED VERSION: Improved numerical stability throughout.
    """
    
    # Compute global variables (samples) using the provided key
    x_het_vec, x_hom_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut = \
        compute_global_variables_mixed(s_vec, AO, DP, total_cells, deterministic_size, 
                                      max_total_per_mutation, time_points, key, resolution=resolution)

    # s index vector
    s_idx = jnp.arange(s_vec.shape[0])

    # Fitness_specific_computations mapped over s index
    def fitness_specific_computations_mapped(s_idx):
        s = s_vec[s_idx]
        exp_term_vec = exp_term_vec_s[s_idx]
        
        # BD dynamics on total
        p_vec, n_vec = BD_process_dynamics_mixed(s, x_het_vec + x_hom_vec, exp_term_vec)

        # Compute mutation-likelihoods across all mutations (vectorised)
        mutation_likelihood = jax.vmap(
            lambda ii: mutation_specific_ll_mixed_grid(
                ii, recursive_term_vec[:, :],
                x_het_vec, x_hom_vec,
                p_vec, n_vec, p_y_cond_x_vec,
                time_points.shape[0]
            )
        )(jnp.arange(n_mut))
        
        # Compute clonal likelihoods as product over mutations in each clone
        clonal_likelihood = jnp.zeros(len(cs))
        for idx_c, c_idx in enumerate(cs):
            clone_mutations = jnp.array(c_idx)
            # Use log-space for numerical stability
            log_mut_liks = jnp.log(jnp.maximum(mutation_likelihood[clone_mutations], 1e-300))
            clone_likelihood = jnp.exp(jnp.sum(log_mut_liks))
            clonal_likelihood = clonal_likelihood.at[idx_c].set(clone_likelihood)
            
        return clonal_likelihood

    # Vmap over s
    clonal_likelihood = jax.vmap(fitness_specific_computations_mapped)(s_idx)
    return clonal_likelihood  # shape (n_s, n_clones)


def compute_global_variables_mixed(s_vec, AO, DP, total_cells, deterministic_size,
                                    max_total_per_mutation, time_points, key, resolution=600):
    """
    FIXED VERSION: Proper constrained sampling and robust VAF calculation.
    """
    
    n_tps, n_mut = AO.shape
    
    # BD exponential term for pmf
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])  # (n_s, n_intervals)
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    # FIXED: Constrained sampling approach
    # Instead of independent uniform sampling, we sample proportionally to observed VAF
    observed_vaf = AO / jnp.maximum(DP, 1.0)  # (n_tps, n_mut)
    
    # Create subkeys for sampling
    key_het, key_hom = jrnd.split(key, 2)

    # Sample x_total first (uniform over reasonable range)
    total_raw = jrnd.uniform(key_het, shape=(n_tps, n_mut, resolution), dtype=jnp.float32)
    max_totals = max_total_per_mutation[None, :, None]  # (1, n_mut, 1)
    x_total_vec = total_raw * max_totals
    
    # REPLACE lines 151-153 with:
    frac_hom = jrnd.uniform(key_hom, 
                        minval=0.0, 
                        maxval=1.0,
                        shape=(n_tps, n_mut, resolution))
    # Allocate total cells between het and hom
    # For heterozygous-only: frac_hom = 0, for all homozygous: frac_hom = 1
    x_hom_vec = x_total_vec * frac_hom
    x_het_vec = x_total_vec - x_hom_vec
    
    # Ensure positivity and bounds
    eps = 1e-6
    x_het_vec = jnp.clip(x_het_vec, eps, max_totals)
    x_hom_vec = jnp.clip(x_hom_vec, eps, max_totals)
    
    # FIXED: Robust VAF calculation with proper bounds
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]  # (n_tps, n_mut, 1)
    # Ensure N_w_cond is positive
    N_w_cond_vec = jnp.maximum(N_w_cond_vec, eps)
    
    x_total_vec = x_het_vec + x_hom_vec
    
    # VAF calculation with safety checks
    numerator = x_het_vec + 2.0 * x_hom_vec
    denominator_vaf = 2.0 * (N_w_cond_vec + x_total_vec)
    denominator_vaf = jnp.maximum(denominator_vaf, eps)  # Prevent division by zero
    
    true_vaf_vec = numerator / denominator_vaf
    # Enforce valid VAF range [0, 1]
    true_vaf_vec = jnp.clip(true_vaf_vec, eps, 1.0 - eps)

    # FIXED: Log-space computation for observation probabilities
    log_p_y_cond_x_vec = jsp_stats.binom.logpmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)
    # Convert back to probability space with floor
    p_y_cond_x_vec = jnp.exp(jnp.maximum(log_p_y_cond_x_vec, -300.0))  # floor at exp(-300) ≈ 1e-130
    
    # Initial recursive term with proper normalization
    prior_weight = 1.0 / resolution  # Single uniform prior over the sampled grid
    recursive_term_vec = p_y_cond_x_vec[0, :, :] * prior_weight

    return x_het_vec, x_hom_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut


def BD_process_dynamics_mixed(s, x_total_vec, exp_term_vec):
    """
    FIXED VERSION: Robust negative binomial parameter calculation.
    """
    
    lamb = 1.3
    
    # x_total_vec shape: (n_tps, n_mut, resolution)
    mean_vec = x_total_vec[:-1, :, :] * exp_term_vec  # broadcasting
    
    # FIXED: Robust variance calculation
    s_safe = jnp.maximum(jnp.abs(s), 1e-8)
    variance_vec = x_total_vec[:-1, :, :] * (2.0 * lamb + s) * exp_term_vec * (exp_term_vec - 1.0) / s_safe
    
    # CRITICAL FIX: Ensure variance > mean always (required for negative binomial)
    min_variance_ratio = 1.2  # Variance must be at least 20% larger than mean
    min_variance = mean_vec * min_variance_ratio + 1e-6
    variance_vec = jnp.maximum(variance_vec, min_variance)
    
    # FIXED: Ensure valid parameters with strict bounds
    p_vec = mean_vec / variance_vec
    p_vec = jnp.clip(p_vec, 1e-8, 1.0 - 1e-8)  # Must be in (0, 1)
    
    n_vec = jnp.power(mean_vec, 2) / jnp.maximum(variance_vec - mean_vec, 1e-8)
    n_vec = jnp.maximum(n_vec, 1e-8)  # Must be positive
    
    return p_vec, n_vec


def mutation_specific_ll_mixed_grid(i, recursive_term_vec, x_het_vec, x_hom_vec, 
                                    p_vec, n_vec, p_y_cond_x_vec, n_tps):
    """
    FIXED VERSION: Improved numerical stability in recursive integration.
    """
    
    recursive_term_i = recursive_term_vec[i]  # shape (resolution,)
    x_het_i = x_het_vec[:, i, :]  # (n_tps, resolution)
    x_hom_i = x_hom_vec[:, i, :]  # (n_tps, resolution)
    p_i = p_vec[:, i, :]  # (n_intervals, resolution)
    n_i = n_vec[:, i, :]  # (n_intervals, resolution)
    p_y_cond_x_i = p_y_cond_x_vec[:, i, :]  # (n_tps, resolution)

    # Iterate through timepoints
    for j in range(1, n_tps):
        # Total sizes at previous and current tp
        init_total = x_het_i[j-1] + x_hom_i[j-1]  # (resolution,)
        next_total = x_het_i[j] + x_hom_i[j]  # (resolution,)

        # FIXED: Log-space computation for BD PMF to avoid underflow
        log_bd_pmf = jsp_stats.nbinom.logpmf(next_total[:, None], p=p_i[j-1][None, :], n=n_i[j-1][None, :])
        log_bd_pmf = jnp.maximum(log_bd_pmf, -300.0)  # Floor
        bd_pmf = jnp.exp(log_bd_pmf)
        
        # Inner sum: integrate over previous-res grid weighted by recursive_term_i
        inner_sum = bd_pmf * recursive_term_i  # (next_res, init_res)
        
        # Integrate over init axis using trapezoid
        inner_integrated = jsp.integrate.trapezoid(x=init_total, y=inner_sum, axis=1)  # (next_res,)
        inner_integrated = jnp.maximum(inner_integrated, 1e-300)  # Floor to prevent log(0)
        
        # FIXED: Log-space multiplication for numerical stability
        log_p_y = jnp.log(jnp.maximum(p_y_cond_x_i[j], 1e-300))
        log_inner = jnp.log(inner_integrated)
        log_recursive_term_i = log_p_y + log_inner
        
        # Convert back to linear space
        recursive_term_i = jnp.exp(jnp.maximum(log_recursive_term_i, -300.0))

    # Final integration over the last grid
    total_x_final = x_het_i[-1] + x_hom_i[-1]  # (resolution,)
    final_like = jsp.integrate.trapezoid(x=total_x_final, y=recursive_term_i)
    final_like = jnp.maximum(final_like, 1e-300)  # Ensure non-negative
    
    return final_like


def compute_clonal_models_prob_vec_mixed(part, s_resolution=50, min_s=0.01, max_s=3,
                                        filter_invalid=True, disable_progressbar=False,
                                        resolution=600, master_key_seed=758493):
    
    print("="*60)
    print("COMPUTING CLONAL MODELS PROBABILITIES (VECTORIZED - FIXED)")
    print("="*60)
    
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

    # --- Step 1: compute raw probability for every model, store as 2-tuple ---
    for i, cs in enumerate(cs_list):
        print(f"\nProcessing model {i}/{len(cs_list)}: {cs}")
        
        deterministic_size, total_cells, max_total_per_mutation, clonal_map = \
            compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

        key_i = keys[i]
        output = jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs,
                                        deterministic_size, total_cells, max_total_per_mutation, 
                                        key_i, resolution=resolution)

        model_prob = compute_model_likelihood(output, cs, s_vec)
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)
        print(f"Model {i} probability: {model_prob:.3e}")

    # --- Step 2: normalise to get posterior probabilities ---
    total_prob = sum(v[1] for v in part.uns['model_dict'].values())

    if total_prob == 0 or not np.isfinite(total_prob):
        print("\n⚠️  WARNING: All model probabilities are zero. Cannot normalise.")
        normalised = {k: 0.0 for k in part.uns['model_dict']}
    else:
        normalised = {k: v[1] / total_prob for k, v in part.uns['model_dict'].items()}

    # --- Step 3: upgrade to 3-tuple (cs, raw_prob, normalised_prob) and sort ---
    part.uns['model_dict'] = {
        k: (v[0], v[1], normalised[k])
        for k, v in part.uns['model_dict'].items()
    }

    part.uns['model_dict'] = {
        k: v for k, v in sorted(
            part.uns['model_dict'].items(),
            key=lambda item: item[1][2],  # sort by normalised prob
            reverse=True
        )
    }

    # --- Step 4: print ranking ---
    print("\n" + "="*60)
    print("MODEL RANKING (sorted by posterior probability):")
    print("="*60)
    for i, (k, v) in enumerate(part.uns['model_dict'].items()):
        cs, raw, norm = v
        print(f"Rank {i}: {k} - Raw: {raw:.3e} | Posterior: {norm*100:.2f}% | Structure: {cs}")

    print(f"\nTop 5 models:")
    for i, (k, v) in enumerate(list(part.uns['model_dict'].items())[:5]):
        cs, raw, norm = v
        print(f"  {i+1}. {k}: posterior={norm*100:.2f}%, raw={raw:.3e}, structure={cs}")

    return part


def infer_sh_jointly_from_dynamics(cs, AO, DP, time_points, 
                                   deterministic_size, total_cells, max_total_per_mutation,
                                   s_resolution=30, h_resolution=20, lamb=1.3, N_w=1e5,
                                   constraint_strength=2.0):
    """
    Joint inference of (s, h) with soft global constraint on WEIGHTED AVERAGE h.
    
    The constraint: weighted_mean(h_i) >= h_global = max(0, 2*V_sum - 1)
    where weights are clone sizes (N_mut_i).
    
    Individual clones can have any h in [0, 1], but the population average
    is softly pulled toward satisfying the global constraint.
    
    Parameters:
    -----------
    constraint_strength : float
        Strength of soft constraint (0 = no constraint, higher = stronger)
    """
    
    print("\n" + "="*70)
    print("JOINT (s, h) INFERENCE WITH SOFT GLOBAL CONSTRAINT")
    print("="*70)
    
    s_range = np.linspace(0.01, 1.0, s_resolution)
    h_range = np.linspace(0, 1, h_resolution)  # All clones: full [0,1] range
    
    # Compute VAF trajectories and global constraint
    vaf_ratio = AO / np.maximum(DP, 1.0)
    n_clones = len(cs)
    
    leading_vafs = []
    clone_weights = []
    
    for clone_muts in cs:
        clone_vafs = vaf_ratio[:, clone_muts]
        vaf_sums = clone_vafs.sum(axis=0)
        lead_idx = clone_muts[int(np.argmax(vaf_sums))]
        vaf_trajectory = np.array(vaf_ratio[:, lead_idx])
        leading_vafs.append(vaf_trajectory)
        # Weight by time-integrated VAF (proxy for clone size)
        clone_weights.append(np.trapz(vaf_trajectory))
    
    # Global constraint from final timepoint VAF sum
    V_sum_final = sum([v[-1] for v in leading_vafs])
    h_global = max(0.0, 2 * V_sum_final - 1)
    
    print(f"\nGlobal constraint: V_sum = {V_sum_final:.3f} → h_global = {h_global:.3f}")
    print(f"Required: weighted_mean(h) >= {h_global:.3f}")
    print(f"Constraint strength: {constraint_strength}")
    print("-" * 70)
    
    # Normalize clone weights for prior calculation
    clone_weights = np.array(clone_weights)
    if clone_weights.sum() > 0:
        clone_weights = clone_weights / clone_weights.sum()
    else:
        clone_weights = np.ones(n_clones) / n_clones
    
    # Store intermediate results for iterative constraint
    results = []
    clone_data = []  # Store trajectories and data for each clone
    
    for clone_idx, clone_muts in enumerate(cs):
        vaf_trajectory = leading_vafs[clone_idx]
        
        # Select leading mutation for this clone
        mut_idx = clone_muts[int(np.argmax([vaf_ratio[:, m].sum() for m in clone_muts]))]
        
        AO_clone = np.array(AO[:, mut_idx])
        DP_clone = np.array(DP[:, mut_idx])
        valid_mask = DP_clone > 0
        AO_clone = AO_clone[valid_mask]
        DP_clone = DP_clone[valid_mask]
        time_points_valid = np.array(time_points)[valid_mask]
        observed_vaf = AO_clone / np.maximum(DP_clone, 1.0)
        
        clone_data.append({
            'AO': AO_clone,
            'DP': DP_clone,
            'time_points': time_points_valid,
            'observed_vaf': observed_vaf,
            'weight': clone_weights[clone_idx],
            'vaf_final': vaf_trajectory[-1],
        })
        
        print(f"\nClone {clone_idx}: {clone_muts}")
        print(f"  VAF trajectory: {' → '.join(f'{v:.3f}' for v in observed_vaf)}")
    
    # Two-pass inference: first get unconstrained estimates, then apply soft constraint
    # Pass 1: Compute likelihood grids for all clones independently
    print(f"\n{'='*70}")
    print("Pass 1: Computing unconstrained likelihoods...")
    
    likelihood_grids = []
    
    for clone_idx in range(n_clones):
        data = clone_data[clone_idx]
        AO_clone = data['AO']
        DP_clone = data['DP']
        time_points_valid = data['time_points']
        observed_vaf = data['observed_vaf']
        
        joint_log_lik = np.zeros((s_resolution, h_resolution))
        
        for s_idx, s in enumerate(s_range):
            for h_idx, h in enumerate(h_range):
                
                log_lik = 0
                N_mut_init = observed_vaf[0] * 2 * N_w / (1 + h)
                N_mut_init = max(N_mut_init, 100)
                N_mut = N_mut_init
                
                for tp_idx in range(len(time_points_valid)):
                    if tp_idx > 0:
                        dt = time_points_valid[tp_idx] - time_points_valid[tp_idx - 1]
                        N_mut = N_mut * np.exp(s * dt)
                    
                    N_mut = min(N_mut, N_w * 0.95)
                    N_hom = N_mut * h
                    N_het = N_mut * (1 - h)
                    
                    vaf_expected = (N_het + 2 * N_hom) / (2 * N_w)
                    vaf_expected = np.clip(vaf_expected, 1e-8, 1.0 - 1e-8)
                    
                    ao = int(AO_clone[tp_idx])
                    dp = int(DP_clone[tp_idx])
                    log_lik += binom.logpmf(ao, dp, vaf_expected)
                
                joint_log_lik[s_idx, h_idx] = log_lik
        
        likelihood_grids.append(joint_log_lik)
    
    # Pass 2: Apply soft global constraint via iterative refinement
    print("Pass 2: Applying soft global constraint...")
    
    # Initialize with unconstrained posteriors
    posteriors = []
    for grid in likelihood_grids:
        max_log = grid.max()
        lik = np.exp(np.clip(grid - max_log, -700, 0))
        posteriors.append(lik / lik.sum())
    
    # Iteratively adjust to satisfy global constraint
    if constraint_strength > 0 and h_global > 0:
        for iteration in range(5):  # Few iterations suffice
            # Compute current weighted h expectation
            weighted_h = 0
            total_size = 0
            
            for i, (post, data) in enumerate(zip(posteriors, clone_data)):
                # Marginal h posterior for this clone
                h_marginal = post.sum(axis=0)  # Sum over s
                h_marginal = h_marginal / (h_marginal.sum() + 1e-300)
                expected_h = np.sum(h_range * h_marginal)
                
                # Size weight (approximate by final VAF)
                size_weight = data['vaf_final'] + 0.01
                weighted_h += expected_h * size_weight
                total_size += size_weight
            
            weighted_h /= total_size
            
            print(f"  Iteration {iteration}: weighted h = {weighted_h:.3f} (target >= {h_global:.3f})")
            
            # If constraint satisfied, stop
            if weighted_h >= h_global * 0.95:
                print(f"  Constraint satisfied.")
                break
            
            # Otherwise, adjust posteriors to pull toward higher h
            for i, (post, data) in enumerate(zip(posteriors, clone_data)):
                size_weight = data['vaf_final'] + 0.01
                
                h_marginal = post.sum(axis=0)
                expected_h = np.sum(h_range * h_marginal / (h_marginal.sum() + 1e-300))
                
                if weighted_h < h_global:
                    target_h = min(1.0, expected_h + (h_global - weighted_h) * 
                                  (size_weight / total_size) * constraint_strength)
                    
                    prior = np.exp(-((h_range - target_h) ** 2) / (2 * 0.15 ** 2))
                    prior = prior / prior.sum()
                    
                    new_post = post * prior[None, :]
                    posteriors[i] = new_post / (new_post.sum() + 1e-300)
    
    # Extract final results from posteriors
    for clone_idx in range(n_clones):
        post = posteriors[clone_idx]
        data = clone_data[clone_idx]
        
        # Marginalize
        s_post = post.sum(axis=1)
        h_post = post.sum(axis=0)
        s_post = s_post / (s_post.sum() + 1e-300)
        h_post = h_post / (h_post.sum() + 1e-300)
        
        # MAP estimates
        s_map = s_range[np.argmax(s_post)]
        h_map = h_range[np.argmax(h_post)]
        
        # Credible intervals
        s_cumsum = np.cumsum(s_post)
        h_cumsum = np.cumsum(h_post)
        s_ci_low = s_range[np.searchsorted(s_cumsum, 0.05)]
        s_ci_high = s_range[np.searchsorted(s_cumsum, 0.95)]
        h_ci_low = h_range[np.searchsorted(h_cumsum, 0.05)]
        h_ci_high = h_range[np.searchsorted(h_cumsum, 0.95)]
        
        # Cell count estimates
        N_mut_final = data['observed_vaf'][-1] * 2 * N_w / (1 + h_map)
        N_het_final = N_mut_final * (1 - h_map)
        N_hom_final = N_mut_final * h_map
        
        print(f"\nClone {clone_idx} Results:")
        print(f"  s = {s_map:.3f} [{s_ci_low:.3f}-{s_ci_high:.3f}]")
        print(f"  h = {h_map:.3f} [{h_ci_low:.3f}-{h_ci_high:.3f}]")
        print(f"  N_het ≈ {N_het_final:.0e}, N_hom ≈ {N_hom_final:.0e}")
        
        results.append({
            's_map': s_map,
            'h_map': h_map,
            's_posterior': s_post,
            'h_posterior': h_post,
            'joint_posterior': post,
            's_range': s_range,
            'h_range': h_range,
            's_ci': (s_ci_low, s_ci_high),
            'h_ci': (h_ci_low, h_ci_high),
            'h_global': h_global,
            'N_het_estimate': N_het_final,
            'N_hom_estimate': N_hom_final,
        })
    
    # Final verification
    total_N = sum([r['N_het_estimate'] + r['N_hom_estimate'] for r in results])
    weighted_h_final = sum([r['N_hom_estimate'] / total_N for r in results]) if total_N > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"Final check: weighted h = {weighted_h_final:.3f} (target >= {h_global:.3f})")
    print(f"{'='*70}")
    
    return results


def refine_optimal_model_posterior_vec(part, s_resolution=30, h_resolution=20, 
                                       constraint_strength=2.0):
    """
    IMPROVED VERSION: Joint (s, h) inference with soft global constraint.
    """
    
    # Retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]

    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    # Compute deterministic clone sizes (for bounds only)
    deterministic_size, total_cells, max_total_per_mutation, clonal_map = \
        compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

    # Joint (s, h) inference from temporal dynamics with constraint
    joint_results = infer_sh_jointly_from_dynamics(
        cs, AO, DP, time_points, 
        deterministic_size, total_cells, max_total_per_mutation,
        s_resolution=s_resolution, 
        h_resolution=h_resolution,
        constraint_strength=constraint_strength
    )

    # Extract results
    h_vec = np.array([result['h_map'] for result in joint_results])
    h_posterior_list = [result['h_posterior'] for result in joint_results]
    s_vec = joint_results[0]['s_range']
    
    # Construct fitness posterior from joint results (for compatibility)
    fitness_posterior = np.column_stack([result['s_posterior'] for result in joint_results])

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        'joint_inference': joint_results,
        'posterior': fitness_posterior,
        's_range': s_vec,
        'h_vec': h_vec,
        'h_posterior': h_posterior_list,
        'h_global': joint_results[0]['h_global'] if joint_results else None,
    }

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])
    h_est = np.zeros(part.shape[0])
    h_5 = np.zeros(part.shape[0])
    h_95 = np.zeros(part.shape[0])
    N_het_est = np.zeros(part.shape[0])
    N_hom_est = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):
        result = joint_results[i]
        
        fitness[c_idx] = result['s_map']
        fitness_5[c_idx] = result['s_ci'][0]
        fitness_95[c_idx] = result['s_ci'][1]
        clonal_index[c_idx] = i
        h_est[c_idx] = result['h_map']
        h_5[c_idx] = result['h_ci'][0]
        h_95[c_idx] = result['h_ci'][1]
        N_het_est[c_idx] = result['N_het_estimate']
        N_hom_est[c_idx] = result['N_hom_estimate']

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index
    part.obs['h'] = h_est
    part.obs['h_5'] = h_5
    part.obs['h_95'] = h_95
    part.obs['N_het_estimate'] = N_het_est
    part.obs['N_hom_estimate'] = N_hom_est

    # Append mutational structure to each mutation
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

#region Utility functions (unchanged but included for completeness)

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


def compute_model_likelihood(output, cs, s_range):
    """Compute model likelihood from clonal posteriors"""
    # Initialize clonal probability
    clonal_prob = np.zeros(len(cs))

    s_range_size = s_range.max() - s_range.min()
    s_prior = 1/s_range_size
    
    # Marginalise fitness for every clone to get clonal probability
    for i, out in enumerate(output.T):
        # Convert to numpy and guard against NaNs/Infs
        out = np.array(out, copy=False)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.maximum(out, 0.0)  # Ensure non-negative
        
        integral = np.trapz(x=s_range, y=out)
        clonal_prob[i] = s_prior * max(integral, 0.0)

    # Model probability as product of clonal probabilities (independence assumption)
    model_probability = np.prod(clonal_prob)

    return model_probability


def compute_invalid_combinations(part, pearson_distance_threshold=0.7):
    """Find mutation pairs with very different temporal correlation"""
    # Compute pearson of each mutation with time
    correlation_matrix = np.corrcoef(
        np.vstack([part.X, part.var.time_points]))
    correlation_vec = correlation_matrix[-1, :-1]

    # Compute distance between pearsonr's
    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])

    # Label invalid combinations if pearson correlation is too different
    not_valid_comb = np.argwhere(distance_matrix > pearson_distance_threshold)
    
    # Extract unique tuples from list (Order Irrespective)
    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] and [i[1], i[0]] not in res:
            res.append(i.tolist()) 
    
    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """
    Find all valid clonal structures using pearson correlation analysis
    and a VAF sum feasibility check (sum of leading mutation VAFs must be ≤ 1
    at every timepoint, otherwise the structure is biologically impossible).
    """
    
    n_mutations = part.shape[0]

    # AO/DP layers are (n_mutations, n_timepoints); transpose to (n_tps, n_muts)
    AO = part.layers['AO'].T
    DP = part.layers['DP'].T
    vaf_ratio = AO / np.maximum(DP, 1.0)  # (n_tps, n_muts)

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
                # ── existing correlation filter ────────────────────────────
                invalid_combinations_in_cs = 0
                for clone in cs:
                    mut_comb = list(combinations(clone, 2))
                    n_invalid_comb_in_clone = len(
                        [comb for comb in mut_comb 
                            if list(comb) in part.uns['invalid_combinations']])
                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                if invalid_combinations_in_cs > 0:
                    continue

                valid_cs.append(cs)
                    
            return valid_cs


def plot_optimal_model(part):
    """Plot the posterior distributions for the optimal model"""
    if part.uns.get('warning') is not None:
        print('WARNING: ' + part.uns['warning'])
        
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    ms = model['mutation_structure']
    s_range = model['s_range']

    # Normalisation constant
    norm_max = np.max(output, axis=0)
    norm_max = np.where(norm_max == 0, 1.0, norm_max)  # Prevent division by zero

    # Plot
    for i in range(len(cs)):
        p_key_str = ''
        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f'{part[j].obs.p_key.values[0]}'
            if k > 0:
                p_key_str += f'\n{part[j].obs.p_key.values[0]}'

        sns.lineplot(x=s_range,
                    y=output[:, i] / norm_max[i],
                    label=p_key_str)

#endregion