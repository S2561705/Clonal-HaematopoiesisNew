"""
Generate Single Pooled Posterior Distribution

Combines data from multiple synthetic participants to create a single
joint posterior P(s, h | all_data) showing population-level inference.

This creates one beautiful oval-shaped posterior from all participants combined.
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.stats import binom

# ==============================================================================
# Configuration
# ==============================================================================

OUTPUT_DIR = Path('../exports/synthetic_pooled/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pooled population parameters
N_PARTICIPANTS = 30
N_TIMEPOINTS_PER_PARTICIPANT = 6
TIME_SPAN = 10.0  # years

# TRUE population parameters (what we want to recover)
S_TRUE_POPULATION = 0.35  # Population-level fitness
H_TRUE_POPULATION = 0.5   # Population-level zygosity

# Individual variation around population mean
S_VARIATION = 0.1   # ±10% variation in fitness
H_VARIATION = 0.15  # ±15% variation in zygosity

# Sequencing parameters
SEQUENCING_DEPTH_MEAN = 5000
SEQUENCING_DEPTH_STD = 500
MIN_DEPTH = 2000

# Population parameters
N_WILDTYPE = 1e5
LAMBDA = 1.3

# Inference grid
S_RESOLUTION = 100
H_RESOLUTION = 100
S_RANGE = (0.1, 0.8)
H_RANGE = (0.0, 1.0)

RANDOM_SEED = 42

# ==============================================================================
# Data Generation
# ==============================================================================

def generate_participant_data(participant_id, s_individual, h_individual, time_points):
    """
    Generate data for one participant with given parameters.
    
    Returns:
    --------
    AO : array (n_timepoints,)
        Alternate allele counts
    DP : array (n_timepoints,)
        Total depth
    """
    
    n_tps = len(time_points)
    
    # Initial clone size
    N0_total = np.random.uniform(500, 2000)
    
    # Simulate trajectory (deterministic with small noise)
    AO = np.zeros(n_tps, dtype=int)
    DP = np.zeros(n_tps, dtype=int)
    
    for i, t in enumerate(time_points):
        # Exponential growth
        N_total = N0_total * np.exp(s_individual * t)
        
        # Add process noise
        N_total *= np.random.normal(1.0, 0.1)
        
        # Cap at wildtype
        N_total = min(N_total, N_WILDTYPE * 0.9)
        
        # Split into het/hom
        N_hom = N_total * h_individual
        N_het = N_total * (1 - h_individual)
        
        # Compute VAF
        vaf = (N_het + 2 * N_hom) / (2 * N_WILDTYPE)
        vaf = np.clip(vaf, 0, 1)
        
        # Sample sequencing
        depth = max(int(np.random.normal(SEQUENCING_DEPTH_MEAN, SEQUENCING_DEPTH_STD)), MIN_DEPTH)
        
        # Add small sequencing error
        vaf_obs = vaf * np.random.normal(1.0, 0.02)
        vaf_obs = np.clip(vaf_obs, 0, 1)
        
        AO[i] = np.random.binomial(depth, vaf_obs)
        DP[i] = depth
    
    return AO, DP


def generate_pooled_cohort():
    """
    Generate cohort with participants drawn from population distribution.
    
    Returns:
    --------
    cohort_data : list of (participant_id, AO, DP, time_points, s_true, h_true)
    """
    
    np.random.seed(RANDOM_SEED)
    
    print("="*80)
    print("GENERATING POOLED SYNTHETIC COHORT")
    print("="*80)
    print(f"\nPopulation parameters:")
    print(f"  s_population = {S_TRUE_POPULATION} ± {S_VARIATION}")
    print(f"  h_population = {H_TRUE_POPULATION} ± {H_VARIATION}")
    print(f"\nCohort:")
    print(f"  Participants: {N_PARTICIPANTS}")
    print(f"  Timepoints per participant: {N_TIMEPOINTS_PER_PARTICIPANT}")
    print(f"  Time span: {TIME_SPAN} years")
    print()
    
    cohort_data = []
    
    # Shared timepoints for all participants
    time_points = np.linspace(0, TIME_SPAN, N_TIMEPOINTS_PER_PARTICIPANT)
    
    for i in range(N_PARTICIPANTS):
        participant_id = f'POOL_{i+1:03d}'
        
        # Sample individual parameters from population distribution
        s_individual = np.random.normal(S_TRUE_POPULATION, S_VARIATION)
        s_individual = np.clip(s_individual, 0.05, 1.0)
        
        h_individual = np.random.normal(H_TRUE_POPULATION, H_VARIATION)
        h_individual = np.clip(h_individual, 0.0, 1.0)
        
        # Generate data
        AO, DP = generate_participant_data(participant_id, s_individual, h_individual, time_points)
        
        cohort_data.append({
            'participant_id': participant_id,
            'AO': AO,
            'DP': DP,
            'time_points': time_points,
            's_individual': s_individual,
            'h_individual': h_individual
        })
        
        vaf_obs = AO / DP
        print(f"[{i+1}/{N_PARTICIPANTS}] {participant_id}: s={s_individual:.3f}, h={h_individual:.3f}, "
              f"VAF: {vaf_obs[0]:.3f}→{vaf_obs[-1]:.3f}")
    
    print(f"\n{'='*80}")
    print("COHORT STATISTICS")
    print("="*80)
    
    s_values = [p['s_individual'] for p in cohort_data]
    h_values = [p['h_individual'] for p in cohort_data]
    
    print(f"s: mean={np.mean(s_values):.3f}, std={np.std(s_values):.3f}, range=[{np.min(s_values):.3f}, {np.max(s_values):.3f}]")
    print(f"h: mean={np.mean(h_values):.3f}, std={np.std(h_values):.3f}, range=[{np.min(h_values):.3f}, {np.max(h_values):.3f}]")
    
    return cohort_data


# ==============================================================================
# Pooled Inference
# ==============================================================================

def compute_likelihood_single_observation(s, h, AO, DP, time_points, N_w=1e5):
    """
    Compute likelihood P(data | s, h) for one participant.
    
    Uses simplified forward model:
    - Exponential growth: N(t) = N0 * exp(s*t)
    - VAF = (N_het + 2*N_hom) / (2*N_w)
    - N_hom/N_het ratio determined by h
    
    Returns:
    --------
    log_likelihood : float
    """
    
    log_lik = 0.0
    
    # Estimate N0 from first observation
    vaf_0 = AO[0] / max(DP[0], 1)
    # Rough estimate: VAF ≈ (1-h + 2h) * N / (2*N_w) = (1+h) * N / (2*N_w)
    N0_est = (vaf_0 * 2 * N_w) / (1 + h + 1e-6)
    N0_est = max(N0_est, 10)  # Minimum size
    
    for i in range(len(time_points)):
        t = time_points[i]
        
        # Expected clone size
        N_expected = N0_est * np.exp(s * t)
        N_expected = min(N_expected, N_w * 0.9)  # Cap at wildtype
        
        # Split into het/hom
        N_hom = N_expected * h
        N_het = N_expected * (1 - h)
        
        # Expected VAF
        vaf_expected = (N_het + 2 * N_hom) / (2 * N_w)
        vaf_expected = np.clip(vaf_expected, 1e-6, 1.0 - 1e-6)
        
        # Binomial log likelihood
        log_lik += binom.logpmf(AO[i], DP[i], vaf_expected)
    
    return log_lik


def compute_pooled_posterior(cohort_data, s_range, h_range):
    """
    Compute pooled posterior P(s, h | all_data).
    
    Assumes participants are independent observations from same population.
    
    Returns:
    --------
    joint_posterior : array (n_s, n_h)
    s_range : array
    h_range : array
    """
    
    print(f"\n{'='*80}")
    print("COMPUTING POOLED POSTERIOR")
    print("="*80)
    print(f"Grid: {len(s_range)} × {len(h_range)} = {len(s_range) * len(h_range)} points")
    print()
    
    # Initialize log likelihood grid
    joint_log_likelihood = np.zeros((len(s_range), len(h_range)))
    
    # Compute likelihood for each (s, h) combination
    print("Computing likelihood grid...")
    for s_idx, s in enumerate(s_range):
        if s_idx % 10 == 0:
            print(f"  Progress: {s_idx}/{len(s_range)}")
        
        for h_idx, h in enumerate(h_range):
            log_lik_total = 0.0
            
            # Sum log likelihoods across all participants
            for participant in cohort_data:
                log_lik = compute_likelihood_single_observation(
                    s, h,
                    participant['AO'],
                    participant['DP'],
                    participant['time_points']
                )
                log_lik_total += log_lik
            
            joint_log_likelihood[s_idx, h_idx] = log_lik_total
    
    print("  Complete!")
    
    # Convert to probability (with numerical stability)
    max_log_lik = joint_log_likelihood.max()
    joint_log_likelihood_shifted = joint_log_likelihood - max_log_lik
    joint_likelihood = np.exp(np.clip(joint_log_likelihood_shifted, -700, 0))
    
    # Uniform prior
    prior = np.ones_like(joint_likelihood) / joint_likelihood.size
    
    # Posterior
    joint_posterior = joint_likelihood * prior
    
    # Normalize
    Z = joint_posterior.sum()
    if Z > 0 and np.isfinite(Z):
        joint_posterior = joint_posterior / Z
    else:
        print("⚠️  Warning: Posterior normalization issue")
        joint_posterior = prior
    
    print(f"\nPosterior statistics:")
    print(f"  Max posterior: {joint_posterior.max():.2e}")
    print(f"  Effective grid points: {(joint_posterior > joint_posterior.max() * 0.01).sum()}")
    
    return joint_posterior, s_range, h_range


# ==============================================================================
# Visualization
# ==============================================================================

def plot_pooled_posterior(joint_posterior, s_range, h_range, cohort_data, output_dir):
    """
    Contour-based visualization of pooled posterior.
    Emphasises geometry, correlation, and identifiability.
    """

    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATION (CONTOUR-BASED)")
    print("="*80)

    # -------------------------------------------------------------------------
    # Marginals
    # -------------------------------------------------------------------------

    s_marginal = joint_posterior.sum(axis=1)
    s_marginal /= s_marginal.sum() + 1e-300

    h_marginal = joint_posterior.sum(axis=0)
    h_marginal /= h_marginal.sum() + 1e-300

    # MAP
    s_map_idx, h_map_idx = np.unravel_index(joint_posterior.argmax(),
                                           joint_posterior.shape)
    s_map = s_range[s_map_idx]
    h_map = h_range[h_map_idx]

    # Marginal MAP
    s_map_marg = s_range[np.argmax(s_marginal)]
    h_map_marg = h_range[np.argmax(h_marginal)]

    # Credible intervals (90%)
    s_cumsum = np.cumsum(s_marginal)
    h_cumsum = np.cumsum(h_marginal)

    s_ci_low = s_range[np.searchsorted(s_cumsum, 0.05)]
    s_ci_high = s_range[np.searchsorted(s_cumsum, 0.95)]
    h_ci_low = h_range[np.searchsorted(h_cumsum, 0.05)]
    h_ci_high = h_range[np.searchsorted(h_cumsum, 0.95)]

    # Correlation
    S, H = np.meshgrid(s_range, h_range, indexing='ij')
    s_mean = np.sum(S * joint_posterior)
    h_mean = np.sum(H * joint_posterior)
    s_var = np.sum((S - s_mean)**2 * joint_posterior)
    h_var = np.sum((H - h_mean)**2 * joint_posterior)
    cov = np.sum((S - s_mean) * (H - h_mean) * joint_posterior)
    correlation = cov / (np.sqrt(s_var * h_var) + 1e-12)

    print(f"\nPosterior summary:")
    print(f"  s_MAP = {s_map_marg:.3f} [90% CI: {s_ci_low:.3f}, {s_ci_high:.3f}]")
    print(f"  h_MAP = {h_map_marg:.3f} [90% CI: {h_ci_low:.3f}, {h_ci_high:.3f}]")
    print(f"  Correlation: {correlation:.3f}")

    # -------------------------------------------------------------------------
    # Compute HPD contour levels
    # -------------------------------------------------------------------------

    P = joint_posterior / joint_posterior.sum()
    P_flat = np.sort(P.ravel())[::-1]
    cumsum = np.cumsum(P_flat)

    def hpd_level(mass):
        return P_flat[np.searchsorted(cumsum, mass)]

    level_68 = hpd_level(0.68)
    level_95 = hpd_level(0.95)

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------

    fig = plt.figure(figsize=(13, 12))
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1, 3, 0.15],
                  width_ratios=[3, 1, 0.15],
                  hspace=0.08, wspace=0.08)

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_h = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    ax_cbar = fig.add_subplot(gs[1, 2])

    # -------------------------------------------------------------------------
    # Joint posterior: FILLED CONTOURS
    # -------------------------------------------------------------------------

    cf = ax_joint.contourf(
        S, H, P,
        levels=50,
        cmap="viridis"
    )

    levels = np.array([level_95, level_68])
    levels = np.unique(levels)          # remove duplicates
    levels = np.sort(levels)             # must be increasing

    if len(levels) > 1:
        ax_joint.contour(
            S, H, P,
            levels=levels,
            colors=["white", "cyan"][:len(levels)],
            linewidths=[2.0, 3.0][:len(levels)]
        )
    else:
        print("⚠️  HPD contours collapsed (posterior effectively 1D)")

    # MAP + truth
    ax_joint.plot(s_map_marg, h_map_marg, "r*", markersize=28,
                  markeredgecolor="white", markeredgewidth=2,
                  label="MAP")

    ax_joint.plot(S_TRUE_POPULATION, H_TRUE_POPULATION, "g*",
                  markersize=28, markeredgecolor="white",
                  markeredgewidth=2, label="True")

    ax_joint.plot([S_TRUE_POPULATION, s_map_marg],
                  [H_TRUE_POPULATION, h_map_marg],
                  "w--", linewidth=2, alpha=0.6)

    ax_joint.set_xlabel("Population fitness (s)", fontsize=14, fontweight="bold")
    ax_joint.set_ylabel("Population zygosity (h)", fontsize=14, fontweight="bold")
    ax_joint.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax_joint.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # Marginals
    # -------------------------------------------------------------------------

    ax_s.plot(s_range, s_marginal, color="navy", linewidth=3)
    ax_s.fill_between(s_range, s_marginal, color="steelblue", alpha=0.6)
    ax_s.axvline(s_map_marg, color="red", linestyle="--")
    ax_s.axvline(S_TRUE_POPULATION, color="green", linestyle="--")
    ax_s.axvspan(s_ci_low, s_ci_high, color="red", alpha=0.2)
    ax_s.set_ylabel("P(s)", fontsize=12)
    ax_s.tick_params(labelbottom=False)
    ax_s.set_ylim(bottom=0)

    ax_h.plot(h_marginal, h_range, color="darkred", linewidth=3)
    ax_h.fill_betweenx(h_range, h_marginal, color="coral", alpha=0.6)
    ax_h.axhline(h_map_marg, color="red", linestyle="--")
    ax_h.axhline(H_TRUE_POPULATION, color="green", linestyle="--")
    ax_h.axhspan(h_ci_low, h_ci_high, color="red", alpha=0.2)
    ax_h.set_xlabel("P(h)", fontsize=12)
    ax_h.tick_params(labelleft=False)
    ax_h.set_xlim(left=0)

    # -------------------------------------------------------------------------
    # Colorbar + title
    # -------------------------------------------------------------------------

    cbar = plt.colorbar(cf, cax=ax_cbar)
    cbar.set_label("Posterior density", fontsize=11)

    title = (
        f"Pooled posterior P(s, h | all data)\n"
        f"N={len(cohort_data)}, "
        f"{N_TIMEPOINTS_PER_PARTICIPANT} timepoints each\n"
        f"Correlation ρ = {correlation:.3f}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    output_file = output_dir / "pooled_posterior.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Saved: {output_file}")

    return s_map_marg, h_map_marg



# ==============================================================================
# Main
# ==============================================================================

def main():
    """Generate pooled synthetic data and compute single posterior."""
    
    # Generate cohort
    cohort_data = generate_pooled_cohort()
    
    # Save cohort data
    cohort_df = pd.DataFrame([{
        'participant_id': p['participant_id'],
        's_individual': p['s_individual'],
        'h_individual': p['h_individual']
    } for p in cohort_data])
    cohort_df.to_csv(OUTPUT_DIR / 'cohort_parameters.csv', index=False)
    
    # Define grid
    s_range = np.linspace(*S_RANGE, S_RESOLUTION)
    h_range = np.linspace(*H_RANGE, H_RESOLUTION)
    
    # Compute pooled posterior
    joint_posterior, s_range, h_range = compute_pooled_posterior(
        cohort_data, s_range, h_range
    )
    
    # Visualize
    s_map, h_map = plot_pooled_posterior(
        joint_posterior, s_range, h_range, cohort_data, OUTPUT_DIR
    )
    
    # Save posterior
    np.savez(
        OUTPUT_DIR / 'pooled_posterior.npz',
        joint_posterior=joint_posterior,
        s_range=s_range,
        h_range=h_range,
        s_map=s_map,
        h_map=h_map,
        s_true=S_TRUE_POPULATION,
        h_true=H_TRUE_POPULATION
    )
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - Visualization: pooled_posterior.png")
    print(f"  - Data: pooled_posterior.npz")
    print(f"  - Parameters: cohort_parameters.csv")


if __name__ == '__main__':
    main()