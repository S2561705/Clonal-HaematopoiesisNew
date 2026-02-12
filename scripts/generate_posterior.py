"""
Synthetic Data Generator for Oval-Shaped Joint Posteriors

Generates synthetic data that produces clear, informative joint posteriors P(s, h | data)
with oval shapes that reveal correlations between fitness and zygosity.

Key features:
- Moderate noise levels (not too clean, not too noisy)
- Strategic timepoint placement to capture both growth and saturation
- Parameter ranges that create identifiability challenges
- Mix of saturated and growing clones
"""

import numpy as np
import pandas as pd
import anndata as ad
import pickle as pk
from pathlib import Path

# ==============================================================================
# Configuration - OPTIMIZED FOR OVAL POSTERIORS
# ==============================================================================

# Output
OUTPUT_DIR = '../exports/synthetic_oval/'
OUTPUT_FILE = 'synthetic_oval_cohort.pk'
GROUND_TRUTH_FILE = 'synthetic_oval_ground_truth.csv'

# Cohort parameters
N_PARTICIPANTS = 20
N_TIMEPOINTS = 4  # Fixed number for consistency
TIME_SPAN = 16.0  # 12 years - enough for some saturation

# Population parameters
N_WILDTYPE = 1e5

# Mutation parameters - STRATEGIC RANGES for interesting posteriors
FITNESS_RANGE = (0.15, 0.8)  # Moderate fitness - not too fast, not too slow
INITIAL_SIZE_RANGE = (500, 2000)  # Moderate initial sizes

# Zygosity distribution - STRATEGIC for correlation
# We want cases where h and s are correlated in the posterior
ZYGOSITY_SCENARIOS = [
    # (h_range, s_range, description)
    ('pure_het', (0.0, 0.0), (0.2, 0.5), "Fast heterozygous"),
    ('pure_hom', (1.0, 1.0), (0.15, 0.4), "Slow homozygous"),
    ('mixed_low', (0.2, 0.4), (0.3, 0.6), "Low mixed, variable fitness"),
    ('mixed_high', (0.6, 0.8), (0.2, 0.4), "High mixed, slower fitness"),
]

# Sequencing parameters - MODERATE DEPTH for realistic noise
SEQUENCING_DEPTH_MEAN = 5000  # 5,000x - realistic
SEQUENCING_DEPTH_STD = 500
MIN_DEPTH = 2000

# Birth-death process
LAMBDA = 1.3

# Random seed
RANDOM_SEED = 42

# ==============================================================================
# Helper Functions
# ==============================================================================

def generate_strategic_timepoints(n_timepoints, time_span, strategy='mixed'):
    """
    Generate timepoints with strategic spacing for capturing dynamics.
    
    Parameters:
    -----------
    n_timepoints : int
        Number of timepoints
    time_span : float
        Total time span
    strategy : str
        'early': Dense early, sparse late (capture growth)
        'late': Sparse early, dense late (capture saturation)
        'mixed': Balanced (default)
    """
    
    if strategy == 'early':
        # More points early for growth phase
        t_norm = np.linspace(0, 1, n_timepoints) ** 1.5
    elif strategy == 'late':
        # More points late for saturation phase
        t_norm = np.linspace(0, 1, n_timepoints) ** 0.7
    else:  # mixed
        # Balanced spacing with slight early bias
        t_norm = np.linspace(0, 1, n_timepoints) ** 1.2
    
    return t_norm * time_span


def simulate_clone_trajectory_analytical(N0, s, h, time_points, N_w=1e5, lamb=1.3):
    """
    Simulate clone trajectory using deterministic growth + small noise.
    
    This produces smoother trajectories that are easier to infer from.
    
    Parameters:
    -----------
    N0 : float
        Initial clone size (total)
    s : float
        Fitness
    h : float
        Zygosity parameter (0 = het, 1 = hom)
    time_points : array
        Timepoints
    N_w : float
        Wildtype population
    lamb : float
        Birth rate
        
    Returns:
    --------
    N_het_traj : array
        Heterozygous cell counts
    N_hom_traj : array
        Homozygous cell counts
    vaf_traj : array
        VAF trajectory
    """
    
    n_tps = len(time_points)
    N_het_traj = np.zeros(n_tps)
    N_hom_traj = np.zeros(n_tps)
    vaf_traj = np.zeros(n_tps)
    
    # Split initial population
    N0_hom = N0 * h
    N0_het = N0 * (1 - h)
    
    for i, t in enumerate(time_points):
        # Deterministic exponential growth
        N_total = N0 * np.exp(s * t)
        
        # Add small process noise (10% CV)
        noise_factor = np.random.normal(1.0, 0.1)
        N_total = N_total * noise_factor
        
        # Cap at wildtype (competition)
        N_total = min(N_total, N_w * 0.9)
        
        # Split into het/hom (maintain ratio)
        N_hom_traj[i] = N_total * h
        N_het_traj[i] = N_total * (1 - h)
        
        # Compute VAF
        vaf_traj[i] = (N_het_traj[i] + 2 * N_hom_traj[i]) / (2 * N_w)
        vaf_traj[i] = np.clip(vaf_traj[i], 0, 1)
    
    return N_het_traj, N_hom_traj, vaf_traj


def sample_sequencing_realistic(vaf, depth):
    """
    Sample reads with realistic sequencing errors.
    
    Parameters:
    -----------
    vaf : float
        True VAF
    depth : int
        Sequencing depth
        
    Returns:
    --------
    ao : int
        Alternate allele count
    dp : int
        Total depth
    """
    
    # Actual depth varies
    dp = max(int(np.random.normal(depth, depth * 0.1)), MIN_DEPTH)
    
    # Add small sequencing error (0.5% error rate)
    vaf_observed = vaf * np.random.normal(1.0, 0.02)
    vaf_observed = np.clip(vaf_observed, 0, 1)
    
    # Binomial sampling
    ao = np.random.binomial(dp, vaf_observed)
    
    return ao, dp


# ==============================================================================
# Main Generation Function
# ==============================================================================

def generate_synthetic_participant_oval(participant_id, scenario_name=None):
    """
    Generate synthetic participant optimized for oval posteriors.
    
    Parameters:
    -----------
    participant_id : str
        Participant ID
    scenario_name : str, optional
        One of: 'pure_het', 'pure_hom', 'mixed_low', 'mixed_high'
        If None, randomly select
        
    Returns:
    --------
    part : AnnData
        Synthetic data
    ground_truth : dict
        Ground truth parameters
    """
    
    # Select scenario
    if scenario_name is None:
        scenario = np.random.choice(ZYGOSITY_SCENARIOS)
        scenario_name = scenario[0]
    else:
        scenario = [s for s in ZYGOSITY_SCENARIOS if s[0] == scenario_name][0]
    
    _, h_range, s_range, description = scenario
    
    # Sample parameters from scenario ranges
    if h_range[0] == h_range[1]:
        h_true = h_range[0]
    else:
        h_true = np.random.uniform(*h_range)
    
    s_true = np.random.uniform(*s_range)
    
    # Initial clone size
    N0_total = np.random.uniform(*INITIAL_SIZE_RANGE)
    
    # Determine timepoint strategy based on expected dynamics
    if s_true > 0.4:
        strategy = 'mixed'  # Fast growth needs both early and late
    else:
        strategy = 'early'  # Slow growth needs more early points
    
    time_points = generate_strategic_timepoints(N_TIMEPOINTS, TIME_SPAN, strategy)
    
    # Simulate trajectory
    N_het_traj, N_hom_traj, vaf_true_traj = simulate_clone_trajectory_analytical(
        N0_total, s_true, h_true, time_points
    )
    
    # Sample sequencing data with realistic noise
    AO = np.zeros(N_TIMEPOINTS, dtype=int)
    DP = np.zeros(N_TIMEPOINTS, dtype=int)
    
    for i in range(N_TIMEPOINTS):
        AO[i], DP[i] = sample_sequencing_realistic(
            vaf_true_traj[i],
            SEQUENCING_DEPTH_MEAN
        )
    
    # Create AnnData
    obs = pd.DataFrame({
        'mutation_id': ['MUT1'],
        'gene': ['GENE1'],
        'p_key': [f'p.{scenario_name}']
    })
    obs.index = obs['mutation_id']
    
    var = pd.DataFrame({
        'timepoint': np.arange(N_TIMEPOINTS),
        'time_points': time_points
    })
    var.index = [f'T{i}' for i in range(N_TIMEPOINTS)]
    
    # VAF matrix
    X = (AO / np.maximum(DP, 1)).reshape(1, -1)
    
    # Create AnnData
    part = ad.AnnData(X=X, obs=obs, var=var)
    part.layers['AO'] = AO.reshape(1, -1)
    part.layers['DP'] = DP.reshape(1, -1)
    
    part.uns['participant_id'] = participant_id
    part.uns['cohort'] = 'SYNTHETIC_OVAL'
    
    # Check saturation
    if len(vaf_true_traj) >= 3:
        late_growth = vaf_true_traj[-1] - vaf_true_traj[-3]
        is_saturated = late_growth < 0.03
    else:
        is_saturated = False
    
    # Theoretical max VAF
    theoretical_max_vaf = (1 - h_true) * 0.25 + h_true * 0.5
    
    # Compute expected posterior characteristics
    # High s + low h → strong correlation (fast growth, low saturation)
    # Low s + high h → weak correlation (slow growth, high saturation)
    correlation_strength = abs(s_true - 0.4) * abs(h_true - 0.5)
    
    # Ground truth
    ground_truth = {
        'participant_id': participant_id,
        'scenario': scenario_name,
        'description': description,
        'h_true': h_true,
        's_true': s_true,
        'N0_total': N0_total,
        'vaf_initial': vaf_true_traj[0],
        'vaf_final': vaf_true_traj[-1],
        'vaf_max': vaf_true_traj.max(),
        'theoretical_max_vaf': theoretical_max_vaf,
        'is_saturated': is_saturated,
        'mean_depth': DP.mean(),
        'expected_correlation_strength': correlation_strength,
        'timepoint_strategy': strategy,
        'clonal_structure': [[0]]
    }
    
    return part, ground_truth


def generate_cohort_oval():
    """
    Generate cohort with balanced scenarios.
    
    Returns:
    --------
    cohort : list of AnnData
    ground_truth_df : DataFrame
    """
    
    np.random.seed(RANDOM_SEED)
    
    print("="*80)
    print("SYNTHETIC DATA GENERATION - OVAL POSTERIOR OPTIMIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Participants: {N_PARTICIPANTS}")
    print(f"  Timepoints: {N_TIMEPOINTS} (strategic placement)")
    print(f"  Time span: {TIME_SPAN} years")
    print(f"  Fitness range: {FITNESS_RANGE}")
    print(f"  Sequencing depth: {SEQUENCING_DEPTH_MEAN}±{SEQUENCING_DEPTH_STD}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"\nScenarios:")
    for scenario_name, h_range, s_range, desc in ZYGOSITY_SCENARIOS:
        print(f"  {scenario_name}: h∈{h_range}, s∈{s_range} - {desc}")
    print()
    
    # Balanced scenario distribution
    n_per_scenario = N_PARTICIPANTS // len(ZYGOSITY_SCENARIOS)
    scenario_list = []
    for scenario_name, _, _, _ in ZYGOSITY_SCENARIOS:
        scenario_list.extend([scenario_name] * n_per_scenario)
    
    # Fill remaining with random
    while len(scenario_list) < N_PARTICIPANTS:
        scenario_list.append(np.random.choice([s[0] for s in ZYGOSITY_SCENARIOS]))
    
    np.random.shuffle(scenario_list)
    
    cohort = []
    ground_truths = []
    
    for i in range(N_PARTICIPANTS):
        participant_id = f'OVAL_{i+1:03d}'
        scenario_name = scenario_list[i]
        
        print(f"[{i+1}/{N_PARTICIPANTS}] Generating {participant_id} ({scenario_name})...", 
              end=' ')
        
        part, gt = generate_synthetic_participant_oval(participant_id, scenario_name)
        
        cohort.append(part)
        ground_truths.append(gt)
        
        sat_marker = "🔴SAT" if gt['is_saturated'] else "🟢GROW"
        corr_marker = "↗️CORR" if gt['expected_correlation_strength'] > 0.05 else "→INDEP"
        print(f"✓ h={gt['h_true']:.3f}, s={gt['s_true']:.3f}, "
              f"VAF:{gt['vaf_initial']:.3f}→{gt['vaf_final']:.3f} {sat_marker} {corr_marker}")
    
    ground_truth_df = pd.DataFrame(ground_truths)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(cohort)} participants")
    print(f"\nScenario distribution:")
    print(ground_truth_df['scenario'].value_counts())
    print(f"\nSaturation status:")
    print(f"  Saturated: {ground_truth_df['is_saturated'].sum()}")
    print(f"  Growing: {(~ground_truth_df['is_saturated']).sum()}")
    print(f"\nExpected posterior characteristics:")
    print(f"  High correlation: {(ground_truth_df['expected_correlation_strength'] > 0.05).sum()}")
    print(f"  Low correlation: {(ground_truth_df['expected_correlation_strength'] <= 0.05).sum()}")
    print(f"\nParameter ranges:")
    print(f"  h: [{ground_truth_df['h_true'].min():.3f}, {ground_truth_df['h_true'].max():.3f}]")
    print(f"  s: [{ground_truth_df['s_true'].min():.3f}, {ground_truth_df['s_true'].max():.3f}]")
    print(f"  Final VAF: [{ground_truth_df['vaf_final'].min():.3f}, {ground_truth_df['vaf_final'].max():.3f}]")
    
    return cohort, ground_truth_df


# ==============================================================================
# Save & Validate
# ==============================================================================

def save_synthetic_data(cohort, ground_truth_df):
    """Save synthetic data."""
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cohort
    cohort_file = output_path / OUTPUT_FILE
    with open(cohort_file, 'wb') as f:
        pk.dump(cohort, f)
    print(f"\n✅ Saved cohort to: {cohort_file}")
    
    # Save ground truth
    gt_file = output_path / GROUND_TRUTH_FILE
    ground_truth_df.to_csv(gt_file, index=False)
    print(f"✅ Saved ground truth to: {gt_file}")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Run inference:")
    print(f"   python run_inference_oval.py")
    print(f"\n2. Plot joint posteriors:")
    print(f"   python plot_joint_posteriors_oval.py")


def validate_synthetic_data(cohort, ground_truth_df):
    """Validate generated data."""
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    issues = []
    
    for part, gt in zip(cohort, ground_truth_df.to_dict('records')):
        pid = part.uns['participant_id']
        
        # Check VAF range
        AO = part.layers['AO'][0]
        DP = part.layers['DP'][0]
        VAF = AO / np.maximum(DP, 1)
        
        if np.any(VAF < 0) or np.any(VAF > 1):
            issues.append(f"{pid}: VAF out of range")
        
        if np.any(DP < MIN_DEPTH):
            issues.append(f"{pid}: Depth below minimum")
        
        # Check for signal
        if VAF.max() < 0.01:
            issues.append(f"{pid}: Very low signal (max VAF < 0.01)")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All validation checks passed!")
    
    # Statistics
    all_depths = np.concatenate([p.layers['DP'].flatten() for p in cohort])
    all_vafs = np.concatenate([
        p.layers['AO'].flatten() / np.maximum(p.layers['DP'].flatten(), 1) 
        for p in cohort
    ])
    
    print(f"\nData quality:")
    print(f"  Mean depth: {all_depths.mean():.0f}x")
    print(f"  VAF range: [{all_vafs.min():.4f}, {all_vafs.max():.4f}]")
    print(f"  Signal-to-noise (mean VAF / std VAF): {all_vafs.mean() / all_vafs.std():.2f}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Generate synthetic data optimized for oval posteriors."""
    
    cohort, ground_truth_df = generate_cohort_oval()
    validate_synthetic_data(cohort, ground_truth_df)
    save_synthetic_data(cohort, ground_truth_df)


if __name__ == '__main__':
    main()