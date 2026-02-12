"""
Synthetic Data Generator for Clonal Evolution Inference Testing
IMPROVED VERSION: Longer follow-up and more timepoints to test saturation

Generates synthetic participants with known ground truth:
- One mutation per patient
- Variable zygosity (heterozygous, homozygous, or mixed)
- Known fitness values
- Long follow-up to reach VAF saturation
- High sequencing depth for clear signal
- Realistic sequencing noise (binomial sampling)

Ground truth is saved alongside synthetic data for validation.
"""

import numpy as np
import pandas as pd
import anndata as ad
import pickle as pk
from pathlib import Path

# ==============================================================================
# Configuration - IMPROVED FOR SATURATION TESTING
# ==============================================================================

# Output
OUTPUT_DIR = '../exports/synthetic/'
OUTPUT_FILE = 'synthetic_single_mutation_cohort.pk'
GROUND_TRUTH_FILE = 'synthetic_ground_truth.csv'

# Cohort parameters
N_PARTICIPANTS = 30  # Total number of synthetic participants
N_TIMEPOINTS_RANGE = (6, 10)  # MORE timepoints to capture saturation dynamics
TIME_SPAN_RANGE = (8.0, 20.0)  # LONGER follow-up to reach saturation

# Population parameters
N_WILDTYPE = 1e5  # Wild-type cell population size

# Mutation parameters - one per participant
FITNESS_RANGE = (0.3, 0.9)  # Moderate to high fitness for saturation
INITIAL_SIZE_RANGE = (1000, 5000)  # Larger initial sizes for better visibility

# Zygosity distribution (probabilities)
ZYGOSITY_PROBS = {
    'heterozygous': 0.35,   # h = 0 (purely het) - saturates at ~0.25
    'homozygous': 0.35,     # h = 1 (purely hom) - saturates at ~0.50
    'mixed': 0.30           # h = random in (0.3, 0.7) - saturates in between
}

# Sequencing parameters - VERY HIGH DEPTH for clear signal
SEQUENCING_DEPTH_MEAN = 10000  # 100,000x depth  
SEQUENCING_DEPTH_STD = 1000
MIN_DEPTH = 5000  # Minimum 50,000x for low noise

# Birth-death process
LAMBDA = 1.3  # Birth rate

# Random seed
RANDOM_SEED = 42

# ==============================================================================
# Helper Functions
# ==============================================================================

def simulate_BD_process(N0, s, time, lamb=LAMBDA, n_steps=1000):
    """
    Simulate birth-death process using Euler-Maruyama approximation.
    
    Parameters:
    -----------
    N0 : float
        Initial population size
    s : float
        Fitness (selection coefficient)
    time : float
        Time duration (years)
    lamb : float
        Birth rate
    n_steps : int
        Number of simulation steps
        
    Returns:
    --------
    N_final : float
        Final population size
    """
    dt = time / n_steps
    N = N0
    
    for _ in range(n_steps):
        # Deterministic drift
        mean_change = N * s * dt
        
        # Stochastic diffusion
        variance = N * (2 * lamb + s) * dt
        if variance > 0:
            noise = np.random.normal(0, np.sqrt(variance))
        else:
            noise = 0
        
        # Update
        N = N + mean_change + noise
        
        # Ensure non-negative
        N = max(N, 0)
        
        # Extinction check
        if N < 1:
            return 0.0
    
    return N


def compute_vaf(N_het, N_hom, N_wildtype):
    """
    Compute variant allele frequency.
    
    VAF = (N_het + 2*N_hom) / (2 * (N_wildtype + N_het + N_hom))
    
    Parameters:
    -----------
    N_het : float
        Number of heterozygous mutant cells
    N_hom : float
        Number of homozygous mutant cells
    N_wildtype : float
        Number of wildtype cells
        
    Returns:
    --------
    vaf : float
        Variant allele frequency [0, 1]
    """
    N_total = N_wildtype + N_het + N_hom
    if N_total == 0:
        return 0.0
    
    vaf = (N_het + 2 * N_hom) / (2 * N_total)
    return np.clip(vaf, 0, 1)


def sample_sequencing(vaf, depth):
    """
    Sample alternate allele counts from binomial distribution.
    
    Parameters:
    -----------
    vaf : float
        True variant allele frequency
    depth : int
        Sequencing depth
        
    Returns:
    --------
    ao : int
        Alternate allele count
    dp : int
        Total depth
    """
    dp = max(int(depth), MIN_DEPTH)
    ao = np.random.binomial(dp, vaf)
    return ao, dp


def generate_timepoints(n_timepoints, time_span):
    """
    Generate timepoints with more density early on, but still covering late saturation.
    
    Parameters:
    -----------
    n_timepoints : int
        Number of timepoints
    time_span : float
        Total time span (years)
        
    Returns:
    --------
    timepoints : array
        Timepoints in years
    """
    # Use mixed spacing: early dense, late sparse
    # First half: square root spacing (more early points)
    # Second half: linear spacing
    mid_point = n_timepoints // 2
    
    early_times = np.linspace(0, np.sqrt(time_span/2), mid_point) ** 2
    late_times = np.linspace(time_span/2, time_span, n_timepoints - mid_point + 1)[1:]
    
    timepoints = np.concatenate([early_times, late_times])
    return timepoints[:n_timepoints]


# ==============================================================================
# Main Synthetic Data Generation
# ==============================================================================

def generate_synthetic_participant(participant_id, zygosity_type=None):
    """
    Generate a single synthetic participant with one mutation.
    
    Parameters:
    -----------
    participant_id : str
        Participant identifier
    zygosity_type : str, optional
        'heterozygous', 'homozygous', or 'mixed'
        If None, randomly sample based on ZYGOSITY_PROBS
        
    Returns:
    --------
    part : AnnData
        Synthetic participant data
    ground_truth : dict
        Ground truth parameters
    """
    
    # Sample zygosity type if not specified
    if zygosity_type is None:
        zygosity_type = np.random.choice(
            list(ZYGOSITY_PROBS.keys()),
            p=list(ZYGOSITY_PROBS.values())
        )
    
    # Sample zygosity parameter h
    if zygosity_type == 'heterozygous':
        h_true = 0.0
    elif zygosity_type == 'homozygous':
        h_true = 1.0
    else:  # mixed
        h_true = np.random.uniform(0.3, 0.7)  # Tighter range for clearer signal
    
    # Sample fitness
    s_true = np.random.uniform(*FITNESS_RANGE)
    
    # Sample initial clone size
    N0_total = np.random.uniform(*INITIAL_SIZE_RANGE)
    
    # Split into het and hom based on h
    N0_hom = N0_total * h_true
    N0_het = N0_total * (1 - h_true)
    
    # Sample timepoints
    n_timepoints = np.random.randint(*N_TIMEPOINTS_RANGE)
    time_span = np.random.uniform(*TIME_SPAN_RANGE)
    timepoints = generate_timepoints(n_timepoints, time_span)
    
    # Simulate clone evolution
    N_het_trajectory = np.zeros(n_timepoints)
    N_hom_trajectory = np.zeros(n_timepoints)
    vaf_true_trajectory = np.zeros(n_timepoints)
    
    for i, t in enumerate(timepoints):
        if i == 0:
            # Initial timepoint
            N_het_trajectory[i] = N0_het
            N_hom_trajectory[i] = N0_hom
        else:
            # Simulate growth from previous timepoint
            delta_t = timepoints[i] - timepoints[i-1]
            
            N_het_trajectory[i] = simulate_BD_process(
                N_het_trajectory[i-1], s_true, delta_t
            )
            N_hom_trajectory[i] = simulate_BD_process(
                N_hom_trajectory[i-1], s_true, delta_t
            )
        
        # Compute true VAF
        vaf_true_trajectory[i] = compute_vaf(
            N_het_trajectory[i], 
            N_hom_trajectory[i],
            N_WILDTYPE
        )
    
    # Sample sequencing data
    AO = np.zeros(n_timepoints, dtype=int)
    DP = np.zeros(n_timepoints, dtype=int)
    
    for i in range(n_timepoints):
        # Sample sequencing depth
        depth = np.random.normal(
            SEQUENCING_DEPTH_MEAN, 
            SEQUENCING_DEPTH_STD
        )
        
        # Sample reads
        AO[i], DP[i] = sample_sequencing(vaf_true_trajectory[i], depth)
    
    # Create AnnData object
    # Structure: mutations (obs) x timepoints (var)
    obs = pd.DataFrame({
        'mutation_id': ['MUT1'],
        'gene': ['GENE1'],
        'p_key': ['p.Arg100Cys']
    })
    obs.index = obs['mutation_id']
    
    var = pd.DataFrame({
        'timepoint': np.arange(n_timepoints),
        'time_points': timepoints
    })
    var.index = [f'T{i}' for i in range(n_timepoints)]
    
    # Main data matrix (VAF for compatibility, though not used directly)
    X = (AO / np.maximum(DP, 1)).reshape(1, -1)
    
    # Create AnnData
    part = ad.AnnData(
        X=X,
        obs=obs,
        var=var
    )
    
    # Add layers
    part.layers['AO'] = AO.reshape(1, -1)
    part.layers['DP'] = DP.reshape(1, -1)
    
    # Add metadata
    part.uns['participant_id'] = participant_id
    part.uns['cohort'] = 'SYNTHETIC'
    
    # Check if saturated (VAF plateaued)
    if len(vaf_true_trajectory) >= 3:
        late_growth = vaf_true_trajectory[-1] - vaf_true_trajectory[-3]
        is_saturated = late_growth < 0.05
    else:
        is_saturated = False
    
    # Theoretical saturation point
    theoretical_max_vaf = (1 - h_true) * 0.25 + h_true * 0.5  # Weighted by h
    
    # Ground truth
    ground_truth = {
        'participant_id': participant_id,
        'n_timepoints': n_timepoints,
        'time_span': time_span,
        'zygosity_type': zygosity_type,
        'h_true': h_true,
        's_true': s_true,
        'N0_total': N0_total,
        'N0_het': N0_het,
        'N0_hom': N0_hom,
        'N_het_final': N_het_trajectory[-1],
        'N_hom_final': N_hom_trajectory[-1],
        'vaf_initial': vaf_true_trajectory[0],
        'vaf_final': vaf_true_trajectory[-1],
        'vaf_max': vaf_true_trajectory.max(),
        'theoretical_max_vaf': theoretical_max_vaf,
        'is_saturated': is_saturated,
        'mean_depth': DP.mean(),
        'clonal_structure': [[0]]  # Single mutation
    }
    
    return part, ground_truth


def generate_synthetic_cohort():
    """
    Generate full synthetic cohort with balanced zygosity types.
    
    Returns:
    --------
    cohort : list of AnnData
        Synthetic participant data
    ground_truth_df : DataFrame
        Ground truth parameters for all participants
    """
    
    np.random.seed(RANDOM_SEED)
    
    print("="*80)
    print("SYNTHETIC DATA GENERATION - SATURATION TESTING VERSION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Participants: {N_PARTICIPANTS}")
    print(f"  Timepoints per participant: {N_TIMEPOINTS_RANGE}")
    print(f"  Time span: {TIME_SPAN_RANGE} years (LONG for saturation)")
    print(f"  Fitness range: {FITNESS_RANGE}")
    print(f"  Sequencing depth: {SEQUENCING_DEPTH_MEAN}±{SEQUENCING_DEPTH_STD} (HIGH)")
    print(f"  Zygosity distribution: {ZYGOSITY_PROBS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()
    
    # Determine zygosity types to ensure balanced distribution
    n_het = int(N_PARTICIPANTS * ZYGOSITY_PROBS['heterozygous'])
    n_hom = int(N_PARTICIPANTS * ZYGOSITY_PROBS['homozygous'])
    n_mix = N_PARTICIPANTS - n_het - n_hom
    
    zygosity_types = (
        ['heterozygous'] * n_het +
        ['homozygous'] * n_hom +
        ['mixed'] * n_mix
    )
    np.random.shuffle(zygosity_types)
    
    print(f"Generating participants:")
    print(f"  Heterozygous: {n_het} (expected saturation ~0.25)")
    print(f"  Homozygous: {n_hom} (expected saturation ~0.50)")
    print(f"  Mixed: {n_mix} (expected saturation in between)")
    print()
    
    cohort = []
    ground_truths = []
    
    for i in range(N_PARTICIPANTS):
        participant_id = f'SYN_{i+1:03d}'
        zygosity_type = zygosity_types[i]
        
        print(f"[{i+1}/{N_PARTICIPANTS}] Generating {participant_id} ({zygosity_type})...", 
              end=' ')
        
        part, gt = generate_synthetic_participant(participant_id, zygosity_type)
        
        cohort.append(part)
        ground_truths.append(gt)
        
        sat_marker = "🔴 SAT" if gt['is_saturated'] else "🟢 GROW"
        print(f"✓ (h={gt['h_true']:.3f}, s={gt['s_true']:.3f}, "
              f"VAF: {gt['vaf_initial']:.3f}→{gt['vaf_final']:.3f}, {sat_marker})")
    
    ground_truth_df = pd.DataFrame(ground_truths)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(cohort)} participants")
    print(f"\nZygosity distribution:")
    print(ground_truth_df['zygosity_type'].value_counts())
    print(f"\nSaturation status:")
    print(f"  Saturated: {ground_truth_df['is_saturated'].sum()}")
    print(f"  Still growing: {(~ground_truth_df['is_saturated']).sum()}")
    print(f"\nFitness statistics:")
    print(ground_truth_df['s_true'].describe())
    print(f"\nZygosity parameter (h) statistics:")
    print(ground_truth_df['h_true'].describe())
    print(f"\nFinal VAF statistics:")
    print(ground_truth_df['vaf_final'].describe())
    
    return cohort, ground_truth_df


# ==============================================================================
# Save Functions
# ==============================================================================

def save_synthetic_data(cohort, ground_truth_df):
    """
    Save synthetic cohort and ground truth.
    
    Parameters:
    -----------
    cohort : list of AnnData
        Synthetic participant data
    ground_truth_df : DataFrame
        Ground truth parameters
    """
    
    # Create output directory
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
    print("COMPLETE")
    print("="*80)
    print(f"\nTo test inference, run:")
    print(f"  1. Load: cohort = pickle.load(open('{cohort_file}', 'rb'))")
    print(f"  2. Run inference on cohort")
    print(f"  3. Compare results to: ground_truth = pd.read_csv('{gt_file}')")


# ==============================================================================
# Validation Function
# ==============================================================================

def validate_synthetic_data(cohort, ground_truth_df):
    """
    Perform basic validation checks on synthetic data.
    
    Parameters:
    -----------
    cohort : list of AnnData
        Synthetic participant data
    ground_truth_df : DataFrame
        Ground truth parameters
    """
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    issues = []
    
    # Check cohort size
    if len(cohort) != len(ground_truth_df):
        issues.append(f"Cohort size mismatch: {len(cohort)} vs {len(ground_truth_df)}")
    
    # Check each participant
    for i, (part, gt) in enumerate(zip(cohort, ground_truth_df.to_dict('records'))):
        pid = part.uns['participant_id']
        
        # Check structure
        if part.shape[0] != 1:
            issues.append(f"{pid}: Expected 1 mutation, got {part.shape[0]}")
        
        # Check timepoints
        if part.shape[1] != gt['n_timepoints']:
            issues.append(f"{pid}: Timepoint mismatch")
        
        # Check VAF range
        AO = part.layers['AO'][0]
        DP = part.layers['DP'][0]
        VAF = AO / np.maximum(DP, 1)
        
        if np.any(VAF < 0) or np.any(VAF > 1):
            issues.append(f"{pid}: VAF out of range [0,1]")
        
        # Check depth
        if np.any(DP < MIN_DEPTH):
            issues.append(f"{pid}: Some depths below minimum")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All validation checks passed!")
    
    # Summary statistics
    print(f"\nData statistics:")
    print(f"  Participants: {len(cohort)}")
    print(f"  Total timepoints: {sum(p.shape[1] for p in cohort)}")
    print(f"  Avg timepoints per participant: {np.mean([p.shape[1] for p in cohort]):.1f}")
    
    all_depths = np.concatenate([p.layers['DP'].flatten() for p in cohort])
    print(f"  Sequencing depth: {all_depths.mean():.0f} ± {all_depths.std():.0f}")
    
    all_vafs = np.concatenate([
        p.layers['AO'].flatten() / np.maximum(p.layers['DP'].flatten(), 1) 
        for p in cohort
    ])
    print(f"  VAF range: [{all_vafs.min():.3f}, {all_vafs.max():.3f}]")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Generate and save synthetic cohort."""
    
    # Generate synthetic data
    cohort, ground_truth_df = generate_synthetic_cohort()
    
    # Validate
    validate_synthetic_data(cohort, ground_truth_df)
    
    # Save
    save_synthetic_data(cohort, ground_truth_df)


if __name__ == '__main__':
    main()