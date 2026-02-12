"""
Synthetic Data Generator for Clonal Evolution Inference Testing
MULTI-CLONE VERSION: Tests clonal structure inference with known ground truth

Generates synthetic participants with:
- 1-3 independent clones
- 1-2 mutations per clone (co-occurring)
- Total 1-6 mutations per participant
- Variable zygosity per clone
- Known fitness values per clone
- Long follow-up to reach VAF saturation
- High sequencing depth for clear signal

Ground truth includes:
- True clonal structure (which mutations belong to which clone)
- True fitness per clone
- True zygosity per clone
"""

import numpy as np
import pandas as pd
import anndata as ad
import pickle as pk
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

# Output
OUTPUT_DIR = '../exports/synthetic_multiclone/'
OUTPUT_FILE = 'synthetic_multiclone_cohort.pk'
GROUND_TRUTH_FILE = 'synthetic_multiclone_ground_truth.csv'

# Cohort parameters
N_PARTICIPANTS = 30
N_TIMEPOINTS_RANGE = (6, 10)
TIME_SPAN_RANGE = (8.0, 20.0)

# Population parameters
N_WILDTYPE = 1e5

# CLONAL STRUCTURE PARAMETERS
N_CLONES_RANGE = (1, 3)  # Number of independent clones per participant
N_MUTATIONS_PER_CLONE_RANGE = (1, 2)  # Mutations per clone

# Clone parameters
FITNESS_RANGE = (0.3, 0.9)
INITIAL_SIZE_RANGE = (1000, 5000)

# Fitness correlation within clones
# Mutations in same clone have nearly identical fitness (same cells)
FITNESS_WITHIN_CLONE_STD = 0.02  # Small variation due to measurement noise

# Zygosity distribution
ZYGOSITY_PROBS = {
    'heterozygous': 0.4,
    'homozygous': 0.3,
    'mixed': 0.3
}

# Sequencing parameters
SEQUENCING_DEPTH_MEAN = 10000
SEQUENCING_DEPTH_STD = 1000
MIN_DEPTH = 5000

# Birth-death process
LAMBDA = 1.3

# Random seed
RANDOM_SEED = 42

# ==============================================================================
# Helper Functions
# ==============================================================================

def simulate_BD_process(N0, s, time, lamb=LAMBDA, n_steps=1000):
    """Simulate birth-death process."""
    dt = time / n_steps
    N = N0
    
    for _ in range(n_steps):
        mean_change = N * s * dt
        variance = N * (2 * lamb + s) * dt
        if variance > 0:
            noise = np.random.normal(0, np.sqrt(variance))
        else:
            noise = 0
        
        N = N + mean_change + noise
        N = max(N, 0)
        
        if N < 1:
            return 0.0
    
    return N


def compute_vaf(N_het, N_hom, N_wildtype):
    """Compute VAF."""
    N_total = N_wildtype + N_het + N_hom
    if N_total == 0:
        return 0.0
    
    vaf = (N_het + 2 * N_hom) / (2 * N_total)
    return np.clip(vaf, 0, 1)


def sample_sequencing(vaf, depth):
    """Sample reads."""
    dp = max(int(depth), MIN_DEPTH)
    ao = np.random.binomial(dp, vaf)
    return ao, dp


def generate_timepoints(n_timepoints, time_span):
    """Generate timepoints with early density."""
    mid_point = n_timepoints // 2
    early_times = np.linspace(0, np.sqrt(time_span/2), mid_point) ** 2
    late_times = np.linspace(time_span/2, time_span, n_timepoints - mid_point + 1)[1:]
    timepoints = np.concatenate([early_times, late_times])
    return timepoints[:n_timepoints]


def generate_clonal_structure(n_clones, n_mutations_per_clone_range):
    """
    Generate a clonal structure (partition of mutations into clones).
    
    Parameters:
    -----------
    n_clones : int
        Number of clones
    n_mutations_per_clone_range : tuple
        (min, max) mutations per clone
        
    Returns:
    --------
    clonal_structure : list of lists
        Each sublist contains mutation indices for that clone
        Example: [[0, 1], [2], [3, 4, 5]] means:
        - Clone 0: mutations 0 and 1
        - Clone 1: mutation 2
        - Clone 2: mutations 3, 4, and 5
    """
    clonal_structure = []
    mutation_idx = 0
    
    for clone_idx in range(n_clones):
        # Sample number of mutations for this clone
        n_muts = np.random.randint(*n_mutations_per_clone_range)
        
        # Assign mutations to this clone
        clone_mutations = list(range(mutation_idx, mutation_idx + n_muts))
        clonal_structure.append(clone_mutations)
        
        mutation_idx += n_muts
    
    return clonal_structure


# ==============================================================================
# Main Synthetic Data Generation
# ==============================================================================

def generate_synthetic_participant_multiclone(participant_id):
    """
    Generate a synthetic participant with multiple clones.
    
    Returns:
    --------
    part : AnnData
        Synthetic participant data
        Shape: (n_mutations, n_timepoints)
    ground_truth : dict
        Includes clonal structure, fitness per clone, zygosity per clone
    """
    
    # Sample number of clones
    n_clones = np.random.randint(*N_CLONES_RANGE)
    
    # Generate clonal structure
    clonal_structure = generate_clonal_structure(n_clones, N_MUTATIONS_PER_CLONE_RANGE)
    n_mutations = sum(len(clone) for clone in clonal_structure)
    
    # Sample parameters for each clone
    clone_params = []
    for clone_idx in range(n_clones):
        # Sample zygosity type
        zyg_type = np.random.choice(
            list(ZYGOSITY_PROBS.keys()),
            p=list(ZYGOSITY_PROBS.values())
        )
        
        if zyg_type == 'heterozygous':
            h = 0.0
        elif zyg_type == 'homozygous':
            h = 1.0
        else:
            h = np.random.uniform(0.3, 0.7)
        
        # Sample fitness for clone
        s_clone = np.random.uniform(*FITNESS_RANGE)
        
        # Sample initial size
        N0_total = np.random.uniform(*INITIAL_SIZE_RANGE)
        N0_het = N0_total * (1 - h)
        N0_hom = N0_total * h
        
        clone_params.append({
            'zygosity_type': zyg_type,
            'h': h,
            's': s_clone,
            'N0_total': N0_total,
            'N0_het': N0_het,
            'N0_hom': N0_hom
        })
    
    # Sample timepoints
    n_timepoints = np.random.randint(*N_TIMEPOINTS_RANGE)
    time_span = np.random.uniform(*TIME_SPAN_RANGE)
    timepoints = generate_timepoints(n_timepoints, time_span)
    
    # Simulate evolution of each clone
    clone_trajectories = []
    
    for clone_idx in range(n_clones):
        params = clone_params[clone_idx]
        
        N_het_traj = np.zeros(n_timepoints)
        N_hom_traj = np.zeros(n_timepoints)
        
        for tp_idx, t in enumerate(timepoints):
            if tp_idx == 0:
                N_het_traj[tp_idx] = params['N0_het']
                N_hom_traj[tp_idx] = params['N0_hom']
            else:
                delta_t = timepoints[tp_idx] - timepoints[tp_idx-1]
                N_het_traj[tp_idx] = simulate_BD_process(
                    N_het_traj[tp_idx-1], params['s'], delta_t
                )
                N_hom_traj[tp_idx] = simulate_BD_process(
                    N_hom_traj[tp_idx-1], params['s'], delta_t
                )
        
        clone_trajectories.append({
            'N_het': N_het_traj,
            'N_hom': N_hom_traj
        })
    
    # Generate mutation data
    # Mutations in same clone have same trajectory (with small noise)
    AO_matrix = np.zeros((n_mutations, n_timepoints), dtype=int)
    DP_matrix = np.zeros((n_mutations, n_timepoints), dtype=int)
    VAF_true_matrix = np.zeros((n_mutations, n_timepoints))
    
    mutation_to_clone = {}  # Map mutation index to clone index
    
    for clone_idx, clone_muts in enumerate(clonal_structure):
        traj = clone_trajectories[clone_idx]
        params = clone_params[clone_idx]
        
        for mut_idx in clone_muts:
            mutation_to_clone[mut_idx] = clone_idx
            
            # Mutations in same clone have nearly identical VAF
            # (small independent sampling noise only)
            for tp_idx in range(n_timepoints):
                vaf_true = compute_vaf(
                    traj['N_het'][tp_idx],
                    traj['N_hom'][tp_idx],
                    N_WILDTYPE
                )
                
                VAF_true_matrix[mut_idx, tp_idx] = vaf_true
                
                # Sample sequencing depth
                depth = np.random.normal(SEQUENCING_DEPTH_MEAN, SEQUENCING_DEPTH_STD)
                
                # Sample reads
                AO_matrix[mut_idx, tp_idx], DP_matrix[mut_idx, tp_idx] = \
                    sample_sequencing(vaf_true, depth)
    
    # Create AnnData
    obs = pd.DataFrame({
        'mutation_id': [f'MUT{i+1}' for i in range(n_mutations)],
        'gene': [f'GENE{i+1}' for i in range(n_mutations)],
        'p_key': [f'p.Arg{100+i}Cys' for i in range(n_mutations)],
        'clone_id': [mutation_to_clone[i] for i in range(n_mutations)]
    })
    obs.index = obs['mutation_id']
    
    var = pd.DataFrame({
        'timepoint': np.arange(n_timepoints),
        'time_points': timepoints
    })
    var.index = [f'T{i}' for i in range(n_timepoints)]
    
    # Main data matrix (observed VAF)
    X = AO_matrix / np.maximum(DP_matrix, 1)
    
    # Create AnnData
    part = ad.AnnData(X=X, obs=obs, var=var)
    part.layers['AO'] = AO_matrix
    part.layers['DP'] = DP_matrix
    part.uns['participant_id'] = participant_id
    part.uns['cohort'] = 'SYNTHETIC_MULTICLONE'
    
    # Ground truth
    ground_truth = {
        'participant_id': participant_id,
        'n_timepoints': n_timepoints,
        'time_span': time_span,
        'n_clones': n_clones,
        'n_mutations': n_mutations,
        'clonal_structure': clonal_structure,
        'clonal_structure_str': str(clonal_structure),  # For CSV
    }
    
    # Add per-clone ground truth
    for clone_idx in range(n_clones):
        params = clone_params[clone_idx]
        traj = clone_trajectories[clone_idx]
        
        ground_truth[f'clone_{clone_idx}_zygosity_type'] = params['zygosity_type']
        ground_truth[f'clone_{clone_idx}_h'] = params['h']
        ground_truth[f'clone_{clone_idx}_s'] = params['s']
        ground_truth[f'clone_{clone_idx}_N0_total'] = params['N0_total']
        ground_truth[f'clone_{clone_idx}_N_final'] = traj['N_het'][-1] + traj['N_hom'][-1]
        ground_truth[f'clone_{clone_idx}_mutations'] = str(clonal_structure[clone_idx])
        
        # VAF for leading mutation in clone
        leading_mut = clonal_structure[clone_idx][0]
        ground_truth[f'clone_{clone_idx}_vaf_initial'] = VAF_true_matrix[leading_mut, 0]
        ground_truth[f'clone_{clone_idx}_vaf_final'] = VAF_true_matrix[leading_mut, -1]
    
    ground_truth['mean_depth'] = DP_matrix.mean()
    
    return part, ground_truth


def generate_synthetic_cohort_multiclone():
    """
    Generate full synthetic cohort with multiple clones.
    """
    
    np.random.seed(RANDOM_SEED)
    
    print("="*80)
    print("SYNTHETIC MULTI-CLONE DATA GENERATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Participants: {N_PARTICIPANTS}")
    print(f"  Clones per participant: {N_CLONES_RANGE}")
    print(f"  Mutations per clone: {N_MUTATIONS_PER_CLONE_RANGE}")
    print(f"  Total mutations: {N_CLONES_RANGE[0]*N_MUTATIONS_PER_CLONE_RANGE[0]} - "
          f"{N_CLONES_RANGE[1]*N_MUTATIONS_PER_CLONE_RANGE[1]}")
    print(f"  Timepoints: {N_TIMEPOINTS_RANGE}")
    print(f"  Time span: {TIME_SPAN_RANGE} years")
    print(f"  Fitness range: {FITNESS_RANGE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()
    
    cohort = []
    ground_truths = []
    
    for i in range(N_PARTICIPANTS):
        participant_id = f'SYN_{i+1:03d}'
        
        print(f"[{i+1}/{N_PARTICIPANTS}] Generating {participant_id}...", end=' ')
        
        part, gt = generate_synthetic_participant_multiclone(participant_id)
        
        cohort.append(part)
        ground_truths.append(gt)
        
        cs_str = str(gt['clonal_structure'])
        print(f"✓ ({gt['n_clones']} clones, {gt['n_mutations']} muts, CS={cs_str})")
    
    ground_truth_df = pd.DataFrame(ground_truths)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(cohort)} participants")
    print(f"\nClone distribution:")
    print(ground_truth_df['n_clones'].value_counts().sort_index())
    print(f"\nMutation distribution:")
    print(ground_truth_df['n_mutations'].value_counts().sort_index())
    print(f"\nClonal structures (first 10):")
    for i in range(min(10, len(ground_truth_df))):
        print(f"  {ground_truth_df.iloc[i]['participant_id']}: "
              f"{ground_truth_df.iloc[i]['clonal_structure_str']}")
    
    return cohort, ground_truth_df


# ==============================================================================
# Save Functions
# ==============================================================================

def save_synthetic_data(cohort, ground_truth_df):
    """Save synthetic cohort and ground truth."""
    
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
    print(f"\nTo test clonal structure inference:")
    print(f"  1. Load: cohort = pickle.load(open('{cohort_file}', 'rb'))")
    print(f"  2. Run clonal structure inference")
    print(f"  3. Compare inferred structure to ground truth:")
    print(f"     gt = pd.read_csv('{gt_file}')")
    print(f"     Check 'clonal_structure' column matches inference")


def validate_synthetic_data(cohort, ground_truth_df):
    """Validate synthetic data."""
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    issues = []
    
    for i, (part, gt) in enumerate(zip(cohort, ground_truth_df.to_dict('records'))):
        pid = part.uns['participant_id']
        
        # Check structure
        if part.shape[0] != gt['n_mutations']:
            issues.append(f"{pid}: Mutation count mismatch")
        
        if part.shape[1] != gt['n_timepoints']:
            issues.append(f"{pid}: Timepoint mismatch")
        
        # Check VAF range
        AO = part.layers['AO']
        DP = part.layers['DP']
        VAF = AO / np.maximum(DP, 1)
        
        if np.any(VAF < 0) or np.any(VAF > 1):
            issues.append(f"{pid}: VAF out of range [0,1]")
        
        # Check that mutations in same clone have similar VAF
        clonal_structure = eval(gt['clonal_structure_str'])
        for clone_idx, clone_muts in enumerate(clonal_structure):
            if len(clone_muts) > 1:
                # Get VAFs for mutations in this clone
                clone_vafs = VAF[clone_muts, :]
                
                # Check correlation across time
                for tp in range(part.shape[1]):
                    vaf_std = clone_vafs[:, tp].std()
                    vaf_mean = clone_vafs[:, tp].mean()
                    
                    # Coefficient of variation should be small (sequencing noise only)
                    if vaf_mean > 0.01:  # Ignore very low VAFs
                        cv = vaf_std / vaf_mean
                        if cv > 0.3:  # 30% CV threshold
                            issues.append(
                                f"{pid}: Clone {clone_idx} mutations have high VAF variance "
                                f"at t={tp} (CV={cv:.2f})"
                            )
                            break
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
    else:
        print("✅ All validation checks passed!")
    
    # Summary
    print(f"\nData statistics:")
    print(f"  Participants: {len(cohort)}")
    print(f"  Total mutations: {sum(p.shape[0] for p in cohort)}")
    print(f"  Avg mutations per participant: {np.mean([p.shape[0] for p in cohort]):.1f}")
    print(f"  Total timepoints: {sum(p.shape[1] for p in cohort)}")
    
    all_depths = np.concatenate([p.layers['DP'].flatten() for p in cohort])
    print(f"  Sequencing depth: {all_depths.mean():.0f} ± {all_depths.std():.0f}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Generate and save synthetic multi-clone cohort."""
    
    cohort, ground_truth_df = generate_synthetic_cohort_multiclone()
    validate_synthetic_data(cohort, ground_truth_df)
    save_synthetic_data(cohort, ground_truth_df)


if __name__ == '__main__':
    main()