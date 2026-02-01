import sys
sys.path.append("..")
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# ==============================================================================
# Synthetic Data Generation with Known Ground Truth
# ==============================================================================

class GroundTruth:
    """Store ground truth parameters for validation"""
    def __init__(self):
        self.clones = []
        self.fitness_values = {}
        self.zygosity = {}
        self.initial_sizes = {}
        
    def add_clone(self, clone_id, mutations, fitness, zygosity, initial_size):
        """Add a clone with known parameters"""
        self.clones.append({
            'id': clone_id,
            'mutations': mutations,
            'fitness': fitness,
            'zygosity': zygosity,
            'initial_size': initial_size
        })
        
        for mut in mutations:
            self.fitness_values[mut] = fitness
            self.zygosity[mut] = zygosity
            self.initial_sizes[mut] = initial_size
    
    def get_summary(self):
        """Get summary of ground truth"""
        summary = {
            'n_clones': len(self.clones),
            'fitness_values': list(self.fitness_values.values()),
            'zygosity_types': list(self.zygosity.values()),
            'mean_fitness': np.mean(list(self.fitness_values.values())),
            'fitness_range': [min(self.fitness_values.values()), 
                            max(self.fitness_values.values())]
        }
        return summary


def simulate_clone_growth(initial_size, fitness, time_points, noise_std=0.05):
    """
    Simulate exponential clone growth with fitness advantage
    
    Parameters:
    -----------
    initial_size : float
        Initial clone size at t=0
    fitness : float
        Fitness coefficient (s)
    time_points : array
        Time points to simulate (in years)
    noise_std : float
        Std deviation of multiplicative noise
    
    Returns:
    --------
    sizes : array
        Clone sizes at each time point
    """
    # Exponential growth: N(t) = N(0) * exp(s * t)
    sizes = initial_size * np.exp(fitness * time_points)
    
    # Add biological noise (multiplicative)
    noise = np.random.lognormal(mean=0, sigma=noise_std, size=len(time_points))
    sizes = sizes * noise
    
    return sizes


def generate_vaf_from_clone_size(het_size, hom_size, total_cells=1e6, 
                                  zygosity='heterozygous'):
    """
    Generate VAF from clone size and zygosity
    
    Parameters:
    -----------
    het_size : float
        Number of heterozygous cells
    hom_size : float
        Number of homozygous cells
    total_cells : float
        Total cell population
    zygosity : str
        'heterozygous', 'homozygous', or 'mixed'
    
    Returns:
    --------
    vaf : float
        Variant allele frequency
    """
    if zygosity == 'heterozygous':
        # Het cells contribute 1 mutant allele, 1 WT allele
        mutant_alleles = het_size
        total_alleles = 2 * total_cells
        
    elif zygosity == 'homozygous':
        # Hom cells contribute 2 mutant alleles
        mutant_alleles = 2 * hom_size
        total_alleles = 2 * total_cells
        
    elif zygosity == 'mixed':
        # Mix of het and hom cells
        mutant_alleles = het_size + 2 * hom_size
        total_alleles = 2 * total_cells
    
    else:
        raise ValueError(f"Unknown zygosity: {zygosity}")
    
    vaf = mutant_alleles / total_alleles
    return np.clip(vaf, 0, 1)  # Ensure VAF is in [0,1]


def sample_sequencing_reads(true_vaf, depth, overdispersion=1.2):
    """
    Sample sequencing reads with binomial + overdispersion
    
    Parameters:
    -----------
    true_vaf : float
        True VAF
    depth : int
        Sequencing depth
    overdispersion : float
        Beta-binomial overdispersion parameter (1.0 = binomial)
    
    Returns:
    --------
    ao : int
        Alternate allele count
    dp : int
        Total depth
    """
    if overdispersion == 1.0:
        # Standard binomial sampling
        ao = np.random.binomial(depth, true_vaf)
    else:
        # Beta-binomial for overdispersion
        alpha = true_vaf / overdispersion
        beta = (1 - true_vaf) / overdispersion
        
        # Sample VAF from beta, then reads from binomial
        sampled_vaf = np.random.beta(alpha, beta)
        ao = np.random.binomial(depth, sampled_vaf)
    
    return ao, depth


def create_synthetic_participant(participant_id, n_clones=3, n_timepoints=5,
                                time_span=10, mean_depth=100,
                                total_cells=1e6):
    """
    Create a synthetic participant with known ground truth
    
    Parameters:
    -----------
    participant_id : str
        Participant identifier
    n_clones : int
        Number of clones
    n_timepoints : int
        Number of time points
    time_span : float
        Time span in years
    mean_depth : int
        Mean sequencing depth
    total_cells : float
        Total cell population
    
    Returns:
    --------
    part : AnnData-like object
        Synthetic participant data
    ground_truth : GroundTruth
        Known ground truth parameters
    """
    # Generate time points
    time_points = np.linspace(0, time_span, n_timepoints)
    
    # Initialize ground truth
    gt = GroundTruth()
    
    # Clone parameters (vary fitness and zygosity)
    clone_configs = []
    
    # Clone 1: Strong positive fitness, heterozygous
    clone_configs.append({
        'fitness': 0.15,
        'zygosity': 'heterozygous',
        'initial_size': 1000,
        'n_mutations': 2
    })
    
    # Clone 2: Weak positive fitness, homozygous
    if n_clones >= 2:
        clone_configs.append({
            'fitness': 0.05,
            'zygosity': 'homozygous',
            'initial_size': 500,
            'n_mutations': 1
        })
    
    # Clone 3: Neutral/slightly negative, mixed
    if n_clones >= 3:
        clone_configs.append({
            'fitness': -0.02,
            'zygosity': 'mixed',
            'initial_size': 2000,
            'n_mutations': 3
        })
    
    # Additional clones (random parameters)
    for i in range(len(clone_configs), n_clones):
        clone_configs.append({
            'fitness': np.random.uniform(-0.05, 0.20),
            'zygosity': np.random.choice(['heterozygous', 'homozygous', 'mixed'],
                                        p=[0.5, 0.3, 0.2]),
            'initial_size': np.random.uniform(500, 3000),
            'n_mutations': np.random.randint(1, 4)
        })
    
    # Generate mutations for each clone
    all_mutations = []
    mutation_counter = 0
    
    for clone_idx, config in enumerate(clone_configs):
        clone_mutations = []
        
        for _ in range(config['n_mutations']):
            mut_name = f"MUT_{mutation_counter:03d}"
            clone_mutations.append(mut_name)
            mutation_counter += 1
        
        all_mutations.extend(clone_mutations)
        
        # Add to ground truth
        gt.add_clone(
            clone_id=f"Clone_{clone_idx+1}",
            mutations=clone_mutations,
            fitness=config['fitness'],
            zygosity=config['zygosity'],
            initial_size=config['initial_size']
        )
    
    n_mutations = len(all_mutations)
    
    # Initialize data matrices
    AO = np.zeros((n_mutations, n_timepoints), dtype=int)
    DP = np.zeros((n_mutations, n_timepoints), dtype=int)
    true_VAF = np.zeros((n_mutations, n_timepoints))
    
    # Simulate each clone's growth and generate VAFs
    for clone_idx, config in enumerate(clone_configs):
        # Simulate clone growth
        clone_size = simulate_clone_growth(
            initial_size=config['initial_size'],
            fitness=config['fitness'],
            time_points=time_points
        )
        
        # Determine het/hom split based on zygosity
        if config['zygosity'] == 'heterozygous':
            het_sizes = clone_size
            hom_sizes = np.zeros_like(clone_size)
            h = 0.0
            
        elif config['zygosity'] == 'homozygous':
            het_sizes = np.zeros_like(clone_size)
            hom_sizes = clone_size
            h = 1.0
            
        elif config['zygosity'] == 'mixed':
            # Mixed: some cells het, some hom (evolving over time)
            h = 0.3 + 0.2 * time_points / time_points[-1]  # Increasing h over time
            het_sizes = clone_size * (1 - h)
            hom_sizes = clone_size * h
        
        # Get mutations for this clone
        clone_mut_indices = []
        for mut_name in gt.clones[clone_idx]['mutations']:
            mut_idx = all_mutations.index(mut_name)
            clone_mut_indices.append(mut_idx)
        
        # Generate VAF for each mutation in clone
        for mut_idx in clone_mut_indices:
            for t_idx in range(n_timepoints):
                # Generate true VAF
                vaf = generate_vaf_from_clone_size(
                    het_size=het_sizes[t_idx],
                    hom_size=hom_sizes[t_idx],
                    total_cells=total_cells,
                    zygosity=config['zygosity']
                )
                
                true_VAF[mut_idx, t_idx] = vaf
                
                # Sample sequencing depth (Poisson around mean)
                depth = np.random.poisson(mean_depth)
                
                # Sample reads
                ao, dp = sample_sequencing_reads(vaf, depth)
                
                AO[mut_idx, t_idx] = ao
                DP[mut_idx, t_idx] = dp
    
    # Create AnnData-like structure
    class SyntheticParticipant:
        def __init__(self):
            self.obs = None
            self.var = None
            self.layers = {}
            self.uns = {}
            self.shape = (0, 0)
    
    part = SyntheticParticipant()
    
    # Observations (mutations)
    obs_data = {
        'mutation': all_mutations,
        'true_fitness': [gt.fitness_values[m] for m in all_mutations],
        'true_zygosity': [gt.zygosity[m] for m in all_mutations],
        'mean_vaf': true_VAF.mean(axis=1),
        'mean_depth': DP.mean(axis=1)
    }
    part.obs = pd.DataFrame(obs_data)
    part.obs.index = all_mutations
    
    # Variables (time points)
    var_data = {
        'time_points': time_points,
        'sample_id': [f't{i}' for i in range(n_timepoints)]
    }
    part.var = pd.DataFrame(var_data)
    
    # Layers
    part.layers['AO'] = AO
    part.layers['DP'] = DP
    part.layers['true_VAF'] = true_VAF
    
    # Metadata
    part.uns['participant_id'] = participant_id
    part.uns['ground_truth'] = gt
    part.uns['total_cells'] = total_cells
    
    part.shape = (n_mutations, n_timepoints)
    
    return part, gt


def generate_synthetic_cohort(n_participants=5, output_path=None):
    """
    Generate a synthetic cohort with multiple participants
    
    Parameters:
    -----------
    n_participants : int
        Number of participants to generate
    output_path : str, optional
        Path to save synthetic cohort
    
    Returns:
    --------
    synthetic_cohort : list
        List of synthetic participants
    ground_truths : list
        List of ground truth objects
    """
    synthetic_cohort = []
    ground_truths = []
    
    print("="*80)
    print("GENERATING SYNTHETIC COHORT")
    print("="*80)
    
    for i in range(n_participants):
        # Vary parameters across participants
        n_clones = np.random.randint(2, 5)
        n_timepoints = np.random.randint(4, 7)
        time_span = np.random.uniform(5, 15)
        mean_depth = np.random.randint(80, 150)
        
        print(f"\n[{i+1}/{n_participants}] Generating participant_{i+1}:")
        print(f"  Clones: {n_clones}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Time span: {time_span:.1f} years")
        print(f"  Mean depth: {mean_depth}x")
        
        part, gt = create_synthetic_participant(
            participant_id=f"participant_{i+1}",
            n_clones=n_clones,
            n_timepoints=n_timepoints,
            time_span=time_span,
            mean_depth=mean_depth
        )
        
        synthetic_cohort.append(part)
        ground_truths.append(gt)
        
        # Print ground truth
        print(f"  Ground truth:")
        for clone in gt.clones:
            print(f"    {clone['id']}: s={clone['fitness']:.3f}, "
                  f"zyg={clone['zygosity']}, n_mut={len(clone['mutations'])}")
    
    print("\n" + "="*80)
    print(f"✅ Generated {n_participants} synthetic participants")
    print("="*80)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'wb') as f:
            pk.dump(synthetic_cohort, f)
        print(f"✅ Saved to: {output_path}")
        
        # Save ground truths separately
        gt_path = output_path.replace('.pk', '_ground_truth.pk')
        with open(gt_path, 'wb') as f:
            pk.dump(ground_truths, f)
        print(f"✅ Saved ground truths to: {gt_path}")
    
    return synthetic_cohort, ground_truths


# ==============================================================================
# Visualization Functions for Synthetic Data
# ==============================================================================

def plot_synthetic_vaf_trajectories(part, ground_truth, figsize=(14, 8)):
    """
    Plot VAF trajectories for synthetic data with ground truth overlay
    """
    participant_id = part.uns['participant_id']
    time_points = part.var.time_points.values
    
    AO = part.layers['AO']
    DP = part.layers['DP']
    observed_VAF = AO / np.maximum(DP, 1)
    true_VAF = part.layers['true_VAF']
    
    n_clones = len(ground_truth.clones)
    colors = sns.color_palette("husl", n_clones)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Plot 1: VAF trajectories
    ax = axes[0]
    
    for clone_idx, clone in enumerate(ground_truth.clones):
        color = colors[clone_idx]
        
        for mut_name in clone['mutations']:
            mut_idx = list(part.obs.index).index(mut_name)
            
            # True VAF (solid line)
            ax.plot(time_points, true_VAF[mut_idx, :], '-', 
                   color=color, linewidth=2.5, alpha=0.8,
                   label=f"{clone['id']}: {mut_name} (true)")
            
            # Observed VAF (points with error bars)
            vaf_obs = observed_VAF[mut_idx, :]
            depth = DP[mut_idx, :]
            vaf_std = np.sqrt(vaf_obs * (1 - vaf_obs) / depth)
            
            ax.plot(time_points, vaf_obs, 'o', color=color, 
                   markersize=10, markeredgecolor='black', 
                   markeredgewidth=1.5, alpha=0.7)
            
            ax.errorbar(time_points, vaf_obs, yerr=1.96*vaf_std,
                       fmt='none', color=color, alpha=0.4, 
                       capsize=5, linewidth=1.5)
    
    # Add reference lines
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5,
              label='Het max (0.5)')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5,
              label='Hom max (1.0)')
    
    ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VAF', fontsize=12, fontweight='bold')
    ax.set_title(f'{participant_id}: Synthetic VAF Trajectories\n'
                f'(Solid lines = true VAF, Points = observed VAF)', 
                fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # Plot 2: Ground truth parameters table
    ax_table = axes[1]
    ax_table.axis('off')
    
    table_data = [['Clone', 'Mutations', 'True Fitness', 'Zygosity', 'Initial Size']]
    
    for clone in ground_truth.clones:
        mut_str = ', '.join(clone['mutations'][:3])
        if len(clone['mutations']) > 3:
            mut_str += f' +{len(clone["mutations"])-3}'
        
        table_data.append([
            clone['id'],
            mut_str,
            f"{clone['fitness']:.3f}",
            clone['zygosity'],
            f"{clone['initial_size']:.0f}"
        ])
    
    table = ax_table.table(cellText=table_data, cellLoc='left',
                          loc='center', colWidths=[0.15, 0.35, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.tight_layout()
    return fig


def plot_cohort_ground_truth_summary(ground_truths, figsize=(14, 6)):
    """
    Plot summary of ground truth parameters across cohort
    """
    # Collect all fitness and zygosity data
    all_fitness = []
    all_zygosity = []
    participant_fitness = []
    
    for i, gt in enumerate(ground_truths):
        for clone in gt.clones:
            all_fitness.append(clone['fitness'])
            all_zygosity.append(clone['zygosity'])
        
        # Mean fitness per participant
        mean_fit = np.mean([c['fitness'] for c in gt.clones])
        participant_fitness.append((f"P{i+1}", mean_fit))
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Fitness distribution
    ax = axes[0]
    ax.hist(all_fitness, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(all_fitness), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(all_fitness):.3f}')
    ax.axvline(np.median(all_fitness), color='orange', linestyle='--',
              linewidth=2, label=f'Median: {np.median(all_fitness):.3f}')
    ax.set_xlabel('True Fitness (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Ground Truth Fitness Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zygosity distribution
    ax = axes[1]
    zyg_counts = pd.Series(all_zygosity).value_counts()
    colors = {'heterozygous': 'skyblue', 'homozygous': 'salmon', 'mixed': 'orange'}
    plot_colors = [colors.get(z, 'gray') for z in zyg_counts.index]
    
    ax.pie(zyg_counts.values, labels=zyg_counts.index, autopct='%1.1f%%',
          colors=plot_colors, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Ground Truth Zygosity Distribution', fontsize=12, fontweight='bold')
    
    # Plot 3: Fitness by participant
    ax = axes[2]
    participants, fitness_vals = zip(*participant_fitness)
    colors_p = sns.color_palette("Set2", len(participants))
    ax.barh(range(len(participants)), fitness_vals, color=colors_p, edgecolor='black')
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants)
    ax.set_xlabel('Mean Fitness (s)', fontsize=11, fontweight='bold')
    ax.set_title('Mean Fitness by Participant', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    fig.suptitle('Ground Truth Summary Across Cohort', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ==============================================================================
# Accuracy Testing Functions
# ==============================================================================

def test_fitness_inference_accuracy(inferred_part, ground_truth, tolerance=0.05):
    """
    Test accuracy of fitness inference
    
    Parameters:
    -----------
    inferred_part : participant object
        Participant with inferred fitness values
    ground_truth : GroundTruth
        Ground truth object
    tolerance : float
        Tolerance for fitness error
    
    Returns:
    --------
    results : dict
        Dictionary with accuracy metrics
    """
    results = {
        'n_mutations': 0,
        'correct': 0,
        'errors': [],
        'mean_absolute_error': None,
        'mean_relative_error': None,
        'within_tolerance': 0
    }
    
    if 'fitness' not in inferred_part.obs:
        print("⚠️  No inferred fitness found")
        return results
    
    for mut_name in inferred_part.obs.index:
        if mut_name not in ground_truth.fitness_values:
            continue
        
        true_fitness = ground_truth.fitness_values[mut_name]
        inferred_fitness = inferred_part.obs.loc[mut_name, 'fitness']
        
        if np.isnan(inferred_fitness):
            continue
        
        results['n_mutations'] += 1
        error = abs(inferred_fitness - true_fitness)
        results['errors'].append(error)
        
        if error <= tolerance:
            results['within_tolerance'] += 1
        
        # Check if confidence interval contains true value
        if 'fitness_5' in inferred_part.obs and 'fitness_95' in inferred_part.obs:
            ci_low = inferred_part.obs.loc[mut_name, 'fitness_5']
            ci_high = inferred_part.obs.loc[mut_name, 'fitness_95']
            
            if not np.isnan(ci_low) and not np.isnan(ci_high):
                if ci_low <= true_fitness <= ci_high:
                    results['correct'] += 1
    
    if len(results['errors']) > 0:
        results['mean_absolute_error'] = np.mean(results['errors'])
        
        # Calculate relative error
        rel_errors = []
        for mut_name in inferred_part.obs.index:
            if mut_name in ground_truth.fitness_values:
                true_fitness = ground_truth.fitness_values[mut_name]
                inferred_fitness = inferred_part.obs.loc[mut_name, 'fitness']
                
                if not np.isnan(inferred_fitness) and true_fitness != 0:
                    rel_error = abs(inferred_fitness - true_fitness) / abs(true_fitness)
                    rel_errors.append(rel_error)
        
        if len(rel_errors) > 0:
            results['mean_relative_error'] = np.mean(rel_errors)
    
    return results


def test_clonal_structure_accuracy(inferred_part, ground_truth):
    """
    Test accuracy of clonal structure inference
    
    Returns:
    --------
    results : dict
        Dictionary with accuracy metrics
    """
    results = {
        'n_clones_true': len(ground_truth.clones),
        'n_clones_inferred': 0,
        'clone_match_score': 0.0,
        'mutations_correctly_grouped': 0,
        'total_mutation_pairs': 0
    }
    
    if 'optimal_model' not in inferred_part.uns:
        print("⚠️  No inferred clonal structure found")
        return results
    
    inferred_cs = inferred_part.uns['optimal_model']['clonal_structure']
    results['n_clones_inferred'] = len(inferred_cs)
    
    # Build true clonal structure
    true_cs = []
    for clone in ground_truth.clones:
        mut_indices = []
        for mut_name in clone['mutations']:
            if mut_name in inferred_part.obs.index:
                mut_idx = list(inferred_part.obs.index).index(mut_name)
                mut_indices.append(mut_idx)
        if len(mut_indices) > 0:
            true_cs.append(mut_indices)
    
    # Check if mutations that should be together are together
    for true_clone in true_cs:
        for i in range(len(true_clone)):
            for j in range(i+1, len(true_clone)):
                results['total_mutation_pairs'] += 1
                
                # Check if this pair is in the same inferred clone
                mut_i = true_clone[i]
                mut_j = true_clone[j]
                
                for inferred_clone in inferred_cs:
                    if mut_i in inferred_clone and mut_j in inferred_clone:
                        results['mutations_correctly_grouped'] += 1
                        break
    
    if results['total_mutation_pairs'] > 0:
        results['clone_match_score'] = (results['mutations_correctly_grouped'] / 
                                       results['total_mutation_pairs'])
    
    return results


def test_zygosity_accuracy(inferred_part, ground_truth):
    """
    Test accuracy of zygosity inference
    
    Returns:
    --------
    results : dict
        Dictionary with accuracy metrics
    """
    results = {
        'n_mutations': 0,
        'correct': 0,
        'accuracy': 0.0,
        'confusion_matrix': {}
    }
    
    if 'zygosity_type' not in inferred_part.obs:
        print("⚠️  No inferred zygosity found")
        return results
    
    zyg_types = ['heterozygous', 'homozygous', 'mixed']
    for true_zyg in zyg_types:
        results['confusion_matrix'][true_zyg] = {inf_zyg: 0 for inf_zyg in zyg_types}
    
    for mut_name in inferred_part.obs.index:
        if mut_name not in ground_truth.zygosity:
            continue
        
        true_zyg = ground_truth.zygosity[mut_name]
        inferred_zyg = inferred_part.obs.loc[mut_name, 'zygosity_type']
        
        results['n_mutations'] += 1
        results['confusion_matrix'][true_zyg][inferred_zyg] += 1
        
        if true_zyg == inferred_zyg:
            results['correct'] += 1
    
    if results['n_mutations'] > 0:
        results['accuracy'] = results['correct'] / results['n_mutations']
    
    return results


def plot_inference_accuracy_comparison(inferred_part, ground_truth, figsize=(16, 5)):
    """
    Create comprehensive accuracy comparison plots
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Fitness comparison (true vs inferred)
    ax = axes[0]
    
    if 'fitness' in inferred_part.obs:
        true_fit = []
        inferred_fit = []
        
        for mut_name in inferred_part.obs.index:
            if mut_name in ground_truth.fitness_values:
                true_val = ground_truth.fitness_values[mut_name]
                inf_val = inferred_part.obs.loc[mut_name, 'fitness']
                
                if not np.isnan(inf_val):
                    true_fit.append(true_val)
                    inferred_fit.append(inf_val)
        
        if len(true_fit) > 0:
            ax.scatter(true_fit, inferred_fit, s=100, alpha=0.6, 
                      edgecolors='black', linewidths=1.5, color='steelblue')
            
            # Add diagonal line (perfect inference)
            min_val = min(min(true_fit), min(inferred_fit))
            max_val = max(max(true_fit), max(inferred_fit))
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect inference')
            
            # Calculate R²
            correlation = np.corrcoef(true_fit, inferred_fit)[0, 1]
            mae = np.mean(np.abs(np.array(true_fit) - np.array(inferred_fit)))
            
            ax.text(0.05, 0.95, f'R = {correlation:.3f}\nMAE = {mae:.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('True Fitness', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inferred Fitness', fontsize=11, fontweight='bold')
    ax.set_title('Fitness Inference Accuracy', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zygosity confusion matrix
    ax = axes[1]
    
    if 'zygosity_type' in inferred_part.obs:
        zyg_types = ['heterozygous', 'homozygous', 'mixed']
        conf_matrix = np.zeros((3, 3))
        
        for i, true_zyg in enumerate(zyg_types):
            for j, inf_zyg in enumerate(zyg_types):
                count = 0
                for mut_name in inferred_part.obs.index:
                    if mut_name in ground_truth.zygosity:
                        if (ground_truth.zygosity[mut_name] == true_zyg and
                            inferred_part.obs.loc[mut_name, 'zygosity_type'] == inf_zyg):
                            count += 1
                conf_matrix[i, j] = count
        
        im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['Het', 'Hom', 'Mix'])
        ax.set_yticklabels(['Het', 'Hom', 'Mix'])
        ax.set_xlabel('Inferred', fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_title('Zygosity Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, int(conf_matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=12)
        
        plt.colorbar(im, ax=ax)
    
    # Plot 3: Error distribution
    ax = axes[2]
    
    if 'fitness' in inferred_part.obs:
        errors = []
        for mut_name in inferred_part.obs.index:
            if mut_name in ground_truth.fitness_values:
                true_val = ground_truth.fitness_values[mut_name]
                inf_val = inferred_part.obs.loc[mut_name, 'fitness']
                
                if not np.isnan(inf_val):
                    errors.append(inf_val - true_val)
        
        if len(errors) > 0:
            ax.hist(errors, bins=15, edgecolor='black', alpha=0.7, color='coral')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, 
                      label='Zero error')
            ax.axvline(np.mean(errors), color='blue', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
            
            ax.text(0.05, 0.95, f'Std: {np.std(errors):.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.set_xlabel('Fitness Error (Inferred - True)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Fitness Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    participant_id = inferred_part.uns.get('participant_id', 'Unknown')
    fig.suptitle(f'Inference Accuracy: {participant_id}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ==============================================================================
# Main Testing Function
# ==============================================================================

def run_full_accuracy_test(inference_function, output_dir='../exports/TEST/accuracy_test/'):
    """
    Run complete accuracy test pipeline
    
    Parameters:
    -----------
    inference_function : callable
        Function that takes a participant and returns it with inferred values
        Should add 'optimal_model', 'fitness', 'zygosity_type' etc.
    output_dir : str
        Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CLONAL FITNESS INFERENCE ACCURACY TEST")
    print("="*80)
    
    # Step 1: Generate synthetic cohort
    print("\n[STEP 1] Generating synthetic cohort...")
    synthetic_cohort, ground_truths = generate_synthetic_cohort(
        n_participants=5,
        output_path=f'{output_dir}synthetic_cohort.pk'
    )
    
    # Step 2: Visualize synthetic data
    print("\n[STEP 2] Visualizing synthetic data...")
    
    for i, (part, gt) in enumerate(zip(synthetic_cohort, ground_truths)):
        fig = plot_synthetic_vaf_trajectories(part, gt)
        fig.savefig(f'{output_dir}{part.uns["participant_id"]}_synthetic_vaf.png',
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ✅ Saved VAF plot for {part.uns['participant_id']}")
    
    # Cohort ground truth summary
    fig = plot_cohort_ground_truth_summary(ground_truths)
    fig.savefig(f'{output_dir}cohort_ground_truth_summary.png',
               bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved cohort ground truth summary")
    
    # Step 3: Run inference on synthetic data
    print("\n[STEP 3] Running inference on synthetic data...")
    
    inferred_cohort = []
    for i, part in enumerate(synthetic_cohort):
        print(f"\n  [{i+1}/{len(synthetic_cohort)}] Inferring {part.uns['participant_id']}...")
        
        try:
            inferred_part = inference_function(part)
            inferred_cohort.append(inferred_part)
            print(f"    ✅ Inference complete")
        except Exception as e:
            print(f"    ❌ Inference failed: {e}")
            inferred_cohort.append(part)  # Keep original if inference fails
    
    # Step 4: Test accuracy
    print("\n[STEP 4] Testing inference accuracy...")
    
    all_fitness_results = []
    all_structure_results = []
    all_zygosity_results = []
    
    for i, (inferred_part, gt) in enumerate(zip(inferred_cohort, ground_truths)):
        print(f"\n  [{i+1}/{len(inferred_cohort)}] Testing {inferred_part.uns['participant_id']}...")
        
        # Fitness accuracy
        fitness_results = test_fitness_inference_accuracy(inferred_part, gt)
        all_fitness_results.append(fitness_results)
        
        if fitness_results['n_mutations'] > 0:
            print(f"    Fitness MAE: {fitness_results['mean_absolute_error']:.4f}")
            print(f"    Within tolerance: {fitness_results['within_tolerance']}/{fitness_results['n_mutations']}")
            print(f"    CI coverage: {fitness_results['correct']}/{fitness_results['n_mutations']}")
        
        # Clonal structure accuracy
        structure_results = test_clonal_structure_accuracy(inferred_part, gt)
        all_structure_results.append(structure_results)
        
        print(f"    Clones (true/inferred): {structure_results['n_clones_true']}/{structure_results['n_clones_inferred']}")
        print(f"    Clone match score: {structure_results['clone_match_score']:.2f}")
        
        # Zygosity accuracy
        zygosity_results = test_zygosity_accuracy(inferred_part, gt)
        all_zygosity_results.append(zygosity_results)
        
        if zygosity_results['n_mutations'] > 0:
            print(f"    Zygosity accuracy: {zygosity_results['accuracy']:.2f}")
        
        # Plot accuracy comparison
        fig = plot_inference_accuracy_comparison(inferred_part, gt)
        fig.savefig(f'{output_dir}{inferred_part.uns["participant_id"]}_accuracy.png',
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # Step 5: Summary statistics
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)
    
    # Fitness accuracy summary
    print("\n[FITNESS INFERENCE]")
    valid_fitness = [r for r in all_fitness_results if r['n_mutations'] > 0]
    if len(valid_fitness) > 0:
        mean_mae = np.mean([r['mean_absolute_error'] for r in valid_fitness])
        mean_mre = np.mean([r['mean_relative_error'] for r in valid_fitness 
                           if r['mean_relative_error'] is not None])
        total_muts = sum(r['n_mutations'] for r in valid_fitness)
        total_within_tol = sum(r['within_tolerance'] for r in valid_fitness)
        total_ci_correct = sum(r['correct'] for r in valid_fitness)
        
        print(f"  Mean Absolute Error: {mean_mae:.4f}")
        print(f"  Mean Relative Error: {mean_mre:.2%}")
        print(f"  Within tolerance (±0.05): {total_within_tol}/{total_muts} ({total_within_tol/total_muts:.1%})")
        print(f"  95% CI coverage: {total_ci_correct}/{total_muts} ({total_ci_correct/total_muts:.1%})")
    
    # Clonal structure summary
    print("\n[CLONAL STRUCTURE]")
    valid_structure = [r for r in all_structure_results if r['total_mutation_pairs'] > 0]
    if len(valid_structure) > 0:
        mean_match_score = np.mean([r['clone_match_score'] for r in valid_structure])
        clone_errors = [abs(r['n_clones_true'] - r['n_clones_inferred']) 
                       for r in all_structure_results]
        
        print(f"  Mean clone match score: {mean_match_score:.2f}")
        print(f"  Mean clone number error: {np.mean(clone_errors):.2f}")
        print(f"  Exact clone number: {sum(1 for e in clone_errors if e == 0)}/{len(clone_errors)}")
    
    # Zygosity accuracy summary
    print("\n[ZYGOSITY INFERENCE]")
    valid_zygosity = [r for r in all_zygosity_results if r['n_mutations'] > 0]
    if len(valid_zygosity) > 0:
        mean_accuracy = np.mean([r['accuracy'] for r in valid_zygosity])
        print(f"  Mean accuracy: {mean_accuracy:.2%}")
    
    print("\n" + "="*80)
    print(f"✅ All results saved to: {output_dir}")
    print("="*80)
    
    return {
        'synthetic_cohort': synthetic_cohort,
        'ground_truths': ground_truths,
        'inferred_cohort': inferred_cohort,
        'fitness_results': all_fitness_results,
        'structure_results': all_structure_results,
        'zygosity_results': all_zygosity_results
    }


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == '__main__':
    print("="*80)
    print("INFERENCE ACCURACY TESTING FRAMEWORK")
    print("="*80)
    print("\nThis script provides:")
    print("  1. Synthetic data generation with known ground truth")
    print("  2. VAF trajectory visualization")
    print("  3. Accuracy testing for fitness, clonal structure, and zygosity")
    print("  4. Comprehensive accuracy reports")
    print("\n" + "="*80)
    
    # Example: Generate synthetic data only
    print("\nGenerating example synthetic cohort...")
    synthetic_cohort, ground_truths = generate_synthetic_cohort(
        n_participants=3,
        output_path='../exports/TEST/synthetic_test_cohort.pk'
    )
    
    print("\nGenerating VAF visualizations...")
    import os
    os.makedirs('../exports/TEST/figures/', exist_ok=True)
    
    for part, gt in zip(synthetic_cohort, ground_truths):
        fig = plot_synthetic_vaf_trajectories(part, gt)
        fig.savefig(f'../exports/TEST/figures/{part.uns["participant_id"]}_synthetic_vaf.png',
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ✅ Saved {part.uns['participant_id']}_synthetic_vaf.png")
    
    fig = plot_cohort_ground_truth_summary(ground_truths)
    fig.savefig('../exports/TEST/figures/cohort_ground_truth.png',
               bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved cohort_ground_truth.png")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    print("\nTo run full accuracy test with your inference pipeline:")
    print("  1. Define your inference function:")
    print("     def my_inference(participant):")
    print("         # Run your inference pipeline")
    print("         # Add results to participant.uns['optimal_model'], participant.obs, etc.")
    print("         return participant")
    print("\n  2. Run full test:")
    print("     results = run_full_accuracy_test(my_inference)")
    print("\n" + "="*80)