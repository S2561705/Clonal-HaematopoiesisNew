import numpy as np
import pandas as pd
from scipy.stats import binom, nbinom
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_clonal_data(
    n_clones=2,
    n_mutations_per_clone=1,  # Changed default to 1
    n_timepoints=5,
    time_points=None,
    fitness_values=None,
    N_w=1e5,
    lamb=1.3,
    depth_mean=100,
    depth_std=20,
    seed=42,
    plot_during_generation=True  # New parameter
):
    """
    Generate synthetic clonal evolution data with known ground truth.
    
    Parameters:
    -----------
    n_clones : int
        Number of clones
    n_mutations_per_clone : int or list
        Number of mutations per clone (if int, same for all clones)
    n_timepoints : int
        Number of sampling timepoints
    time_points : array-like, optional
        Specific timepoints (default: evenly spaced from 0 to 10)
    fitness_values : array-like, optional
        Fitness values for each clone (default: random between 0.1 and 0.8)
    N_w : float
        Wild-type population size
    lamb : float
        Birth rate parameter
    depth_mean : int
        Mean sequencing depth
    depth_std : int
        Standard deviation of sequencing depth
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict with keys:
        - 'AO': Alternate allele observations (n_timepoints x n_mutations)
        - 'DP': Total depth (n_timepoints x n_mutations)
        - 'time_points': Timepoints array
        - 'ground_truth': Dict with true values (clonal_structure, fitness, sizes)
    """
    
    np.random.seed(seed)
    
    # Setup timepoints
    if time_points is None:
        time_points = np.linspace(0, 10, n_timepoints)
    else:
        n_timepoints = len(time_points)
        
    # Setup mutations per clone
    if isinstance(n_mutations_per_clone, int):
        mutations_per_clone = [n_mutations_per_clone] * n_clones
    else:
        mutations_per_clone = n_mutations_per_clone
        
    n_mutations = sum(mutations_per_clone)
    
    # Setup fitness values
    if fitness_values is None:
        fitness_values = np.random.uniform(0.1, 0.8, n_clones)
    else:
        fitness_values = np.array(fitness_values)
        
    # Create clonal structure
    clonal_structure = []
    mut_idx = 0
    for n_mut in mutations_per_clone:
        clonal_structure.append(list(range(mut_idx, mut_idx + n_mut)))
        mut_idx += n_mut
        
    print(f"Ground truth clonal structure: {clonal_structure}")
    print(f"Ground truth fitness values: {fitness_values}")
    
    # Initialize arrays
    AO = np.zeros((n_timepoints, n_mutations))
    DP = np.zeros((n_timepoints, n_mutations))
    
    # Track true clone sizes over time
    clone_sizes = np.zeros((n_timepoints, n_clones))
    mutation_sizes_het = np.zeros((n_timepoints, n_mutations))
    mutation_sizes_hom = np.zeros((n_timepoints, n_mutations))
    
    # Generate initial clone sizes (larger starting populations)
    # Start with 1-5% of wild-type population
    initial_sizes = np.random.uniform(N_w * 0.01, N_w * 0.05, n_clones)
    clone_sizes[0] = initial_sizes
    
    print(f"  Initial clone sizes: {initial_sizes}")
    
    # Simulate clone growth over time using birth-death process
    for t in range(1, n_timepoints):
        dt = time_points[t] - time_points[t-1]
        
        for c in range(n_clones):
            s = fitness_values[c]
            prev_size = clone_sizes[t-1, c]
            
            # Mean and variance of birth-death process
            exp_term = np.exp(dt * s)
            mean = prev_size * exp_term
            
            # Add more variance for realistic stochasticity
            variance = prev_size * (2*lamb + s) * exp_term * (exp_term - 1) / max(s, 1e-8)
            variance = max(variance, mean * 1.5)  # Ensure sufficient variance
            
            # Ensure valid negative binomial parameters
            if variance <= mean:
                variance = mean * 2.0
                
            # Negative binomial parameters
            p = mean / variance
            n = mean**2 / max(variance - mean, 1e-8)
            
            # Sample new size
            new_size = nbinom.rvs(n=n, p=p)
            # Ensure growth (with some minimum)
            new_size = max(new_size, prev_size * 0.8)  # Don't shrink too much
            clone_sizes[t, c] = new_size
    
    print(f"  Final clone sizes: {clone_sizes[-1]}")
    
    # Assign mutations to het/hom randomly (with bias toward het)
    het_prob = 0.7  # 70% chance of heterozygous
    mutation_zygosity = np.random.choice(['het', 'hom'], size=n_mutations, p=[het_prob, 1-het_prob])
    
    # Create mutation names early (needed for diagnostics)
    mutation_names = []
    for c, clone_muts in enumerate(clonal_structure):
        for i, m in enumerate(clone_muts):
            mutation_names.append(f"Clone{c}_Mut{i}")
    
    # For each mutation, determine its size based on clone size
    mut_idx = 0
    for c, clone_muts in enumerate(clonal_structure):
        for m in clone_muts:
            # Mutation size is close to full clone size (with small noise)
            fraction = np.random.uniform(0.9, 1.0)  # 90-100% of clone carries mutation
            
            for t in range(n_timepoints):
                total_mut_size = clone_sizes[t, c] * fraction
                
                if mutation_zygosity[m] == 'het':
                    mutation_sizes_het[t, m] = total_mut_size
                    mutation_sizes_hom[t, m] = 0
                else:
                    mutation_sizes_het[t, m] = 0
                    mutation_sizes_hom[t, m] = total_mut_size
    
    # Calculate true VAFs and generate sequencing observations
    # Adjust N_w to ensure VAFs are in reasonable range
    total_cells = N_w + clone_sizes.sum(axis=1)
    
    print(f"  Total cells over time: {total_cells}")
    print(f"  Wild-type fraction: {(N_w / total_cells).round(3)}")
    
    for t in range(n_timepoints):
        for m in range(n_mutations):
            # True VAF calculation  
            numerator = mutation_sizes_het[t, m] + 2 * mutation_sizes_hom[t, m]
            denominator = 2 * (N_w + mutation_sizes_het[t, m] + mutation_sizes_hom[t, m])
            true_vaf = numerator / denominator if denominator > 0 else 0
            
            # Clip VAF to valid range
            true_vaf = np.clip(true_vaf, 0.001, 0.999)
            
            # Sample sequencing depth (higher for later timepoints = better data)
            depth_mean_t = depth_mean + t * 20  # Increase depth over time
            depth = int(np.random.normal(depth_mean_t, depth_std))
            depth = max(depth, 50)  # Ensure minimum depth
            
            # Sample alternate allele count
            alt_count = binom.rvs(n=depth, p=true_vaf)
            
            AO[t, m] = alt_count
            DP[t, m] = depth
    
    # Print VAF ranges for diagnostics
    VAF_check = AO / DP
    print(f"  VAF ranges:")
    for m, mut_name in enumerate(mutation_names):
        vaf_range = VAF_check[:, m]
        print(f"    {mut_name}: [{vaf_range.min():.3f}, {vaf_range.max():.3f}]")
        if vaf_range.max() < 0.01:
            print(f"      ⚠️  WARNING: Very low VAFs - may cause numerical issues")
    
    # Plot VAFs immediately if requested
    if plot_during_generation:
        print("\n  📊 Plotting VAFs during generation...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: VAF trajectories
        ax = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_mutations))
        for m, mut_name in enumerate(mutation_names):
            vaf_trajectory = VAF_check[:, m]
            ax.plot(time_points, vaf_trajectory, 'o-', 
                   label=mut_name, color=colors[m], 
                   linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('VAF', fontsize=12, fontweight='bold')
        ax.set_title('Generated VAF Trajectories', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Plot 2: Clone sizes
        ax = axes[1]
        for c in range(n_clones):
            ax.plot(time_points, clone_sizes[:, c], 's-',
                   label=f'Clone {c} (s={fitness_values[c]:.2f})',
                   linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clone Size (cells)', fontsize=12, fontweight='bold')
        ax.set_title('True Clone Sizes', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('generation_vaf_plot.png', dpi=150, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        print(f"    ✓ Saved: generation_vaf_plot.png")
        print(f"    ✓ Plot displayed (close to continue)")
        plt.close()
    
    # Package results
    ground_truth = {
        'clonal_structure': clonal_structure,
        'fitness_values': fitness_values,
        'clone_sizes': clone_sizes,
        'mutation_sizes_het': mutation_sizes_het,
        'mutation_sizes_hom': mutation_sizes_hom,
        'mutation_zygosity': mutation_zygosity,
        'mutation_names': mutation_names,  # Already created above
        'total_cells': total_cells
    }
    
    data = {
        'AO': AO,
        'DP': DP,
        'time_points': time_points,
        'ground_truth': ground_truth
    }
    
    return data


def create_anndata_from_synthetic(synthetic_data):
    """
    Convert synthetic data to AnnData-like structure expected by the inference script.
    
    Returns a mock object with the same interface as AnnData
    """
    class MockAnnData:
        def __init__(self, AO, DP, time_points, mutation_names):
            # IMPORTANT: X should be (n_mutations × n_timepoints) for compatibility
            # Your inference code expects mutations as rows, timepoints as columns
            VAF = AO / DP
            self.X = VAF.T  # Transpose! Should be (n_mutations, n_timepoints)
            
            # Observations (mutations) - one row per mutation
            self.obs = pd.DataFrame({
                'p_key': mutation_names
            }, index=mutation_names)
            
            # Variables (timepoints) - one row per timepoint
            self.var = pd.DataFrame({
                'time_points': time_points
            }, index=[f'T{i}' for i in range(len(time_points))])
            
            # Layers for raw counts - also (n_mutations × n_timepoints)
            self.layers = {
                'AO': pd.DataFrame(AO.T, index=mutation_names, 
                                  columns=[f'T{i}' for i in range(len(time_points))]),
                'DP': pd.DataFrame(DP.T, index=mutation_names,
                                  columns=[f'T{i}' for i in range(len(time_points))])
            }
            
            # Unstructured data
            self.uns = {}
            
            # Shape as property (not method!) - (n_mutations, n_timepoints)
            self.shape = self.X.shape
            
        def __getitem__(self, idx):
            # Simple indexing support
            new_obj = MockAnnData.__new__(MockAnnData)
            if isinstance(idx, int):
                new_obj.X = self.X[[idx], :]
                new_obj.obs = self.obs.iloc[[idx]]
            else:
                new_obj.X = self.X
                new_obj.obs = self.obs
            new_obj.var = self.var
            new_obj.layers = self.layers
            new_obj.uns = self.uns
            new_obj.shape = new_obj.X.shape
            return new_obj
    
    gt = synthetic_data['ground_truth']
    
    part = MockAnnData(
        synthetic_data['AO'],
        synthetic_data['DP'],
        synthetic_data['time_points'],
        gt['mutation_names']
    )
    
    return part


def plot_synthetic_data(synthetic_data, save_path=None):
    """
    Visualize the synthetic data
    """
    AO = synthetic_data['AO']
    DP = synthetic_data['DP']
    time_points = synthetic_data['time_points']
    gt = synthetic_data['ground_truth']
    
    VAF = AO / DP
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: VAFs over time
    ax = axes[0, 0]
    for m, mut_name in enumerate(gt['mutation_names']):
        clone_idx = [i for i, c in enumerate(gt['clonal_structure']) if m in c][0]
        ax.plot(time_points, VAF[:, m], marker='o', label=mut_name, 
                alpha=0.7, linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('VAF', fontsize=12)
    ax.set_title('Variant Allele Frequencies Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Clone sizes over time
    ax = axes[0, 1]
    for c in range(len(gt['clonal_structure'])):
        ax.plot(time_points, gt['clone_sizes'][:, c], marker='s', 
                label=f"Clone {c} (s={gt['fitness_values'][c]:.3f})",
                linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Clone Size', fontsize=12)
    ax.set_title('True Clone Sizes Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sequencing depth
    ax = axes[1, 0]
    depth_df = pd.DataFrame(DP, columns=gt['mutation_names'])
    depth_df.boxplot(ax=ax)
    ax.set_xlabel('Mutation', fontsize=12)
    ax.set_ylabel('Sequencing Depth', fontsize=12)
    ax.set_title('Sequencing Depth Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: VAF heatmap
    ax = axes[1, 1]
    sns.heatmap(VAF.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=[f'T{i}' for i in range(len(time_points))],
                yticklabels=gt['mutation_names'], ax=ax, cbar_kws={'label': 'VAF'})
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Mutation', fontsize=12)
    ax.set_title('VAF Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def print_summary(synthetic_data):
    """Print summary of synthetic data"""
    gt = synthetic_data['ground_truth']
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA SUMMARY")
    print("="*60)
    print(f"\nClonal Structure: {gt['clonal_structure']}")
    print(f"Fitness Values: {gt['fitness_values']}")
    print(f"Mutation Names: {gt['mutation_names']}")
    print(f"Zygosity: {dict(zip(gt['mutation_names'], gt['mutation_zygosity']))}")
    print(f"\nNumber of timepoints: {len(synthetic_data['time_points'])}")
    print(f"Timepoints: {synthetic_data['time_points']}")
    print(f"\nData shapes:")
    print(f"  AO: {synthetic_data['AO'].shape}")
    print(f"  DP: {synthetic_data['DP'].shape}")
    print(f"\nVAF ranges:")
    VAF = synthetic_data['AO'] / synthetic_data['DP']
    for m, mut_name in enumerate(gt['mutation_names']):
        print(f"  {mut_name}: [{VAF[:, m].min():.3f}, {VAF[:, m].max():.3f}]")


if __name__ == "__main__":
    # Example 1: Simple 2-clone scenario
    print("Generating Example 1: 2 clones, 2 mutations each")
    data1 = generate_synthetic_clonal_data(
        n_clones=2,
        n_mutations_per_clone=2,
        n_timepoints=5,
        fitness_values=[0.3, 0.6],
        seed=42
    )
    print_summary(data1)
    fig1 = plot_synthetic_data(data1, save_path='synthetic_data_example1.png')
    
    # Save to CSV
    pd.DataFrame(data1['AO']).to_csv('AO_example1.csv', index=False)
    pd.DataFrame(data1['DP']).to_csv('DP_example1.csv', index=False)
    
    # Example 2: More complex scenario
    print("\n\nGenerating Example 2: 3 clones, varying mutations")
    data2 = generate_synthetic_clonal_data(
        n_clones=3,
        n_mutations_per_clone=[2, 1, 3],
        n_timepoints=6,
        time_points=np.array([0, 2, 4, 6, 8, 10]),
        fitness_values=[0.2, 0.5, 0.8],
        seed=123
    )
    print_summary(data2)
    fig2 = plot_synthetic_data(data2, save_path='synthetic_data_example2.png')
    
    # Save to CSV
    pd.DataFrame(data2['AO']).to_csv('AO_example2.csv', index=False)
    pd.DataFrame(data2['DP']).to_csv('DP_example2.csv', index=False)
    
    print("\n\nSynthetic data generation complete!")
    print("Files saved:")
    print("  - AO_example1.csv, DP_example1.csv")
    print("  - AO_example2.csv, DP_example2.csv")
    print("  - synthetic_data_example1.png")
    print("  - synthetic_data_example2.png")