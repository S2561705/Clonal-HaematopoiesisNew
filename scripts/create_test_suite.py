# create_temporal_emergence_simple.py
"""
Simple version focusing on homozygous mutations only.
"""

import numpy as np
import pandas as pd
import anndata as ad
import pickle as pk

class SimpleTemporalSimulator:
    """Simple simulator for temporal emergence (homozygous only)."""
    
    def __init__(self, N_w=1e5, seed=42):
        self.N_w = N_w
        self.rng = np.random.RandomState(seed)
        self.time_points = np.array([0, 3, 6, 9, 12, 18, 24])
        
        # Simple clone types
        self.clone_params = [
            {'fitness': 0.2, 'emergence': 0, 'initial': 1000},   # Early slow
            {'fitness': 0.5, 'emergence': 0, 'initial': 500},    # Early moderate
            {'fitness': 1.0, 'emergence': 6, 'initial': 50},     # Late aggressive
            {'fitness': 1.8, 'emergence': 12, 'initial': 10},    # Very late aggressive
        ]
    
    def create_simple_patient(self, patient_id, clone_indices, mutations_per_clone):
        """Create simple test patient."""
        all_mutations = []
        
        for clone_idx, (param_idx, n_muts) in enumerate(zip(clone_indices, mutations_per_clone)):
            params = self.clone_params[param_idx]
            
            for mut in range(n_muts):
                mutation_id = f"Clone{clone_idx}_Mut{mut}"
                
                # Add some variation within clone
                fitness = params['fitness'] * self.rng.uniform(0.9, 1.1)
                M0 = params['initial'] * self.rng.uniform(0.5, 1.5)
                
                # Simulate growth
                cell_counts = np.zeros(len(self.time_points))
                for t_idx, t in enumerate(self.time_points):
                    if t >= params['emergence']:
                        t_since = t - params['emergence']
                        cells = M0 * np.exp(fitness * t_since)
                        cells = min(cells, self.N_w * 0.8)  # Cap at 80%
                        cell_counts[t_idx] = cells
                
                # VAF
                vaf = cell_counts / (self.N_w + cell_counts)
                
                # Sequencing with realistic noise
                DP = self.rng.poisson([50, 60, 80, 100, 120, 150, 200])
                AO = np.zeros_like(DP, dtype=int)
                
                for i in range(len(self.time_points)):
                    if vaf[i] < 0.01:
                        AO[i] = 0  # Below detection
                    else:
                        noisy_vaf = vaf[i] + self.rng.normal(0, 0.08)
                        noisy_vaf = np.clip(noisy_vaf, 0.01, 0.99)
                        AO[i] = self.rng.binomial(DP[i], noisy_vaf)
                
                all_mutations.append({
                    'mutation_id': mutation_id,
                    'fitness': fitness,
                    'M0': M0,
                    'clonal_group': clone_idx,
                    'emergence': params['emergence'],
                    'true_vaf': vaf,
                    'AO': AO,
                    'DP': DP
                })
        
        # Convert to AnnData
        n_muts = len(all_mutations)
        n_timepoints = len(self.time_points)
        
        AO_matrix = np.zeros((n_timepoints, n_muts), dtype=int)
        DP_matrix = np.zeros((n_timepoints, n_muts), dtype=int)
        
        mutation_info = []
        
        for i, mut in enumerate(all_mutations):
            AO_matrix[:, i] = mut['AO']
            DP_matrix[:, i] = mut['DP']
            
            mutation_info.append({
                'mutation_id': mut['mutation_id'],
                'clonal_group': mut['clonal_group'],
                'emergence': mut['emergence']
            })
        
        obs = pd.DataFrame(mutation_info)
        var = pd.DataFrame({'time_points': self.time_points})
        
        adata = ad.AnnData(
            X=np.random.randn(n_muts, n_timepoints),
            obs=obs,
            var=var,
            layers={'AO': AO_matrix.T, 'DP': DP_matrix.T}
        )
        
        # Add true structure
        groups = {}
        for i, mut in enumerate(all_mutations):
            group = mut['clonal_group']
            if group not in groups:
                groups[group] = []
            groups[group].append(i)
        
        adata.uns['patient_id'] = patient_id
        adata.uns['true_clonal_structure'] = [groups[g] for g in sorted(groups.keys())]
        adata.uns['clone_indices'] = clone_indices
        adata.uns['clone_params'] = [self.clone_params[i] for i in clone_indices]
        
        return adata
    
    def create_test_suite(self):
        """Create simple test suite."""
        test_patients = []
        
        # Test 1: Early slow vs Late aggressive
        print("Creating: Early Slow vs Late Aggressive")
        test_patients.append(self.create_simple_patient(
            "EARLY_SLOW_vs_LATE_AGGRESSIVE",
            clone_indices=[0, 2],  # index 0 and 2 from clone_params
            mutations_per_clone=[3, 2]
        ))
        
        # Test 2: Three clones
        print("Creating: Three Clone Competition")
        test_patients.append(self.create_simple_patient(
            "THREE_CLONE_COMPETITION",
            clone_indices=[0, 1, 2],
            mutations_per_clone=[2, 2, 2]
        ))
        
        # Test 3: Very late takeover
        print("Creating: Very Late Takeover")
        test_patients.append(self.create_simple_patient(
            "VERY_LATE_TAKEOVER",
            clone_indices=[1, 3],  # Moderate early vs Very late aggressive
            mutations_per_clone=[4, 1]  # Late clone has fewer mutations
        ))
        
        return test_patients
    
    def visualize(self, test_suite):
        """Quick visualization."""
        try:
            import matplotlib.pyplot as plt
            
            for patient in test_suite:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot 1: VAF trajectories
                true_struct = patient.uns['true_clonal_structure']
                colors = ['red', 'blue', 'green', 'orange']
                
                for clone_idx, clone_muts in enumerate(true_struct):
                    for mut in clone_muts:
                        vaf = patient.layers['AO'][mut] / patient.layers['DP'][mut]
                        ax1.plot(self.time_points, vaf, 'o-', 
                                color=colors[clone_idx % len(colors)],
                                alpha=0.6,
                                label=f'Clone {clone_idx}' if mut == clone_muts[0] else "")
                        
                        # Mark emergence
                        emergence = patient.uns['clone_params'][clone_idx]['emergence']
                        ax1.axvline(x=emergence, color=colors[clone_idx % len(colors)], 
                                   linestyle=':', alpha=0.3)
                
                ax1.set_title(f"{patient.uns['patient_id']}")
                ax1.set_xlabel('Time (months)')
                ax1.set_ylabel('VAF')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Summary
                ax2.axis('off')
                text = f"True structure: {patient.uns['true_clonal_structure']}\n\n"
                for clone_idx, params in enumerate(patient.uns['clone_params']):
                    text += f"Clone {clone_idx}:\n"
                    text += f"  Fitness: {params['fitness']:.2f}\n"
                    text += f"  Emerges: {params['emergence']} months\n"
                    text += f"  Initial: {params['initial']} cells\n\n"
                
                ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                plt.savefig(f'../exports/{patient.uns["patient_id"]}.png', 
                           dpi=150, bbox_inches='tight')
                plt.show()
                
        except ImportError:
            print("Matplotlib not available")


def main():
    """Main function."""
    print("Generating simple temporal emergence test suite...")
    
    simulator = SimpleTemporalSimulator()
    test_suite = simulator.create_test_suite()
    
    # Save
    with open('../exports/simple_temporal_test_cohort.pk', 'wb') as f:
        pk.dump(test_suite, f)
    
    print(f"\nSaved 3 patients to ../exports/simple_temporal_test_cohort.pk")
    
    # Summarize
    for i, patient in enumerate(test_suite):
        print(f"\nPatient {i}: {patient.uns['patient_id']}")
        print(f"  Mutations: {patient.shape[0]}")
        print(f"  Clones: {len(patient.uns['true_clonal_structure'])}")
        print(f"  True structure: {patient.uns['true_clonal_structure']}")
    
    # Visualize
    simulator.visualize(test_suite)
    
    return test_suite


if __name__ == "__main__":
    test_suite = main()