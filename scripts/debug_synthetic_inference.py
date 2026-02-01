"""
Debug script for testing clonal inference on synthetic data.
Runs the full pipeline on synthetic patients and compares results to ground truth.
"""

import sys
sys.path.append("..")   # allow imports from project root

import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
import time
import traceback
from collections import defaultdict

# Try to import your inference modules
try:
    from src.hom_inference import (
        compute_clonal_models_prob_vec,
        refine_optimal_model_posterior_vec
    )
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import inference modules: {e}")
    INFERENCE_AVAILABLE = False

# Try to import het inference if exists
try:
    from src.het_inference import (
        compute_clonal_models_prob_vec as compute_clonal_models_prob_vec_het,
        refine_optimal_model_posterior_vec as refine_optimal_model_posterior_vec_het
    )
    HET_INFERENCE_AVAILABLE = True
except ImportError:
    HET_INFERENCE_AVAILABLE = False


class SyntheticDataDebugger:
    """
    Debug and validate clonal inference on synthetic data.
    """
    
    def __init__(self, synthetic_data_path="../exports/synthetic_test_cohort.pk"):
        """Load synthetic test data."""
        print("Loading synthetic test data...")
        with open(synthetic_data_path, 'rb') as f:
            self.test_cohort = pk.load(f)
        
        print(f"Loaded {len(self.test_cohort)} synthetic patients")
        self.results = []
    
    def analyze_synthetic_patient(self, patient, patient_idx):
        """Analyze a synthetic patient's ground truth."""
        analysis = {
            'patient_id': patient.uns['patient_id'],
            'n_mutations': patient.shape[0],
            'n_clones': patient.uns['n_clones'],
            'scenario': patient.uns['scenario'],
            'mutations_per_clone': patient.uns['mutations_per_clone'],
            'true_clonal_structure': patient.uns['true_clonal_structure'],
            'zygosity_distribution': defaultdict(int)
        }
        
        # Analyze zygosity
        zygosity_counts = patient.obs['zygosity'].value_counts()
        analysis['zygosity_distribution'] = dict(zygosity_counts)
        
        # Extract true parameters
        true_params = []
        for i, param_dict in enumerate(patient.uns['true_parameters']):
            true_params.append({
                'mutation_idx': i,
                'mutation_id': patient.obs.iloc[i]['mutation_id'],
                'true_fitness': param_dict['true_fitness'],
                'true_H0': param_dict['true_H0'],
                'true_M0': param_dict['true_M0'],
                'true_zygosity': 'hom' if param_dict['true_M0'] > 0 else 'het',
                'clonal_group': patient.obs.iloc[i]['clonal_group']
            })
        
        analysis['true_parameters'] = true_params
        
        # Calculate VAF statistics
        vaf_stats = []
        AO = patient.layers['AO']
        DP = patient.layers['DP']
        
        for i in range(patient.shape[0]):
            vaf = AO[i] / DP[i]
            vaf_stats.append({
                'mean_vaf': np.mean(vaf),
                'std_vaf': np.std(vaf),
                'max_vaf': np.max(vaf),
                'min_vaf': np.min(vaf),
                'vaf_trajectory': vaf.tolist()
            })
        
        analysis['vaf_statistics'] = vaf_stats
        
        return analysis
    
    def run_hom_inference(self, patient, s_resolution=50):
        """Run homozygous inference pipeline."""
        if not INFERENCE_AVAILABLE:
            return None
        
        print(f"\n  Running hom_inference (s_resolution={s_resolution})...")
        
        # Make a copy to avoid modifying original
        patient_inf = patient.copy()
        
        try:
            # Time the inference
            start_time = time.time()
            
            # Step 1: Clonal landscape inference
            patient_inf = compute_clonal_models_prob_vec(
                patient_inf, 
                s_resolution=s_resolution,
                filter_invalid=True
            )
            step1_time = time.time() - start_time
            
            # Step 2: Refine optimal model
            start_time = time.time()
            patient_inf = refine_optimal_model_posterior_vec(
                patient_inf,
                s_resolution=min(100, s_resolution * 2)  # Higher resolution for refinement
            )
            step2_time = time.time() - start_time
            
            total_time = step1_time + step2_time
            
            # Extract results
            results = {
                'success': True,
                'total_time': total_time,
                'step1_time': step1_time,
                'step2_time': step2_time,
                'inferred_structure': None,
                'inferred_fitness': {},
                'model_probabilities': {}
            }
            
            # Get inferred clonal structure
            if 'model_dict' in patient_inf.uns and patient_inf.uns['model_dict']:
                # Get top model
                top_model_key = list(patient_inf.uns['model_dict'].keys())[0]
                results['inferred_structure'] = patient_inf.uns['model_dict'][top_model_key][0]
                
                # Get all model probabilities
                for model_key, (cs, prob) in patient_inf.uns['model_dict'].items():
                    results['model_probabilities'][model_key] = {
                        'structure': cs,
                        'probability': prob
                    }
            
            # Get inferred fitness values
            if 'fitness' in patient_inf.obs.columns:
                for i, row in patient_inf.obs.iterrows():
                    results['inferred_fitness'][i] = {
                        'fitness': row.get('fitness', np.nan),
                        'fitness_5': row.get('fitness_5', np.nan),
                        'fitness_95': row.get('fitness_95', np.nan),
                        'clonal_index': row.get('clonal_index', np.nan)
                    }
            
            return results
            
        except Exception as e:
            print(f"    ERROR in hom_inference: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'total_time': 0
            }
    
    def run_het_inference(self, patient, s_resolution=50):
        """Run heterozygous inference pipeline if available."""
        if not HET_INFERENCE_AVAILABLE:
            return None
        
        print(f"\n  Running het_inference (s_resolution={s_resolution})...")
        
        patient_inf = patient.copy()
        
        try:
            start_time = time.time()
            
            patient_inf = compute_clonal_models_prob_vec_het(
                patient_inf,
                s_resolution=s_resolution,
                filter_invalid=True
            )
            
            patient_inf = refine_optimal_model_posterior_vec_het(
                patient_inf,
                s_resolution=min(100, s_resolution * 2)
            )
            
            total_time = time.time() - start_time
            
            results = {
                'success': True,
                'total_time': total_time,
                'inferred_structure': None
            }
            
            if 'model_dict' in patient_inf.uns and patient_inf.uns['model_dict']:
                top_model_key = list(patient_inf.uns['model_dict'].keys())[0]
                results['inferred_structure'] = patient_inf.uns['model_dict'][top_model_key][0]
            
            return results
            
        except Exception as e:
            print(f"    ERROR in het_inference: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time': 0
            }
    
    def evaluate_inference_accuracy(self, ground_truth, inference_results):
        """Evaluate how well inference matches ground truth."""
        if not inference_results or not inference_results['success']:
            return {
                'structure_correct': False,
                'fitness_correlation': 0,
                'clustering_accuracy': 0,
                'zygosity_accuracy': 0
            }
        
        evaluation = {}
        
        # 1. Evaluate clonal structure recovery
        true_structure = ground_truth['true_clonal_structure']
        inferred_structure = inference_results.get('inferred_structure')
        
        if inferred_structure:
            # Convert to comparable format
            true_sets = [set(clone) for clone in true_structure]
            inferred_sets = [set(clone) for clone in inferred_structure]
            
            # Calculate Adjusted Rand Index (simple version)
            evaluation['structure_correct'] = (true_sets == inferred_sets)
            evaluation['n_true_clones'] = len(true_sets)
            evaluation['n_inferred_clones'] = len(inferred_sets)
            
            # Calculate clustering similarity
            evaluation['clustering_accuracy'] = self._calculate_clustering_accuracy(
                true_sets, inferred_sets
            )
        else:
            evaluation['structure_correct'] = False
            evaluation['clustering_accuracy'] = 0
        
        # 2. Evaluate fitness inference
        if 'inferred_fitness' in inference_results:
            true_fitness = [p['true_fitness'] for p in ground_truth['true_parameters']]
            inferred_fitness = []
            
            for i in range(len(true_fitness)):
                if i in inference_results['inferred_fitness']:
                    inf_fit = inference_results['inferred_fitness'][i]['fitness']
                    if not np.isnan(inf_fit):
                        inferred_fitness.append(inf_fit)
                    else:
                        inferred_fitness.append(0)
                else:
                    inferred_fitness.append(0)
            
            if len(inferred_fitness) == len(true_fitness):
                correlation = np.corrcoef(true_fitness, inferred_fitness)[0, 1]
                evaluation['fitness_correlation'] = correlation if not np.isnan(correlation) else 0
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((np.array(true_fitness) - np.array(inferred_fitness)) ** 2))
                evaluation['fitness_rmse'] = rmse
            else:
                evaluation['fitness_correlation'] = 0
                evaluation['fitness_rmse'] = np.nan
        
        return evaluation
    
    def _calculate_clustering_accuracy(self, true_sets, inferred_sets):
        """Calculate accuracy of mutation clustering."""
        n_mutations = sum(len(s) for s in true_sets)
        
        # Create true cluster labels
        true_labels = {}
        for cluster_idx, cluster in enumerate(true_sets):
            for mut in cluster:
                true_labels[mut] = cluster_idx
        
        # Create inferred cluster labels
        inferred_labels = {}
        for cluster_idx, cluster in enumerate(inferred_sets):
            for mut in cluster:
                inferred_labels[mut] = cluster_idx
        
        # Calculate agreement
        correct = 0
        total_pairs = 0
        
        # Check all pairs of mutations
        for i in range(n_mutations):
            for j in range(i + 1, n_mutations):
                # Are they in same cluster in true structure?
                same_in_true = (true_labels.get(i) == true_labels.get(j))
                same_in_inferred = (inferred_labels.get(i) == inferred_labels.get(j))
                
                if same_in_true == same_in_inferred:
                    correct += 1
                total_pairs += 1
        
        return correct / total_pairs if total_pairs > 0 else 0
    
    def run_comprehensive_test(self, s_resolutions=[20, 50, 100]):
        """Run comprehensive tests on all synthetic patients."""
        print("\n" + "="*80)
        print("COMPREHENSIVE SYNTHETIC DATA DEBUG SESSION")
        print("="*80)
        
        all_results = []
        
        for patient_idx, patient in enumerate(self.test_cohort):
            print(f"\n{'='*60}")
            print(f"PATIENT {patient_idx}: {patient.uns['patient_id']}")
            print(f"{'='*60}")
            
            # 1. Analyze ground truth
            ground_truth = self.analyze_synthetic_patient(patient, patient_idx)
            
            print(f"  Ground Truth:")
            print(f"    Mutations: {ground_truth['n_mutations']}, Clones: {ground_truth['n_clones']}")
            print(f"    Scenario: {ground_truth['scenario']}")
            print(f"    Zygosity: {dict(ground_truth['zygosity_distribution'])}")
            print(f"    True structure: {ground_truth['true_clonal_structure']}")
            
            # Show VAF patterns for first few mutations
            print(f"\n  VAF Patterns (first 3 mutations):")
            for i in range(min(3, ground_truth['n_mutations'])):
                vaf_stats = ground_truth['vaf_statistics'][i]
                print(f"    Mutation {i}: mean VAF = {vaf_stats['mean_vaf']:.3f}")
            
            patient_results = {
                'patient_id': ground_truth['patient_id'],
                'ground_truth': ground_truth,
                'inference_results': {},
                'evaluations': {}
            }
            
            # 2. Run inference with different resolutions
            for s_res in s_resolutions:
                print(f"\n  Testing with s_resolution = {s_res}")
                
                # Run homozygous inference
                hom_results = self.run_hom_inference(patient, s_res)
                if hom_results:
                    patient_results['inference_results'][f'hom_sres{s_res}'] = hom_results
                    
                    # Evaluate accuracy
                    evaluation = self.evaluate_inference_accuracy(ground_truth, hom_results)
                    patient_results['evaluations'][f'hom_sres{s_res}'] = evaluation
                    
                    print(f"    Homozygous inference: {'SUCCESS' if hom_results['success'] else 'FAILED'}")
                    if hom_results['success']:
                        print(f"      Time: {hom_results['total_time']:.2f}s")
                        print(f"      Inferred structure: {hom_results.get('inferred_structure')}")
                        print(f"      Structure correct: {evaluation.get('structure_correct', False)}")
                        if 'fitness_correlation' in evaluation:
                            print(f"      Fitness correlation: {evaluation['fitness_correlation']:.3f}")
                
                # Run heterozygous inference if available
                het_results = self.run_het_inference(patient, s_res)
                if het_results:
                    patient_results['inference_results'][f'het_sres{s_res}'] = het_results
                    evaluation = self.evaluate_inference_accuracy(ground_truth, het_results)
                    patient_results['evaluations'][f'het_sres{s_res}'] = evaluation
                    
                    print(f"    Heterozygous inference: {'SUCCESS' if het_results['success'] else 'FAILED'}")
                    if het_results['success']:
                        print(f"      Time: {het_results['total_time']:.2f}s")
                        print(f"      Inferred structure: {het_results.get('inferred_structure')}")
            
            all_results.append(patient_results)
            
            # Save intermediate results
            with open(f'../exports/debug_results_patient_{patient_idx}.pk', 'wb') as f:
                pk.dump(patient_results, f)
        
        # 3. Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """Generate comprehensive summary report."""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        summary_data = []
        
        for result in all_results:
            patient_id = result['patient_id']
            ground_truth = result['ground_truth']
            
            for method in ['hom_sres50', 'het_sres50']:  # Focus on default resolution
                if method in result['evaluations']:
                    eval_data = result['evaluations'][method]
                    inf_data = result['inference_results'][method]
                    
                    summary_data.append({
                        'patient_id': patient_id,
                        'n_mutations': ground_truth['n_mutations'],
                        'n_clones': ground_truth['n_clones'],
                        'scenario': ground_truth['scenario'],
                        'method': method,
                        'success': inf_data.get('success', False),
                        'time_seconds': inf_data.get('total_time', 0),
                        'structure_correct': eval_data.get('structure_correct', False),
                        'clustering_accuracy': eval_data.get('clustering_accuracy', 0),
                        'fitness_correlation': eval_data.get('fitness_correlation', 0),
                        'fitness_rmse': eval_data.get('fitness_rmse', np.nan)
                    })
        
        # Create summary DataFrame
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            print("\nOverall Performance Summary:")
            print("-" * 40)
            
            # Group by method
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                print(f"\n{method}:")
                print(f"  Success rate: {method_df['success'].mean():.1%} ({method_df['success'].sum()}/{len(method_df)})")
                print(f"  Avg time: {method_df['time_seconds'].mean():.2f}s")
                print(f"  Structure accuracy: {method_df['structure_correct'].mean():.1%}")
                print(f"  Clustering accuracy: {method_df['clustering_accuracy'].mean():.3f}")
                print(f"  Fitness correlation: {method_df['fitness_correlation'].mean():.3f}")
            
            # Group by scenario
            print("\n\nPerformance by Scenario:")
            print("-" * 40)
            
            for scenario in df['scenario'].unique():
                scenario_df = df[df['scenario'] == scenario]
                print(f"\n{scenario}:")
                print(f"  Success rate: {scenario_df['success'].mean():.1%}")
                print(f"  Structure accuracy: {scenario_df[scenario_df['success']]['structure_correct'].mean():.1%}")
            
            # Save detailed report
            report_path = "../exports/synthetic_debug_summary.csv"
            df.to_csv(report_path, index=False)
            print(f"\nDetailed report saved to: {report_path}")
            
            # Save full results
            with open('../exports/full_debug_results.pk', 'wb') as f:
                pk.dump(all_results, f)
            print(f"Full results saved to: ../exports/full_debug_results.pk")
        
        else:
            print("No valid inference results to summarize.")
    
    def visualize_results(self, all_results):
        """Create visualizations of inference results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract data for visualization
            viz_data = []
            for result in all_results:
                for method in result['evaluations']:
                    if result['evaluations'][method]:
                        viz_data.append({
                            'patient': result['patient_id'],
                            'method': method,
                            'n_mutations': result['ground_truth']['n_mutations'],
                            'clustering_accuracy': result['evaluations'][method].get('clustering_accuracy', 0),
                            'fitness_correlation': result['evaluations'][method].get('fitness_correlation', 0),
                            'time': result['inference_results'][method].get('total_time', 0)
                        })
            
            if not viz_data:
                return
            
            df_viz = pd.DataFrame(viz_data)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Clustering accuracy by mutation count
            sns.boxplot(data=df_viz, x='n_mutations', y='clustering_accuracy', 
                       hue='method', ax=axes[0, 0])
            axes[0, 0].set_title('Clustering Accuracy vs Mutation Count')
            axes[0, 0].set_ylabel('Clustering Accuracy')
            axes[0, 0].set_ylim(0, 1)
            
            # Plot 2: Fitness correlation by method
            sns.boxplot(data=df_viz, x='method', y='fitness_correlation', ax=axes[0, 1])
            axes[0, 1].set_title('Fitness Inference Correlation')
            axes[0, 1].set_ylabel('Correlation with True Fitness')
            axes[0, 1].set_ylim(-1, 1)
            
            # Plot 3: Computation time
            sns.scatterplot(data=df_viz, x='n_mutations', y='time', 
                          hue='method', size='time', ax=axes[1, 0])
            axes[1, 0].set_title('Computation Time vs Mutation Count')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_xlabel('Number of Mutations')
            
            # Plot 4: Success rate by scenario
            axes[1, 1].axis('off')  # Placeholder for more complex visualization
            
            plt.tight_layout()
            plt.savefig('../exports/debug_visualizations.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Visualization skipped (matplotlib/seaborn not available)")


def main():
    """Main debug function."""
    print("Starting synthetic data debug session...")
    
    # Create debugger
    debugger = SyntheticDataDebugger()
    
    # Run comprehensive tests
    all_results = debugger.run_comprehensive_test(s_resolutions=[20, 50])
    
    # Try to visualize results
    try:
        debugger.visualize_results(all_results)
    except:
        print("\nVisualization failed or skipped.")
    
    print("\n" + "="*80)
    print("DEBUG SESSION COMPLETE")
    print("="*80)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("1. Check ../exports/synthetic_debug_summary.csv for detailed results")
    print("2. Check individual patient results in ../exports/debug_results_patient_*.pk")
    print("3. Look for patterns in failure cases")
    print("4. Compare hom vs het inference performance")
    
    return all_results


if __name__ == "__main__":
    main()