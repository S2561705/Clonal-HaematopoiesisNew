"""
Validation Pipeline for Multi-Clone Synthetic Data

Runs the full inference pipeline on synthetic multi-clone data and
compares results against ground truth.

Tests:
1. Clonal structure inference accuracy
2. Fitness (s) inference per clone
3. Zygosity (h) inference per clone
"""

import sys
sys.path.append("..")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import your inference functions
from src.KI_clonal_inference_2 import (
    compute_clonal_models_prob_vec_mixed,
    refine_optimal_model_posterior_vec
)

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path('../exports/synthetic_multiclone/')
COHORT_FILE = DATA_DIR / 'synthetic_multiclone_cohort.pk'
GROUND_TRUTH_FILE = DATA_DIR / 'synthetic_multiclone_ground_truth.csv'

OUTPUT_DIR = DATA_DIR / 'validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inference parameters
S_RESOLUTION = 50
H_RESOLUTION = 30
FILTER_INVALID = True

# ==============================================================================
# Helper Functions
# ==============================================================================

def compare_clonal_structures(inferred_cs, true_cs):
    """
    Compare inferred and true clonal structures.
    
    Two structures are considered equal if they represent the same partition,
    regardless of clone ordering.
    
    Parameters:
    -----------
    inferred_cs : list of lists
        Inferred clonal structure
    true_cs : list of lists
        True clonal structure
        
    Returns:
    --------
    is_correct : bool
        True if structures match
    jaccard_similarity : float
        Jaccard similarity between structures
    """
    
    # Convert to sets of frozensets for comparison
    inferred_set = {frozenset(clone) for clone in inferred_cs}
    true_set = {frozenset(clone) for clone in true_cs}
    
    # Exact match
    is_correct = inferred_set == true_set
    
    # Jaccard similarity
    intersection = len(inferred_set & true_set)
    union = len(inferred_set | true_set)
    jaccard = intersection / union if union > 0 else 0.0
    
    return is_correct, jaccard


def extract_clone_parameters(part, clone_idx, clonal_structure):
    """
    Extract inferred parameters for a specific clone.
    
    Parameters:
    -----------
    part : AnnData
        Participant with inference results in .obs
    clone_idx : int
        Clone index
    clonal_structure : list of lists
        Inferred clonal structure
        
    Returns:
    --------
    params : dict
        Inferred s, h for this clone
    """
    
    if clone_idx >= len(clonal_structure):
        return None
    
    # Get mutations in this clone
    clone_mutations = clonal_structure[clone_idx]
    
    if len(clone_mutations) == 0:
        return None
    
    # Get parameters from first mutation in clone
    # (all mutations in clone should have same parameters)
    mut_idx = clone_mutations[0]
    
    # Check if zygosity columns exist (they may not if using h-free inference)
    obs_cols = part.obs.columns
    
    result = {
        's': part.obs.iloc[mut_idx]['fitness'],
        's_5': part.obs.iloc[mut_idx].get('fitness_5', part.obs.iloc[mut_idx]['fitness']),
        's_95': part.obs.iloc[mut_idx].get('fitness_95', part.obs.iloc[mut_idx]['fitness']),
    }
    
    # Add h if it exists in the results
    if 'zygosity' in obs_cols:
        result['h'] = part.obs.iloc[mut_idx]['zygosity']
        result['h_5'] = part.obs.iloc[mut_idx].get('zygosity_5', part.obs.iloc[mut_idx]['zygosity'])
        result['h_95'] = part.obs.iloc[mut_idx].get('zygosity_95', part.obs.iloc[mut_idx]['zygosity'])
    elif 'h_vec' in part.uns.get('optimal_model', {}):
        # Try to get h from uns if not in obs
        h_vec = part.uns['optimal_model']['h_vec']
        if clone_idx < len(h_vec):
            result['h'] = h_vec[clone_idx]
            result['h_5'] = h_vec[clone_idx]
            result['h_95'] = h_vec[clone_idx]
        else:
            result['h'] = np.nan
            result['h_5'] = np.nan
            result['h_95'] = np.nan
    else:
        # No zygosity inference available
        print(f"⚠️  Warning: No zygosity data found for {part.uns['participant_id']}")
        print(f"   Available columns: {list(obs_cols)}")
        result['h'] = np.nan
        result['h_5'] = np.nan
        result['h_95'] = np.nan
    
    return result


def parse_clonal_structure(cs_str):
    """Parse clonal structure string from CSV."""
    import ast
    return ast.literal_eval(cs_str)


# ==============================================================================
# Main Validation Pipeline
# ==============================================================================

def run_inference_on_cohort(cohort):
    """
    Run inference on entire cohort.
    
    Parameters:
    -----------
    cohort : list of AnnData
        Synthetic participants
        
    Returns:
    --------
    cohort_with_results : list of AnnData
        Participants with inference results
    inference_summary : list of dict
        Summary of inference for each participant
    """
    
    print("="*80)
    print("RUNNING INFERENCE ON MULTI-CLONE SYNTHETIC COHORT")
    print("="*80)
    print(f"\nCohort size: {len(cohort)}")
    print(f"Inference parameters:")
    print(f"  s_resolution: {S_RESOLUTION}")
    print(f"  h_resolution: {H_RESOLUTION}")
    print(f"  filter_invalid: {FILTER_INVALID}")
    print()
    
    inference_summary = []
    
    for i, part in enumerate(tqdm(cohort, desc="Running inference")):
        participant_id = part.uns['participant_id']
        
        try:
            # Step 1: Compute clonal models (structure inference)
            part = compute_clonal_models_prob_vec_mixed(
                part,
                s_resolution=S_RESOLUTION,
                min_s=0.01,
                max_s=1.0,
                filter_invalid=FILTER_INVALID,
                disable_progressbar=True
            )
            
            # Step 2: Refine optimal model (s, h inference)
            part = refine_optimal_model_posterior_vec(
                part,
                s_resolution=100,  # Higher resolution for refinement
                h_resolution=H_RESOLUTION
            )
            
            # Extract results
            inferred_cs = part.uns['optimal_model']['clonal_structure']
            model_prob = list(part.uns['model_dict'].values())[0][1]
            
            inference_summary.append({
                'participant_id': participant_id,
                'inference_success': True,
                'inferred_clonal_structure': str(inferred_cs),
                'n_clones_inferred': len(inferred_cs),
                'n_mutations': part.shape[0],
                'model_probability': model_prob,
                'has_warning': part.uns.get('warning') is not None
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {participant_id}: {str(e)}")
            
            inference_summary.append({
                'participant_id': participant_id,
                'inference_success': False,
                'inferred_clonal_structure': None,
                'n_clones_inferred': None,
                'n_mutations': part.shape[0],
                'model_probability': None,
                'has_warning': True,
                'error': str(e)
            })
    
    print(f"\n✅ Inference complete: {sum(r['inference_success'] for r in inference_summary)}/{len(cohort)} successful")
    
    return cohort, inference_summary


def validate_against_ground_truth(cohort, inference_summary, ground_truth_df):
    """
    Compare inference results against ground truth.
    
    Parameters:
    -----------
    cohort : list of AnnData
        Participants with inference results
    inference_summary : list of dict
        Inference summary
    ground_truth_df : DataFrame
        Ground truth data
        
    Returns:
    --------
    validation_results : DataFrame
        Detailed comparison
    """
    
    print("\n" + "="*80)
    print("VALIDATING AGAINST GROUND TRUTH")
    print("="*80)
    
    # Diagnostic: Check what columns are available in first successful inference
    for part, inf_summary in zip(cohort, inference_summary):
        if inf_summary['inference_success']:
            print(f"\n📊 Available inference results for {part.uns['participant_id']}:")
            print(f"   .obs columns: {list(part.obs.columns)}")
            if 'optimal_model' in part.uns:
                print(f"   .uns['optimal_model'] keys: {list(part.uns['optimal_model'].keys())}")
            break
    print()
    
    validation_results = []
    
    for part, inf_summary in zip(cohort, inference_summary):
        participant_id = part.uns['participant_id']
        
        # Get ground truth
        gt = ground_truth_df[ground_truth_df['participant_id'] == participant_id].iloc[0]
        
        if not inf_summary['inference_success']:
            validation_results.append({
                'participant_id': participant_id,
                'inference_success': False,
                'structure_correct': False,
                'structure_jaccard': 0.0,
            })
            continue
        
        # Parse structures
        true_cs = parse_clonal_structure(gt['clonal_structure_str'])
        inferred_cs = parse_clonal_structure(inf_summary['inferred_clonal_structure'])
        
        # Compare structures
        structure_correct, jaccard = compare_clonal_structures(inferred_cs, true_cs)
        
        # Compare per-clone parameters
        n_clones_true = gt['n_clones']
        n_clones_inferred = len(inferred_cs)
        
        result = {
            'participant_id': participant_id,
            'inference_success': True,
            'n_mutations': gt['n_mutations'],
            'n_clones_true': n_clones_true,
            'n_clones_inferred': n_clones_inferred,
            'structure_correct': structure_correct,
            'structure_jaccard': jaccard,
            'true_clonal_structure': str(true_cs),
            'inferred_clonal_structure': str(inferred_cs),
        }
        
        # For each true clone, find best matching inferred clone
        for clone_idx in range(n_clones_true):
            true_clone_muts = set(true_cs[clone_idx])
            
            # Get true parameters
            s_true = gt[f'clone_{clone_idx}_s']
            h_true = gt[f'clone_{clone_idx}_h']
            
            # Find best matching inferred clone (by mutation overlap)
            best_overlap = 0
            best_inferred_params = None
            
            for inf_clone_idx, inf_clone_muts in enumerate(inferred_cs):
                overlap = len(true_clone_muts & set(inf_clone_muts))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_inferred_params = extract_clone_parameters(
                        part, inf_clone_idx, inferred_cs
                    )
            
            if best_inferred_params is not None:
                s_inferred = best_inferred_params['s']
                h_inferred = best_inferred_params['h']
                
                result[f'clone_{clone_idx}_s_true'] = s_true
                result[f'clone_{clone_idx}_s_inferred'] = s_inferred
                result[f'clone_{clone_idx}_s_error'] = abs(s_true - s_inferred)
                
                result[f'clone_{clone_idx}_h_true'] = h_true
                result[f'clone_{clone_idx}_h_inferred'] = h_inferred
                result[f'clone_{clone_idx}_h_error'] = abs(h_true - h_inferred)
            else:
                result[f'clone_{clone_idx}_s_true'] = s_true
                result[f'clone_{clone_idx}_h_true'] = h_true
                result[f'clone_{clone_idx}_s_inferred'] = np.nan
                result[f'clone_{clone_idx}_h_inferred'] = np.nan
                result[f'clone_{clone_idx}_s_error'] = np.nan
                result[f'clone_{clone_idx}_h_error'] = np.nan
        
        validation_results.append(result)
    
    validation_df = pd.DataFrame(validation_results)
    
    # Summary statistics
    print(f"\nStructure Inference:")
    print(f"  Correct: {validation_df['structure_correct'].sum()}/{len(validation_df)}")
    print(f"  Mean Jaccard similarity: {validation_df['structure_jaccard'].mean():.3f}")
    
    print(f"\nClone Count Accuracy:")
    clone_count_correct = (validation_df['n_clones_true'] == validation_df['n_clones_inferred']).sum()
    print(f"  Correct: {clone_count_correct}/{len(validation_df)}")
    
    # Parameter errors (aggregate across all clones)
    s_errors = []
    h_errors = []
    
    for col in validation_df.columns:
        if col.endswith('_s_error'):
            s_errors.extend(validation_df[col].dropna().values)
        if col.endswith('_h_error'):
            h_errors.extend(validation_df[col].dropna().values)
    
    if s_errors:
        print(f"\nFitness (s) Inference:")
        print(f"  Mean error: {np.mean(s_errors):.3f}")
        print(f"  Median error: {np.median(s_errors):.3f}")
    
    if h_errors:
        print(f"\nZygosity (h) Inference:")
        print(f"  Mean error: {np.mean(h_errors):.3f}")
        print(f"  Median error: {np.median(h_errors):.3f}")
    
    return validation_df


def plot_validation_results(validation_df, output_dir):
    """
    Create visualizations of validation results.
    """
    
    print("\n" + "="*80)
    print("GENERATING VALIDATION PLOTS")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Clonal structure accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    structure_acc = validation_df['structure_correct'].value_counts()
    colors = ['green' if idx else 'red' for idx in structure_acc.index]
    ax1.bar(['Incorrect', 'Correct'][:len(structure_acc)], structure_acc.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Clonal Structure Accuracy\n{structure_acc.get(True, 0)}/{len(validation_df)} correct', fontsize=12)
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. Jaccard similarity distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(validation_df['structure_jaccard'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(validation_df['structure_jaccard'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {validation_df["structure_jaccard"].mean():.3f}')
    ax2.set_xlabel('Jaccard Similarity', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Structure Similarity Distribution', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Clone count accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    valid_df = validation_df.dropna(subset=['n_clones_true', 'n_clones_inferred'])
    ax3.scatter(valid_df['n_clones_true'], valid_df['n_clones_inferred'], 
               s=100, alpha=0.6, edgecolor='black')
    ax3.plot([0, 4], [0, 4], 'k--', linewidth=2, label='Perfect')
    ax3.set_xlabel('True # Clones', fontsize=11)
    ax3.set_ylabel('Inferred # Clones', fontsize=11)
    ax3.set_title('Clone Count Inference', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xticks([1, 2, 3])
    ax3.set_yticks([1, 2, 3])
    
    # 4. Fitness errors (all clones aggregated)
    ax4 = fig.add_subplot(gs[1, 0])
    s_errors = []
    for col in validation_df.columns:
        if col.endswith('_s_error'):
            s_errors.extend(validation_df[col].dropna().values)
    
    if s_errors:
        ax4.hist(s_errors, bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax4.axvline(np.mean(s_errors), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(s_errors):.3f}')
        ax4.set_xlabel('Fitness (s) Error', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Fitness Inference Errors', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
    
    # 5. Zygosity errors (all clones aggregated)
    ax5 = fig.add_subplot(gs[1, 1])
    h_errors = []
    for col in validation_df.columns:
        if col.endswith('_h_error'):
            h_errors.extend(validation_df[col].dropna().values)
    
    if h_errors:
        ax5.hist(h_errors, bins=20, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax5.axvline(np.mean(h_errors), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(h_errors):.3f}')
        ax5.set_xlabel('Zygosity (h) Error', fontsize=11)
        ax5.set_ylabel('Count', fontsize=11)
        ax5.set_title('Zygosity Inference Errors', fontsize=12)
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""
VALIDATION SUMMARY
{'='*40}

Participants: {len(validation_df)}
Successful inference: {validation_df['inference_success'].sum()}

Clonal Structure:
  Exact match: {validation_df['structure_correct'].sum()}/{len(validation_df)}
  Mean Jaccard: {validation_df['structure_jaccard'].mean():.3f}

Clone Count:
  Correct: {(valid_df['n_clones_true'] == valid_df['n_clones_inferred']).sum()}/{len(valid_df)}

Fitness (s):
  Mean error: {np.mean(s_errors):.3f}
  Median error: {np.median(s_errors):.3f}
  <0.1 error: {sum(e < 0.1 for e in s_errors)}/{len(s_errors)}

Zygosity (h):
  Mean error: {np.mean(h_errors):.3f}
  Median error: {np.median(h_errors):.3f}
  <0.2 error: {sum(e < 0.2 for e in h_errors)}/{len(h_errors)}
"""
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Multi-Clone Inference Validation', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = output_dir / 'validation_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved validation plot: {output_file}")
    
    plt.close()


def plot_vaf_trajectories(cohort, ground_truth_df, validation_df, output_dir, n_examples=9):
    """
    Plot VAF trajectories colored by true vs inferred clones.
    
    Creates side-by-side comparisons showing:
    - Left: VAF colored by TRUE clone assignment
    - Right: VAF colored by INFERRED clone assignment
    
    Parameters:
    -----------
    cohort : list of AnnData
        Participants with inference results
    ground_truth_df : DataFrame
        Ground truth
    validation_df : DataFrame
        Validation results
    output_dir : Path
        Output directory
    n_examples : int
        Number of examples to plot
    """
    
    print(f"\n📊 Generating VAF trajectory comparisons for {n_examples} participants...")
    
    # Select diverse examples
    # Prioritize: correct structure, incorrect structure, different # clones
    correct_structures = validation_df[validation_df['structure_correct'] == True]
    incorrect_structures = validation_df[validation_df['structure_correct'] == False]
    
    n_correct = min(6, len(correct_structures))
    n_incorrect = min(3, len(incorrect_structures))
    
    example_indices = []
    
    if len(correct_structures) > 0:
        example_indices.extend(
            correct_structures.sample(n=n_correct, random_state=42).index.tolist()
        )
    
    if len(incorrect_structures) > 0:
        example_indices.extend(
            incorrect_structures.sample(n=n_incorrect, random_state=42).index.tolist()
        )
    
    # Pad with random if needed
    while len(example_indices) < n_examples and len(example_indices) < len(validation_df):
        remaining = validation_df.index.difference(example_indices)
        if len(remaining) > 0:
            example_indices.append(remaining[0])
        else:
            break
    
    example_indices = example_indices[:n_examples]
    
    # Color palette for clones
    clone_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure
    n_rows = int(np.ceil(len(example_indices) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for plot_idx, val_idx in enumerate(example_indices):
        val_row = validation_df.iloc[val_idx]
        participant_id = val_row['participant_id']
        
        # Get participant data
        part = [p for p in cohort if p.uns['participant_id'] == participant_id][0]
        gt = ground_truth_df[ground_truth_df['participant_id'] == participant_id].iloc[0]
        
        # Get structures
        true_cs = parse_clonal_structure(gt['clonal_structure_str'])
        inferred_cs = parse_clonal_structure(val_row['inferred_clonal_structure'])
        
        # Get VAF data
        AO = part.layers['AO']
        DP = part.layers['DP']
        VAF = AO / np.maximum(DP, 1)
        timepoints = part.var['time_points'].values
        
        n_mutations = part.shape[0]
        
        ax = axes[plot_idx]
        
        # Create two subplots side by side
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), 
                                     wspace=0.3, hspace=0)
        ax.remove()
        ax_true = fig.add_subplot(gs[0])
        ax_inferred = fig.add_subplot(gs[1])
        
        # LEFT PANEL: Color by TRUE clone
        for clone_idx, clone_muts in enumerate(true_cs):
            color = clone_colors[clone_idx % len(clone_colors)]
            for mut_idx in clone_muts:
                ax_true.plot(timepoints, VAF[mut_idx, :], 
                           'o-', color=color, alpha=0.7, linewidth=2,
                           markersize=6, label=f'Clone {clone_idx}' if mut_idx == clone_muts[0] else '')
        
        ax_true.set_xlabel('Time', fontsize=10)
        ax_true.set_ylabel('VAF', fontsize=10)
        ax_true.set_title('TRUE Clones', fontsize=11, fontweight='bold')
        ax_true.grid(alpha=0.3)
        ax_true.set_ylim(-0.05, 1.05)
        if len(true_cs) <= 3:
            ax_true.legend(fontsize=8, loc='upper left')
        
        # RIGHT PANEL: Color by INFERRED clone
        for clone_idx, clone_muts in enumerate(inferred_cs):
            color = clone_colors[clone_idx % len(clone_colors)]
            for mut_idx in clone_muts:
                ax_inferred.plot(timepoints, VAF[mut_idx, :],
                               'o-', color=color, alpha=0.7, linewidth=2,
                               markersize=6, label=f'Clone {clone_idx}' if mut_idx == clone_muts[0] else '')
        
        ax_inferred.set_xlabel('Time', fontsize=10)
        ax_inferred.set_ylabel('VAF', fontsize=10)
        ax_inferred.set_title('INFERRED Clones', fontsize=11, fontweight='bold')
        ax_inferred.grid(alpha=0.3)
        ax_inferred.set_ylim(-0.05, 1.05)
        if len(inferred_cs) <= 3:
            ax_inferred.legend(fontsize=8, loc='upper left')
        
        # Add super title for this participant
        structure_match = "✅ CORRECT" if val_row['structure_correct'] else "❌ INCORRECT"
        fig.text(0.11 + (plot_idx % 3) * 0.31, 
                0.98 - (plot_idx // 3) * (5/n_rows/5.5), 
                f"{participant_id} - {structure_match}\nTrue: {val_row['true_clonal_structure']}\nInf:  {val_row['inferred_clonal_structure']}",
                fontsize=9, ha='left', va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue' if val_row['structure_correct'] else 'lightcoral', alpha=0.3))
    
    # Hide unused axes
    for idx in range(len(example_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('VAF Trajectories: True vs Inferred Clonal Structure', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = output_dir / 'vaf_trajectories_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved VAF trajectory comparison: {output_file}")
    
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run full validation pipeline."""
    
    print("="*80)
    print("MULTI-CLONE SYNTHETIC DATA VALIDATION PIPELINE")
    print("="*80)
    
    # Load data
    print(f"\nLoading synthetic data from: {COHORT_FILE}")
    with open(COHORT_FILE, 'rb') as f:
        cohort = pickle.load(f)
    print(f"✅ Loaded {len(cohort)} participants")
    
    print(f"\nLoading ground truth from: {GROUND_TRUTH_FILE}")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    print(f"✅ Loaded ground truth for {len(ground_truth_df)} participants")
    
    # Run inference
    cohort_with_results, inference_summary = run_inference_on_cohort(cohort)
    
    # Save inference summary
    inference_summary_df = pd.DataFrame(inference_summary)
    inference_summary_df.to_csv(OUTPUT_DIR / 'inference_summary.csv', index=False)
    print(f"\n✅ Saved inference summary: {OUTPUT_DIR / 'inference_summary.csv'}")
    
    # Validate against ground truth
    validation_df = validate_against_ground_truth(
        cohort_with_results, 
        inference_summary, 
        ground_truth_df
    )
    
    # Save validation results
    validation_df.to_csv(OUTPUT_DIR / 'validation_detailed.csv', index=False)
    print(f"\n✅ Saved validation results: {OUTPUT_DIR / 'validation_detailed.csv'}")
    
    # Plot results
    plot_validation_results(validation_df, OUTPUT_DIR)
    
    # Plot VAF trajectories
    plot_vaf_trajectories(
        cohort_with_results, 
        ground_truth_df, 
        validation_df, 
        OUTPUT_DIR,
        n_examples=9
    )
    
    # Save processed cohort
    with open(OUTPUT_DIR / 'cohort_with_inference.pk', 'wb') as f:
        pickle.dump(cohort_with_results, f)
    print(f"✅ Saved cohort with inference: {OUTPUT_DIR / 'cohort_with_inference.pk'}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - inference_summary.csv: High-level summary")
    print(f"  - validation_detailed.csv: Detailed per-clone comparison")
    print(f"  - validation_results.png: Visualization")
    print(f"  - cohort_with_inference.pk: Processed data")


if __name__ == '__main__':
    main()