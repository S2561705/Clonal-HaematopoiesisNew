import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.KI_clonal_inference_3 import *
import pickle as pk
import numpy as np
import traceback

def validate_participant_data(part, index):
    """Validate that participant data has required structure."""
    print(f"\n{'='*60}")
    print(f"VALIDATING PARTICIPANT {index+1}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Participant ID: {part.uns.get('participant_id', 'Unknown')}")
    print(f"Number of mutations: {part.n_obs}")
    print(f"Number of timepoints: {part.n_vars}")
    
    # Check layers
    print(f"\nAvailable layers: {list(part.layers.keys())}")
    
    required_layers = ['AO', 'DP']
    for layer_name in required_layers:
        if layer_name not in part.layers:
            raise ValueError(f"Missing required layer: {layer_name}")
        
        layer = part.layers[layer_name]
        print(f"\n{layer_name} layer:")
        print(f"  Type: {type(layer)}")
        print(f"  Shape: {layer.shape}")
        print(f"  Dtype: {layer.dtype}")
        print(f"  Expected shape: ({part.n_obs}, {part.n_vars})")
        
        # Validate shape
        if layer.shape != (part.n_obs, part.n_vars):
            raise ValueError(
                f"{layer_name} layer shape {layer.shape} doesn't match "
                f"expected ({part.n_obs}, {part.n_vars})"
            )
        
        # Check for valid values
        if layer_name == 'AO':
            print(f"  AO range: [{layer.min():.1f}, {layer.max():.1f}]")
        elif layer_name == 'DP':
            print(f"  DP range: [{layer.min():.1f}, {layer.max():.1f}]")
            if np.any(layer <= 0):
                print(f"  WARNING: Found {np.sum(layer <= 0)} zero/negative DP values")
    
    # Check var
    if 'time_points' not in part.var.columns:
        raise ValueError("Missing 'time_points' column in var")
    
    print(f"\nTime points: {part.var.time_points.values}")
    
    # Test JAX conversion
    import jax.numpy as jnp
    try:
        test_ao = jnp.array(part.layers['AO'].T)
        test_dp = jnp.array(part.layers['DP'].T)
        print(f"\n✅ Successfully converted to JAX arrays:")
        print(f"   AO transposed shape: {test_ao.shape}")
        print(f"   DP transposed shape: {test_dp.shape}")
    except Exception as e:
        raise ValueError(f"Failed to convert layers to JAX arrays: {e}")
    
    return True


def run_inference_on_participant(part, index, s_resolution=50, 
                                 refine_resolution=201, 
                                 min_s=0.01, max_s=3,
                                 filter_invalid=True,
                                 grid_resolution=600):
    """Run complete clonal inference pipeline on a single participant."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING INFERENCE ON PARTICIPANT {index+1}")
    print(f"{'='*60}")
    
    # Step 1: Validate data
    try:
        validate_participant_data(part, index)
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        raise
    
    # Step 2: Run clonal model probability computation
    print(f"\n{'*'*60}")
    print("STEP 1: Computing clonal model probabilities")
    print(f"{'*'*60}")
    print(f"Parameters:")
    print(f"  s_resolution: {s_resolution}")
    print(f"  s_range: [{min_s}, {max_s}]")
    print(f"  grid_resolution: {grid_resolution}")
    print(f"  filter_invalid: {filter_invalid}")
    
    try:
        part = compute_clonal_models_prob_vec_mixed(
            part, 
            s_resolution=s_resolution,
            min_s=min_s,
            max_s=max_s,
            filter_invalid=filter_invalid,
            resolution=grid_resolution
        )
        
        # Display top models
        print(f"\n✅ Successfully computed {len(part.uns['model_dict'])} models")
        print(f"\nTop 5 models:")
        for i, (k, v) in enumerate(list(part.uns['model_dict'].items())[:5]):
            cs, prob = v
            print(f"  {i+1}. {k}: prob={prob:.3e}, structure={cs}")
            
    except Exception as e:
        print(f"❌ MODEL COMPUTATION FAILED: {e}")
        traceback.print_exc()
        raise
    
    # Step 3: Refine optimal model posterior
    print(f"\n{'*'*60}")
    print("STEP 2: Refining optimal model posterior")
    print(f"{'*'*60}")
    print(f"Parameters:")
    print(f"  s_resolution: {refine_resolution}")
    
    try:
        part = refine_optimal_model_posterior_vec(
            part, 
            s_resolution=refine_resolution
        )
        
        # Display results
        print(f"\n✅ Successfully refined optimal model")
        print(f"\nOptimal model structure:")
        opt_model = part.uns['optimal_model']
        print(f"  Clonal structure: {opt_model['clonal_structure']}")
        print(f"  Mutation structure: {opt_model['mutation_structure']}")
        
        if 'warning' in part.uns and part.uns['warning'] is not None:
            print(f"\n⚠️  WARNING: {part.uns['warning']}")
        
        print(f"\nFitness estimates:")
        print(part.obs[['fitness', 'fitness_5', 'fitness_95', 'clonal_index']])
        
    except Exception as e:
        print(f"❌ REFINEMENT FAILED: {e}")
        traceback.print_exc()
        raise
    
    return part


def main():
    """Main inference pipeline."""
    
    # Configuration
    INPUT_FILE = '../exports/MDS/MDS_cohort_processed.pk'
    OUTPUT_FILE = '../exports/MDS/MDS_cohort_fitted.pk'
    
    # Inference parameters
    S_RESOLUTION = 50          # Number of s values for initial model comparison
    REFINE_RESOLUTION = 201    # Number of s values for refined posterior
    MIN_S = 0.01              # Minimum fitness value
    MAX_S = 3.0               # Maximum fitness value
    GRID_RESOLUTION = 600     # Number of grid points for het/hom sampling
    FILTER_INVALID = True     # Filter invalid clonal structures
    
    print(f"{'='*60}")
    print("MDS COHORT CLONAL INFERENCE PIPELINE")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Initial s_resolution: {S_RESOLUTION}")
    print(f"  Refined s_resolution: {REFINE_RESOLUTION}")
    print(f"  Fitness range: [{MIN_S}, {MAX_S}]")
    print(f"  Grid resolution: {GRID_RESOLUTION}")
    print(f"  Filter invalid structures: {FILTER_INVALID}")
    
    # Load data
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    try:
        with open(INPUT_FILE, 'rb') as f:
            MDS_cohort = pk.load(f)
        print(f"✅ Loaded {len(MDS_cohort)} participants")
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
        return
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        traceback.print_exc()
        return
    
    # Process each participant
    processed_part_list = []
    success_count = 0
    error_count = 0
    
    for i, part in enumerate(MDS_cohort):
        try:
            processed_part = run_inference_on_participant(
                part, 
                i,
                s_resolution=S_RESOLUTION,
                refine_resolution=REFINE_RESOLUTION,
                min_s=MIN_S,
                max_s=MAX_S,
                filter_invalid=FILTER_INVALID,
                grid_resolution=GRID_RESOLUTION
            )
            
            processed_part_list.append(processed_part)
            success_count += 1
            print(f"\n✅ PARTICIPANT {i+1} COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            print(f"\n❌ PARTICIPANT {i+1} FAILED: {e}")
            error_count += 1
            continue
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(MDS_cohort)} participants")
    print(f"Errors: {error_count}")
    
    if processed_part_list:
        try:
            with open(OUTPUT_FILE, 'wb') as f:
                pk.dump(processed_part_list, f, protocol=4)
            print(f"\n✅ Results saved to {OUTPUT_FILE}")
            
            # Verify save
            with open(OUTPUT_FILE, 'rb') as f:
                verified = pk.load(f)
            print(f"✅ Verified: saved {len(verified)} participants")
            
        except Exception as e:
            print(f"❌ ERROR saving results: {e}")
            traceback.print_exc()
    else:
        print("\n❌ No participants were successfully processed. Output file not created.")
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total participants: {len(MDS_cohort)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Success rate: {100*success_count/len(MDS_cohort):.1f}%")


if __name__ == "__main__":
    main()