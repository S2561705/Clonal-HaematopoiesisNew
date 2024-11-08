import sys
sys.path.append("..")  # fix to import modules from root
from src.general_imports import *
import pandas as pd
import anndata as ad
import pickle as pk
from src.blood_markers_aux import *

def load_data():
    """
    Load and preprocess the data for analysis.

    Returns:
    tuple: Preprocessed data (AnnData object) and marker map
    """
    # Load participant data
    participant_df = pd.read_csv('../results/participant_df.csv', index_col=0)
    participant_df['participant_id'] = participant_df.apply(
        lambda row: row['cohort'] + row['participant_id'] if not row['cohort'].startswith('LBC') else row['participant_id'], 
        axis=1
    )
    duplicated_ids = participant_df[participant_df.duplicated(subset='participant_id', keep=False)]['participant_id'].unique()
    participant_df = participant_df.drop_duplicates(subset='participant_id', keep='first')

    # Load cohort data
    with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
        cohort = pk.load(f)

    # Preprocess cohort data
    new_cohort = []
    for part in cohort:
        if part.uns['cohort'] != 'LBC':
            part.uns['participant_id'] = part.uns['cohort'] + str(part.uns['participant_id'])
        participant_id = part.uns['participant_id']
        if participant_id in duplicated_ids:
            if not any(p.uns['participant_id'] == participant_id for p in new_cohort):
                new_cohort.append(part)
        else:
            new_cohort.append(part)
    cohort = new_cohort

    # Load longitudinal data
    long_lbc21 = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC21/LBC21_LongitudinalVariables.matrix.tsv', sep='\t')
    long_lbc36 = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC36/LBC1936_LongitudinalVariables.matrix.tsv', sep='\t')
    long_sardinia = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/SardiNIA/SardiNIA_LongitudinalVariables.matrix.tsv', sep='\t')
    long_WHI = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/WHI/WHI_LongitudinalVariables.matrix.tsv', sep='\t')
    long_WHI = long_WHI.rename(columns={'ID':'participant_id'})
    marker_map = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/cohort_param_values_map.txt', sep='\t')

    # Modify cohort-specific IDs
    long_sardinia['participant_id'] = 'sardiNIA' + long_sardinia['participant_id'].astype('str')
    long_WHI['participant_id'] = 'WHI' + long_WHI['participant_id'].astype('str')

    # Add missing global_key to LBC36
    long_lbc36['global_key'] = long_lbc36['parameter_key'].map(
        dict(zip(marker_map['LBC36'], marker_map['global_name']))
    )
    column_order = ['participant_id', 'global_key', 'parameter_key', '1', '2', '3', '4', '5'] 
    long_lbc36 = long_lbc36[column_order]

    cohort_map = {
        'WHI': long_WHI,
        'sardiNIA': long_sardinia,
        'LBC21': long_lbc21,
        'LBC36': long_lbc36
    }

    # Filter cohort to only participants for which we have information
    all_part_ids = set.union(*[set(cohort.participant_id) for cohort in cohort_map.values()])
    cohort = [part for part in cohort if part.uns['participant_id'] in all_part_ids]

    # Process participant data
    processed_part_list = []
    for part in cohort:
        age = np.full((1, 8), np.nan)
        age[0, :part.shape[1]] = part.var.time_points

        cohort_data = cohort_map[part.uns['sub_cohort']]
        part_layers = cohort_data[cohort_data.participant_id == part.uns['participant_id']].copy()

        data = ad.AnnData(age, obs=participant_df[participant_df.participant_id == part.uns['participant_id']])

        for key in marker_map.global_name:
            layer_data = np.full((1, 8), np.nan)
            if key in part_layers.global_key.unique():
                marker_data = part_layers[part_layers.global_key == key].iloc[:, 3:].to_numpy().flatten()
                layer_data[0, :marker_data.shape[0]] = marker_data
            data.layers[key] = layer_data

        processed_part_list.append(data)

    data = ad.concat(processed_part_list)
    return data, marker_map

def preprocess_data(data, marker_map):
    """
    Preprocess the data for analysis.

    Args:
    data (AnnData): Input AnnData object
    marker_map (pd.DataFrame): Mapping of markers

    Returns:
    tuple: Preprocessed data, layer classification, and normality results
    """
    # Drop empty layers
    data, dropped_layers = drop_empty_layers(data)
    print(f"Dropped layers (all NaNs): {dropped_layers}")

    # Check for single observations
    single_observation = check_single_observation(data)
    layer_classification = classify_layers(single_observation)

    # Test and transform for normality by cohort
    data, normality_results = test_and_transform_normality_by_cohort(data, marker_map)

    # Convert to z-scores
    data = convert_to_z_scores(data, marker_map, normality_results)

    # Normalize observations
    normalise_obs(data, 'max_fitness')
    normalise_obs(data, 'max_size_prediction_120')

    return data, layer_classification, normality_results

def analyze_correlations(data, layer_classification, normality_results, predictor_columns):
    """
    Analyze correlations between predictors and markers.

    Args:
    data (AnnData): Input AnnData object
    layer_classification (dict): Classification of layers
    normality_results (dict): Results of normality tests
    predictor_columns (list): List of predictor column names

    Returns:
    tuple: Results dictionary and warnings log
    """
    results = {}
    warnings_log = {}
    for key in data.layers.keys():
        print(f"Analyzing {key}")
        if key in layer_classification['longitudinal']:
            result, warning = analyze_longitudinal(data, key, predictor_columns)
        elif key in layer_classification['single_time_point']:
            if normality_results[key] == 'binary':
                result, warning = analyze_binary(data, key, predictor_columns)
            else:
                result, warning = analyze_single_timepoint(data, key, predictor_columns)
        
        if warning:
            warnings_log[key] = warning
        else:
            results[key] = result
    
    return results, warnings_log

# Load data
data, marker_map = load_data()

# Preprocess data
data, layer_classification, normality_results = preprocess_data(data, marker_map)

# Analyze correlations for max_size_prediction_120_z_score
predictor_columns = ['max_size_prediction_120_z_score']
results_size, warnings_log_size = analyze_correlations(data, layer_classification, normality_results, predictor_columns)

print("Warnings encountered for max_size_prediction_120_z_score:")
for key, warning in warnings_log_size.items():
    print(f"{key}: {warning}")

print(f"\nNumber of markers analyzed: {len(results_size)}")
print(f"Number of markers dropped due to warnings: {len(warnings_log_size)}")

# Create summary table for max_size_prediction_120_z_score
summary_table_size = create_summary_table(results_size, predictor_columns)
print("\nSummary for max_size_prediction_120_z_score:")
print(summary_table_size)

# Analyze correlations for max_fitness_z_score
predictor_columns = ['max_fitness_z_score']
results_fitness, warnings_log_fitness = analyze_correlations(data, layer_classification, normality_results, predictor_columns)

print("\nWarnings encountered for max_fitness_z_score:")
for key, warning in warnings_log_fitness.items():
    print(f"{key}: {warning}")

print(f"\nNumber of markers analyzed: {len(results_fitness)}")
print(f"Number of markers dropped due to warnings: {len(warnings_log_fitness)}")

# Create summary table for max_fitness_z_score
summary_table_fitness = create_summary_table(results_fitness, predictor_columns)
print("\nSummary for max_fitness_z_score:")
print(summary_table_fitness)

# Combine summary tables and plot significant interactions
summary_table = pd.concat([summary_table_fitness, summary_table_size])
plot_significant_interactions(summary_table)