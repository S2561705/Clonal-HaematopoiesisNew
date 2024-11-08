# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm, ols, logit
from scipy import stats

# Import results
participant_df = pd.read_csv('../results/participant_df.csv', index_col=0)

# create cohort_specific_id
participant_df['participant_id'] = participant_df.apply(lambda row: row['cohort'] + row['participant_id'] if not row['cohort'].startswith('LBC') else row['participant_id'], axis=1)
duplicated_ids = participant_df[participant_df.duplicated(subset='participant_id', keep=False)]['participant_id'].unique()
participant_df = participant_df.drop_duplicates(subset='participant_id', keep='first')

# Import cohort
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

new_cohort = []

for part in cohort:
    # modify cohort specific id
    if part.uns['cohort'] != 'LBC':
        part.uns['participant_id'] = part.uns['cohort'] + str(part.uns['participant_id'])

    participant_id = part.uns['participant_id']
    
    if participant_id in duplicated_ids:
        # Check if this is the first instance of this participant_id
        if not any(p.uns['participant_id'] == participant_id for p in new_cohort):
            new_cohort.append(part)
    else:
        # If not a duplicated ID, always keep it
        new_cohort.append(part)

# Replace the original cohort with the new one
cohort = new_cohort


long_lbc21 = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC21/LBC21_LongitudinalVariables.matrix.tsv', sep='\t')
long_lbc36 = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC36/LBC1936_LongitudinalVariables.matrix.tsv', sep='\t')
long_sardinia = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/SardiNIA/SardiNIA_LongitudinalVariables.matrix.tsv', sep='\t')
long_WHI = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/WHI/WHI_LongitudinalVariables.matrix.tsv', sep='\t')
long_WHI = long_WHI.rename(columns={'ID':'participant_id'})
marker_map = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/cohort_param_values_map.txt', sep='\t')

# modify cohort_specific_id
long_sardinia['participant_id'] = 'sardiNIA' + long_sardinia['participant_id'].astype('str')
long_WHI['participant_id'] = 'WHI' + long_WHI['participant_id'].astype('str')

# Add missing global_key to LBC36
long_lbc36['global_key'] =long_lbc36['parameter_key'].map(
    dict(zip(marker_map['LBC36'], marker_map['global_name'])))

column_order = ['participant_id', 'global_key', 'parameter_key', '1','2','3','4','5'] 
long_lbc36 = long_lbc36[column_order]

cohort_map = {'WHI': long_WHI,
                'sardiNIA': long_sardinia,
                'LBC21': long_lbc21,
                'LBC36': long_lbc36}


# filtered cohort to only participants for which we have information
all_part_ids = [set(cohort.participant_id) for cohort in [long_WHI, long_sardinia, long_lbc21, long_lbc36]]
all_part_ids = set.union(*all_part_ids)

cohort_ids = [part.uns['participant_id'] for part in cohort]

cohort = [part for part in cohort if part.uns['participant_id'] in all_part_ids]

# all_part_ids_str = [str(part_id) for part_id in all_part_ids]
processed_part_list = []
for part in cohort:
    # create age array with maximum shape 8
    # this is because we want all participants to have same shape
    age = np.empty((1,8))
    # fill with NaNs to handle participants with less time points
    age[:] = np.nan
    age[0,: part.shape[1]] = part.var.time_points


    cohort_data = cohort_map[part.uns['sub_cohort']]
    part_layers = cohort_data[cohort_data.participant_id == part.uns['participant_id']].copy()

    data = ad.AnnData(age, obs=participant_df[participant_df.participant_id == part.uns['participant_id']])

    for key in marker_map.global_name:

        if key in part_layers.global_key.unique():
            marker_data = part_layers[part_layers.global_key==key].iloc[:, 3:].to_numpy().flatten()

            layer_data = np.empty((1,8))
            # fill with NaNs to handle participants with less time points
            layer_data[:] = np.nan
            layer_data[0,:marker_data.shape[0] ] = marker_data
            data.layers[key] = layer_data

        else:
            # fill layer with NaNs

            layer_data = np.empty((1,8))
            layer_data[:] = np.nan
            data.layers[key] = layer_data

    processed_part_list.append(data)

data = ad.concat(processed_part_list)

# %%

def drop_empty_layers(data):
    layers_to_drop = []
    for key in data.layers.keys():
        if np.all(np.isnan(data.layers[key])):
            layers_to_drop.append(key)
    
    for key in layers_to_drop:
        del data.layers[key]
    
    return data, layers_to_drop

def check_single_observation(data):
    single_observation = {}
    for key in data.layers.keys():
        if key in data.layers:
            layer = data.layers[key]
            # Count non-NaN values for each participant
            obs_count = np.sum(~np.isnan(layer), axis=1)
            # Check if all participants have at most one observation
            single_observation[key] = np.all(obs_count <= 1)

    return single_observation

def classify_layers(single_observation):
    layer_classification = {
        'single_time_point': [],
        'longitudinal': []
    }
    for key, is_single in single_observation.items():
        if is_single:
            layer_classification['single_time_point'].append(key)
        else:
            layer_classification['longitudinal'].append(key)
    return layer_classification


def find_best_transformation(data):
    transformations = {
        'log': np.log1p,
        'sqrt': np.sqrt,
        'boxcox': lambda x: stats.boxcox(x + 1)[0],
        'yeojohnson': stats.yeojohnson,
        'no_transform': np.array
    }

    flat_data = data.ravel()
    mask = ~np.isnan(flat_data)
    non_nan_data = flat_data[mask]
    
    # make data >1 by simply translating 
    # this doesn't affect the shape of values, so fine for converting into z-scores
    non_nan_data = non_nan_data + 1 + -min(np.min(non_nan_data), 0)

    best_score = -np.inf 
    best_transform = None

    for name, func in transformations.items():
        try:
            transformed = func(non_nan_data)
            _, p_value = stats.normaltest(transformed)
            if p_value > best_score:
                best_score = p_value
                best_transform = name
        except:
            continue

    # tranform data using optimal transformation
    best_transform_func = transformations[best_transform]
    flat_data[mask] = best_transform_func(non_nan_data)
    transformed_data = flat_data.reshape(data.shape)

    return transformed_data, best_transform, best_score

def handle_outliers_iqr(layer_data, factor=3):
    """
    Replace outliers with NaN using the Interquartile Range (IQR) method.
    
    Args:
    data (np.array): Input data
    factor (float): The IQR factor to use for determining outliers (default 1.5)
    
    Returns:
    np.array: Data with outliers replaced by NaN
    """
    q5 = np.nanpercentile(layer_data, 5)
    q95 = np.nanpercentile(layer_data, 95)
    iqr = q95 - q5
    lower_bound = q5 - (factor * iqr)
    upper_bound = q95 + (factor * iqr)
    
    return np.where((layer_data >= lower_bound) & (layer_data <= upper_bound), layer_data, np.nan)

def transform_binary_to_01 (data):

    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    # Apply the transformation to the original data
    data = np.where(data == min_val, 0, data)
    data = np.where(data == max_val, 1, data)
    
    return data

def test_and_transform_normality_by_cohort(data, marker_map):
    transformation_dict = {}
    cohorts = data.obs.cohort.unique()

    for key in marker_map.global_name:
        if key not in data.layers:
            continue
        
        transformation_dict[key] = {}

        # check if data is binary

        layer_data = data.layers[key]
        # avoid outliers using iqr method
        layer_data = handle_outliers_iqr(layer_data)  
        non_nan_data = layer_data[~np.isnan(layer_data)]

        # Check if the data is binary
        if np.unique(non_nan_data).shape[0] == 2:
            # If key has binary values, 
            # make sure they are transformed to 0 and 1
            data.layers[key] = transform_binary_to_01(data.layers[key])

            transformation_dict[key] = 'binary'
            continue

        for cohort in cohorts:
            cohort_mask = data.obs.cohort == cohort
            layer_data = data.layers[key][cohort_mask].flatten()

            # use only non_nan_data 
            non_nan_data = layer_data[~np.isnan(layer_data)]

            if len(non_nan_data) == 0:
                transformation_dict[key][cohort] = 'NaN'
                continue

            # avoid outliers using iqr method
            layer_data = handle_outliers_iqr(layer_data)  
            non_nan_data = layer_data[~np.isnan(layer_data)]
            
            # Check if the data is binary
            if np.unique(non_nan_data).shape[0] == 2:
                # If key has binary values, 
                # make sure they are transformed to 0 and 1
                data.layers[key] = transform_binary_to_01(data.layers[key])

                transformation_dict[key][cohort] = 'binary'
                continue

            # find optimal transformation to normalise data
            data.layers[key], best_transform, p_value = find_best_transformation(data.layers[key])   
            transformation_dict[key][cohort] = [best_transform, p_value]

    return data, transformation_dict

def z_score(data):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return (data-mean)/std

def convert_to_z_scores(data, marker_map, normality_results):
    cohorts = data.obs.cohort.unique()
    
    for key in marker_map.global_name:
        if key not in data.layers:
            continue
        if normality_results[key] == 'binary':
            continue
        for cohort in cohorts:
            cohort_mask = data.obs.cohort == cohort
            layer_data = data.layers[key][cohort_mask]
            non_nan_mask = ~np.isnan(layer_data)
            
            if np.any(non_nan_mask):
                mean = np.mean(layer_data[non_nan_mask])
                std = np.std(layer_data[non_nan_mask])
                if std != 0:
                    data.layers[key][cohort_mask] = (layer_data - mean) / std
    
    return data

# Assuming 'data' and 'marker_map' are already defined

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

def normalise_obs (data, observation):
    """Normalise and convert markers to z_scores """
    transformed_data, best_func, p_value = find_best_transformation(data.obs[observation])

    z_data = z_score(transformed_data)
    data.obs[observation+'_z_score'] = z_data
    
normalise_obs(data, 'max_fitness')
normalise_obs(data, 'max_size_prediction_120')


# %%

def analyze_correlations(data, layer_classification, normality_results, predictor_columns):
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

def analyze_longitudinal(data, key, predictor_columns):
    df_long = pd.DataFrame({
        'participant_id': np.repeat(data.obs.index, data.X.shape[1]),
        'age': data.X.flatten(),
        'marker': data.layers[key].flatten(),
        **{col: np.repeat(data.obs[col], data.X.shape[1]) for col in predictor_columns}
    })
    df_long = df_long.dropna()
    
    formula = f"marker ~ age * ({' + '.join(predictor_columns)})"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model = mixedlm(formula, data=df_long, groups="participant_id")
            result = model.fit()
            
            if len(w) > 0:
                return None, str(w[-1].message)
            
            return {
                'model': 'Linear Mixed Model',
                'results': {f'{col}_p_value': result.pvalues[f'age:{col}'] for col in predictor_columns},
                'coefficients': {f'{col}_coefficient': result.params[f'age:{col}'] for col in predictor_columns},
                'conf_int': {f'{col}_conf_int': result.conf_int().loc[f'age:{col}'].tolist() for col in predictor_columns}
            }, None
        except Exception as e:
            return None, str(e)

def analyze_single_timepoint(data, key, predictor_columns):
    df = pd.DataFrame({
        'marker': np.nansum(data.layers[key], axis=1),
        **{col: data.obs[col] for col in predictor_columns}
    })
    df = df.dropna()
    
    formula = f"marker ~ {' + '.join(predictor_columns)}"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model = ols(formula, data=df)
            result = model.fit()
            
            if len(w) > 0:
                return None, str(w[-1].message)
            
            return {
                'model': 'Linear Model',
                'results': {f'{col}_p_value': result.pvalues[col] for col in predictor_columns},
                'coefficients': {f'{col}_coefficient': result.params[col] for col in predictor_columns},
                'conf_int': {f'{col}_conf_int': result.conf_int().loc[col].tolist() for col in predictor_columns}
            }, None
        except Exception as e:
            return None, str(e)

def analyze_binary(data, key, predictor_columns):
    df = pd.DataFrame({
        'marker': np.nansum(data.layers[key], axis=1),
        **{col: data.obs[col] for col in predictor_columns}
    })
    df = df.dropna()
    
    formula = f"marker ~ {' + '.join(predictor_columns)}"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model = logit(formula, data=df)
            result = model.fit()
            
            if len(w) > 0:
                return None, str(w[-1].message)
            
            return {
                'model': 'Logistic Model',
                'results': {f'{col}_p_value': result.pvalues[col] for col in predictor_columns},
                'coefficients': {f'{col}_coefficient': result.params[col] for col in predictor_columns},
                'conf_int': {f'{col}_conf_int': result.conf_int().loc[col].tolist() for col in predictor_columns}
            }, None
        except Exception as e:
            return None, str(e)

def create_summary_table(results, predictor_columns):
    summary_data = []
    for marker, result in results.items():
        for predictor in predictor_columns:
            summary_data.append({
                'Marker': marker,
                'Predictor': predictor,
                'Model': result['model'],
                'Coefficient': result['coefficients'][f'{predictor}_coefficient'],
                'P-value': result['results'][f'{predictor}_p_value'],
                'CI Lower': result['conf_int'][f'{predictor}_conf_int'][0],
                'CI Upper': result['conf_int'][f'{predictor}_conf_int'][1]
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Significant'] = summary_df['P-value'] < 0.05
    return summary_df

def plot_significant_interactions(summary_df):
    significant_df = summary_df[summary_df['Significant']].reset_index(drop=True)
    
    if len(significant_df) == 0:
        print("No significant interactions found.")
        return
    
    plt.figure(figsize=(12, len(significant_df) * 0.3))
    
    # Create a categorical y-axis
    significant_df['y_position'] = significant_df.groupby('Marker').ngroup()
    
    # Plot points
    sns.scatterplot(x='Coefficient', y='y_position', hue='Predictor', data=significant_df, 
                    style='Predictor', s=100, zorder=3)
    
    # Plot confidence intervals
    for _, row in significant_df.iterrows():
        plt.plot([row['CI Lower'], row['CI Upper']], [row['y_position'], row['y_position']], 
                 color='gray', alpha=0.5, zorder=2)
    
    # Customize the plot
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, zorder=1)
    plt.title('Significant Interactions with Confidence Intervals')
    plt.xlabel('Coefficient')
    plt.ylabel('Marker')
    
    # Set y-tick labels
    plt.yticks(significant_df['y_position'], significant_df['Marker'])
    
    plt.tight_layout()
    plt.show()
# %%

# For max_size_prediction_120_z_score
predictor_columns = ['max_VAF_z_score']
results_size, warnings_log_size = analyze_correlations(data, layer_classification, normality_results, predictor_columns)

print("Warnings encountered for max_size_prediction_120_z_score:")
for key, warning in warnings_log_size.items():
    print(f"{key}: {warning}")

print(f"\nNumber of markers analyzed: {len(results_size)}")
print(f"Number of markers dropped due to warnings: {len(warnings_log_size)}")

# Create summary table
summary_table_size = create_summary_table(results_size, predictor_columns)
plot_significant_interactions(summary_table_size)

# For max_fitness_z_score
predictor_columns = ['max_fitness_z_score']
results_fitness, warnings_log_fitness = analyze_correlations(data, layer_classification, normality_results, predictor_columns)

print("\nWarnings encountered for max_fitness_z_score:")
for key, warning in warnings_log_fitness.items():
    print(f"{key}: {warning}")

print(f"\nNumber of markers analyzed: {len(results_fitness)}")
print(f"Number of markers dropped due to warnings: {len(warnings_log_fitness)}")

# Create summary table
summary_table_fitness = create_summary_table(results_fitness, predictor_columns)

summary_table = pd.concat([summary_table_fitness, summary_table_size])
plot_significant_interactions(summary_table)

# %%

summary_table[summary_table['P-value']<0.05]