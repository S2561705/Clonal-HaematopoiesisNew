# auxiliary_functions.py

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm, ols, logit
from scipy import stats

def drop_empty_layers(data):
    """
    Drop layers from AnnData object that contain only NaN values.

    Args:
    data (AnnData): Input AnnData object

    Returns:
    tuple: Updated AnnData object and list of dropped layer names
    """
    layers_to_drop = [key for key in data.layers.keys() if np.all(np.isnan(data.layers[key]))]
    for key in layers_to_drop:
        del data.layers[key]
    return data, layers_to_drop

def check_single_observation(data):
    """
    Check if each layer in AnnData object has at most one non-NaN observation per participant.

    Args:
    data (AnnData): Input AnnData object

    Returns:
    dict: Dictionary indicating whether each layer has at most one observation
    """
    return {key: np.all(np.sum(~np.isnan(layer), axis=1) <= 1) for key, layer in data.layers.items()}

def classify_layers(single_observation):
    """
    Classify layers as single time point or longitudinal based on observation count.

    Args:
    single_observation (dict): Dictionary of single observation results

    Returns:
    dict: Classification of layers into 'single_time_point' and 'longitudinal'
    """
    classification = {'single_time_point': [], 'longitudinal': []}
    for key, is_single in single_observation.items():
        classification['single_time_point' if is_single else 'longitudinal'].append(key)
    return classification


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
    layer_data (np.array): Input data
    factor (float): The IQR factor to use for determining outliers (default 3)
    
    Returns:
    np.array: Data with outliers replaced by NaN
    """
    q5, q95 = np.nanpercentile(layer_data, [5, 95])
    iqr = q95 - q5
    lower_bound = q5 - (factor * iqr)
    upper_bound = q95 + (factor * iqr)
    
    return np.where((layer_data >= lower_bound) & (layer_data <= upper_bound), layer_data, np.nan)

def transform_binary_to_01(data):
    """
    Transform binary data to 0 and 1.

    Args:
    data (np.array): Input data

    Returns:
    np.array: Transformed data
    """
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    return np.where(data == min_val, 0, np.where(data == max_val, 1, data))

# def test_and_transform_normality_by_cohort(data, marker_map):
#     """
#     Test and transform data for normality by cohort.

#     Args:
#     data (AnnData): Input AnnData object
#     marker_map (pd.DataFrame): Mapping of markers

#     Returns:
#     tuple: Updated AnnData object and dictionary of transformation results
#     """
#     transformation_dict = {}
#     cohorts = data.obs.cohort.unique()

#     for key in marker_map.global_name:
#         if key not in data.layers:
#             continue
        
#         transformation_dict[key] = {}
#         layer_data = handle_outliers_iqr(data.layers[key])
#         non_nan_data = layer_data[~np.isnan(layer_data)]

#         if np.unique(non_nan_data).shape[0] == 2:
#             data.layers[key] = transform_binary_to_01(data.layers[key])
#             transformation_dict[key] = 'binary'
#             continue

#         for cohort in cohorts:
#             cohort_mask = data.obs.cohort == cohort
#             layer_data = data.layers[key][cohort_mask].flatten()
#             non_nan_data = layer_data[~np.isnan(layer_data)]

#             if len(non_nan_data) == 0:
#                 transformation_dict[key][cohort] = 'NaN'
#                 continue

#             layer_data = handle_outliers_iqr(layer_data)  
#             non_nan_data = layer_data[~np.isnan(layer_data)]
            
#             if np.unique(non_nan_data).shape[0] == 2:
#                 data.layers[key] = transform_binary_to_01(data.layers[key])
#                 transformation_dict[key][cohort] = 'binary'
#                 continue

#             data.layers[key], best_transform, p_value = find_best_transformation(data.layers[key])   
#             transformation_dict[key][cohort] = [best_transform, p_value]

#     return data, transformation_dict

def z_score(data):
    """
    Calculate z-scores for input data.

    Args:
    data (np.array): Input data

    Returns:
    np.array: Z-scored data
    """
    mean, std = np.nanmean(data), np.nanstd(data)
    return (data - mean) / std

def convert_to_z_scores(data, marker_map, normality_results):
    """
    Convert data to z-scores by cohort.

    Args:
    data (AnnData): Input AnnData object
    marker_map (pd.DataFrame): Mapping of markers
    normality_results (dict): Results of normality tests

    Returns:
    AnnData: Updated AnnData object with z-scored layers
    """
    cohorts = data.obs.cohort.unique()
    
    for key in marker_map.global_name:
        if key not in data.layers or normality_results[key] == 'binary':
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

def normalise_obs(data, observation):
    """
    Normalize and convert markers to z-scores.

    Args:
    data (AnnData): Input AnnData object
    observation (str): Name of the observation to normalize

    Returns:
    None: Updates the data object in-place
    """
    transformed_data, _, _ = find_best_transformation(data.obs[observation])
    z_data = z_score(transformed_data)
    data.obs[observation+'_z_score'] = z_data

def analyze_longitudinal(data, key, predictor_columns):
    """
    Analyze longitudinal data using a linear mixed model.

    Args:
    data (AnnData): Input AnnData object
    key (str): Key for the layer to analyze
    predictor_columns (list): List of predictor column names

    Returns:
    tuple: Dictionary of results and warning message (if any)
    """
    df_long = pd.DataFrame({
        'participant_id': np.repeat(data.obs.index, data.X.shape[1]),
        'age': data.X.flatten(),
        'marker': data.layers[key].flatten(),
        **{col: np.repeat(data.obs[col], data.X.shape[1]) for col in predictor_columns}
    }).dropna()
    
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
    """
    Analyze single time point data using a linear model.

    Args:
    data (AnnData): Input AnnData object
    key (str): Key for the layer to analyze
    predictor_columns (list): List of predictor column names

    Returns:
    tuple: Dictionary of results and warning message (if any)
    """
    df = pd.DataFrame({
        'marker': np.nansum(data.layers[key], axis=1),
        **{col: data.obs[col] for col in predictor_columns}
    }).dropna()
    
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
    """
    Analyze binary data using a logistic model.

    Args:
    data (AnnData): Input AnnData object
    key (str): Key for the layer to analyze
    predictor_columns (list): List of predictor column names

    Returns:
    tuple: Dictionary of results and warning message (if any)
    """
    df = pd.DataFrame({
        'marker': np.nansum(data.layers[key], axis=1),
        **{col: data.obs[col] for col in predictor_columns}
    }).dropna()
    
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
    """
    Create a summary table of analysis results.

    Args:
    results (dict): Dictionary of analysis results
    predictor_columns (list): List of predictor column names

    Returns:
    pd.DataFrame: Summary table of results
    """
    summary_data = [
        {
            'Marker': marker,
            'Predictor': predictor,
            'Model': result['model'],
            'Coefficient': result['coefficients'][f'{predictor}_coefficient'],
            'P-value': result['results'][f'{predictor}_p_value'],
            'CI Lower': result['conf_int'][f'{predictor}_conf_int'][0],
            'CI Upper': result['conf_int'][f'{predictor}_conf_int'][1]
        }
        for marker, result in results.items()
        for predictor in predictor_columns
    ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Significant'] = summary_df['P-value'] < 0.05
    return summary_df

def plot_significant_interactions(summary_df):
    """
    Plot significant interactions from the summary table.

    Args:
    summary_df (pd.DataFrame): Summary table of results

    Returns:
    None: Displays the plot
    """
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