# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import statsmodels.formula.api as smf

participant_df = pd.read_csv('../results/participant_df.csv', index_col=0)
death_df = pd.read_csv('../data/sardiNIA/fabre_deaths upd 2024.csv', index_col=0)
sardinia_df = participant_df[participant_df.cohort=='sardiNIA'].copy()
sheet_names = ['I F', 'II F', 'III F','IV F', 'V F', 'Foglio5']

# Read Excel file with multiple sheets
xls = pd.read_excel('../data/sardiNIA/blood and biochemistry data for Margarete.xlsx', sheet_name=sheet_names, index_col=0)

# create list of dataframes
tp_df = [xls[s_n] for s_n in sheet_names[:-1]]

# overlapping participants
sardinia_df['participant_id'] = sardinia_df.participant_id.astype(int)
sardinia_df = sardinia_df.set_index('participant_id')
value_counts_sardinia = sardinia_df.index.value_counts()
keep_sardinia_idx = value_counts_sardinia[value_counts_sardinia<2].index

tp_df_index_list = [set(df.index) for df in tp_df]

overlapping_part = list(set.intersection(*tp_df_index_list, keep_sardinia_idx))

marker_list = [set(df.loc[overlapping_part].columns[2:]) for df in tp_df]
all_markers = set.union(*marker_list)

# drop markers:
all_markers = list(all_markers - set(['Unnamed: 17', 'A_PCR', 'ESA_ZNPP']))

# Ensure all dataframes have the same columns
for i, df in enumerate(tp_df):
    missing_columns = set(all_markers) - set(df.columns)
    for col in missing_columns:
        df[col] = np.nan

# create AnnData Object
adata = ad.AnnData(np.zeros((len(overlapping_part), 5)), obs=sardinia_df.loc[overlapping_part].copy())
for i in ['AGE1', 'AGE2', 'AGE3', 'AGE4', 'AGE5']:
    adata.obs[i] = death_df.loc[overlapping_part][i].values

for marker in all_markers:
    adata.layers[marker] = pd.concat([df.loc[overlapping_part][[marker]] for df in tp_df], axis=1)

# %%

# Run LMM for each marker
single_marker_results = {}
longitudinal_marker_results = {}
adata.layers.keys()
marker = 'ALBUMINA'
d = adata.layers[marker]
# single measurement associations
for key, d in adata.layers.items():
    d_dropped_nan_time_points = d[:,~np.all(np.isnan(d), axis=0)].copy()

    if d_dropped_nan_time_points.shape[1] == 0:
        continue

    elif d_dropped_nan_time_points.shape[1] == 1:
        df = pd.DataFrame.from_dict({'measurement' : d_dropped_nan_time_points.flatten(),
                      'fitness': adata.obs['norm_max_fitness'],
                      'max_size_prediction': adata.obs['norm_log_max_size_prediction_120'],
                       'age':adata.obs['AGE1']})

        # Center age and fitness for easier interpretation of main effects
        df['age_centered'] = df['age'] - df['age'].mean()
        df['fitness_centered'] = df['fitness'] - df['fitness'].mean()

        # Remove rows with NaN values
        df = df.dropna()
        
        # Specify the model formula
        model_formula = "scale(measurement) ~ age_centered + fitness_centered"
        
        # Fit the Linear Mixed-Effects Model
        linear_model = smf.ols(
            formula=model_formula,
            data=df) 

        single_marker_results[key] = linear_model.fit()
key ='ALBUMINA'
    elif d_dropped_nan_time_points.shape[1] > 1:
        # Reshape the data
        reshaped_data = []
        for participant_idx, participant_id in enumerate(adata.obs.index):
            for time_point in range(1, 6):  # 5 time points
                age_col = f'AGE{time_point}'
                reshaped_data.append({
                    'participant_id': participant_id,
                    'age': adata.obs[age_col][participant_idx],
                    'measurement': adata.layers[key][participant_idx, time_point-1],
                    'fitness': adata.obs['norm_max_fitness'][participant_idx]
                })
        
        df = pd.DataFrame(reshaped_data)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Prepare the data
        df['participant_id'] = df['participant_id'].astype('category')
        
        # Center age and fitness for easier interpretation of main effects
        df['age_centered'] = df['age'] - df['age'].mean()
        df['fitness_centered'] = df['fitness'] - df['fitness'].mean()
        
        # Specify the model formula
        model_formula = "scale(measurement) ~ age_centered + fitness_centered + age_centered:fitness_centered"
        df
        # Fit the Linear Mixed-Effects Model
        mixed_model = smf.mixedlm(
            formula=model_formula,
            data=df,
            groups='participant_id'
        )
        longitudinal_marker_results[key] = mixed_model.fit()
    
# %%

# Extract specific coefficients and p-values
long_coefficient_df = pd.DataFrame({
    marker: {
        'age_coef': fit.params['age_centered'],
        'age_pvalue': fit.pvalues['age_centered'],
        'fitness_coef': fit.params['fitness_centered'],
        'fitness_pvalue': fit.pvalues['fitness_centered'],
        'interaction_coef': fit.params['age_centered:fitness_centered'],
        'interaction_pvalue': fit.pvalues['age_centered:fitness_centered']
    } for marker, fit in longitudinal_marker_results.items()
}).T

print("\nCoefficients and p-values for all markers:")
print(long_coefficient_df)


# %%

# Extract specific coefficients and p-values
single_coefficient_df = pd.DataFrame({
    marker: {
        'age_coef': fit.params['age_centered'],
        'age_pvalue': fit.pvalues['age_centered'],
        'fitness_coef': fit.params['fitness_centered'],
        'fitness_pvalue': fit.pvalues['fitness_centered'],
    } for marker, fit in single_marker_results.items()
}).T

print("\nCoefficients and p-values for all markers:")
print(single_coefficient_df)

# %%

def plot_effect_sizes(longitudinal_marker_results, single_marker_results, p_value_threshold=0.05):
    # Extract coefficients, confidence intervals, and p-values for longitudinal markers
    long_coefficient_df = pd.DataFrame({
        marker: {
            'effect_size': fit.params['age_centered:fitness_centered'],
            'ci_lower': fit.conf_int().loc['age_centered:fitness_centered', 0],
            'ci_upper': fit.conf_int().loc['age_centered:fitness_centered', 1],
            'p_value': fit.pvalues['age_centered:fitness_centered']
        } for marker, fit in longitudinal_marker_results.items()
    }).T

    # Extract coefficients, confidence intervals, and p-values for single time point markers
    single_coefficient_df = pd.DataFrame({
        marker: {
            'effect_size': fit.params['fitness_centered'],
            'ci_lower': fit.conf_int().loc['fitness_centered', 0],
            'ci_upper': fit.conf_int().loc['fitness_centered', 1],
            'p_value': fit.pvalues['fitness_centered']
        } for marker, fit in single_marker_results.items()
    }).T

    # Filter for significant interactions
    long_df_significant = long_coefficient_df[long_coefficient_df['p_value'] < p_value_threshold].sort_values('effect_size')
    single_df_significant = single_coefficient_df[single_coefficient_df['p_value'] < p_value_threshold].sort_values('effect_size')

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Effect Sizes and Confidence Intervals for Markers (p < {p_value_threshold})', fontsize=16)

    # Plot longitudinal markers
    ax1.errorbar(long_df_significant['effect_size'], range(len(long_df_significant)), 
                 xerr=[long_df_significant['effect_size'] - long_df_significant['ci_lower'], 
                       long_df_significant['ci_upper'] - long_df_significant['effect_size']], 
                 fmt='o', capsize=5)
    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.set_yticks(range(len(long_df_significant)))
    ax1.set_yticklabels(long_df_significant.index)
    ax1.set_title(f'Longitudinal Markers (Interaction Coefficient)\n{len(long_df_significant)} / {len(long_coefficient_df)} significant')
    ax1.set_xlabel('Effect Size')

    # Plot single time point markers
    ax2.errorbar(single_df_significant['effect_size'], range(len(single_df_significant)), 
                 xerr=[single_df_significant['effect_size'] - single_df_significant['ci_lower'], 
                       single_df_significant['ci_upper'] - single_df_significant['effect_size']], 
                 fmt='o', capsize=5)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_yticks(range(len(single_df_significant)))
    ax2.set_yticklabels(single_df_significant.index)
    ax2.set_title(f'Single Time Point Markers (Fitness Coefficient)\n{len(single_df_significant)} / {len(single_coefficient_df)} significant')
    ax2.set_xlabel('Effect Size')

    plt.tight_layout()
    plt.show()

    return long_df_significant, single_df_significant

# Usage example:
long_significant, single_significant = plot_effect_sizes(longitudinal_marker_results, single_marker_results, p_value_threshold=0.05)
# %%


long_significant, single_significant = plot_effect_sizes(longitudinal_marker_results, single_marker_results, p_value_threshold=0.05)

# %%


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def run_analysis_and_plot(adata, all_markers, p_value_threshold=0.05):
    # Run LMM for each marker
    single_marker_results = {}
    longitudinal_marker_results = {}

    for key in all_markers:
        d = adata.layers[key]
        d_dropped_nan_time_points = d[:, ~np.all(np.isnan(d), axis=0)].copy()

        if d_dropped_nan_time_points.shape[1] == 0:
            continue
        elif d_dropped_nan_time_points.shape[1] == 1:
            df = pd.DataFrame.from_dict({
                'measurement': d_dropped_nan_time_points.flatten(),
                'fitness': adata.obs['norm_max_fitness'],
                'size': adata.obs['norm_log_max_size_prediction_120'],
                'age': adata.obs['AGE1']
            })

            # Center variables for easier interpretation of main effects
            df['age_centered'] = df['age'] - df['age'].mean()
            df['fitness_centered'] = df['fitness'] - df['fitness'].mean()
            df['size_centered'] = df['size'] - df['size'].mean()

            # Remove rows with NaN values
            df = df.dropna()
            
            # Specify the model formula
            model_formula = "scale(measurement) ~ age_centered + fitness_centered + size_centered"
            
            # Fit the Linear Model
            linear_model = smf.ols(formula=model_formula, data=df)
            single_marker_results[key] = linear_model.fit()

        elif d_dropped_nan_time_points.shape[1] > 1:
            # Reshape the data
            reshaped_data = []
            for participant_idx, participant_id in enumerate(adata.obs.index):
                for time_point in range(1, 6):  # 5 time points
                    age_col = f'AGE{time_point}'
                    reshaped_data.append({
                        'participant_id': participant_id,
                        'age': adata.obs[age_col][participant_idx],
                        'measurement': adata.layers[key][participant_idx, time_point-1],
                        'fitness': adata.obs['norm_max_fitness'][participant_idx],
                        'size': adata.obs['norm_log_max_size_prediction_120'][participant_idx]
                    })
            
            df = pd.DataFrame(reshaped_data)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Prepare the data
            df['participant_id'] = df['participant_id'].astype('category')
            
            # Center variables for easier interpretation of main effects
            df['age_centered'] = df['age'] - df['age'].mean()
            df['fitness_centered'] = df['fitness'] - df['fitness'].mean()
            df['size_centered'] = df['size'] - df['size'].mean()
            
            # Specify the model formula
            model_formula = "measurement ~ age_centered + fitness_centered + size_centered + age_centered:fitness_centered + age_centered:size_centered"
            
            # Fit the Linear Mixed-Effects Model
            mixed_model = smf.mixedlm(formula=model_formula, data=df, groups='participant_id')
            longitudinal_marker_results[key] = mixed_model.fit()

    # Extract results and create plots
    outcomes = ['fitness', 'size']
    for outcome in outcomes:
        long_coefficient_df = pd.DataFrame({
            marker: {
                'effect_size': fit.params[f'age_centered:{outcome}_centered'],
                'ci_lower': fit.conf_int().loc[f'age_centered:{outcome}_centered', 0],
                'ci_upper': fit.conf_int().loc[f'age_centered:{outcome}_centered', 1],
                'p_value': fit.pvalues[f'age_centered:{outcome}_centered']
            } for marker, fit in longitudinal_marker_results.items()
        }).T

        single_coefficient_df = pd.DataFrame({
            marker: {
                'effect_size': fit.params[f'{outcome}_centered'],
                'ci_lower': fit.conf_int().loc[f'{outcome}_centered', 0],
                'ci_upper': fit.conf_int().loc[f'{outcome}_centered', 1],
                'p_value': fit.pvalues[f'{outcome}_centered']
            } for marker, fit in single_marker_results.items()
        }).T

        # Filter for significant interactions
        long_df_significant = long_coefficient_df[long_coefficient_df['p_value'] < p_value_threshold].sort_values('effect_size')
        single_df_significant = single_coefficient_df[single_coefficient_df['p_value'] < p_value_threshold].sort_values('effect_size')

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Effect Sizes and Confidence Intervals for {outcome.capitalize()} (p < {p_value_threshold})', fontsize=16)

        # Plot longitudinal markers
        ax1.errorbar(long_df_significant['effect_size'], range(len(long_df_significant)), 
                     xerr=[long_df_significant['effect_size'] - long_df_significant['ci_lower'], 
                           long_df_significant['ci_upper'] - long_df_significant['effect_size']], 
                     fmt='o', capsize=5)
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_yticks(range(len(long_df_significant)))
        ax1.set_yticklabels(long_df_significant.index)
        ax1.set_title(f'Longitudinal Markers (Interaction with {outcome.capitalize()})\n{len(long_df_significant)} / {len(long_coefficient_df)} significant')
        ax1.set_xlabel('Effect Size')

        # Plot single time point markers
        ax2.errorbar(single_df_significant['effect_size'], range(len(single_df_significant)), 
                     xerr=[single_df_significant['effect_size'] - single_df_significant['ci_lower'], 
                           single_df_significant['ci_upper'] - single_df_significant['effect_size']], 
                     fmt='o', capsize=5)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_yticks(range(len(single_df_significant)))
        ax2.set_yticklabels(single_df_significant.index)
        ax2.set_title(f'Single Time Point Markers ({outcome.capitalize()} Coefficient)\n{len(single_df_significant)} / {len(single_coefficient_df)} significant')
        ax2.set_xlabel('Effect Size')

        plt.tight_layout()
        plt.show()

    return longitudinal_marker_results, single_marker_results

# Usage example:
longitudinal_results, single_results = run_analysis_and_plot(adata, all_markers, p_value_threshold=0.05)
# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def run_analysis_and_plot(marker, adata, all_markers, p_value_threshold=0.05):
    # Run LMM for each marker
    single_marker_results = {}
    longitudinal_marker_results = {}

    for key in all_markers:
        d = adata.layers[key]
        d_dropped_nan_time_points = d[:, ~np.all(np.isnan(d), axis=0)].copy()

        if d_dropped_nan_time_points.shape[1] == 0:
            continue

        elif d_dropped_nan_time_points.shape[1] == 1:
            df = pd.DataFrame.from_dict({
                'measurement': d_dropped_nan_time_points.flatten(),
                'marker': adata.obs[marker],
                'age': adata.obs['AGE1']
            })

            # Center variables for easier interpretation of main effects
            df['age_centered'] = df['age'] - df['age'].mean()
            df['marker_centered'] = df['marker'] - df['marker'].mean()

            # Remove rows with NaN values
            df = df.dropna()
            
            # Specify the model formula
            model_formula = "scale(measurement) ~ age_centered + marker_centered"
            
            # Fit the Linear Model
            linear_model = smf.ols(formula=model_formula, data=df)
            single_marker_results[key] = linear_model.fit()

        elif d_dropped_nan_time_points.shape[1] > 1:
            # Reshape the data
            reshaped_data = []
            for participant_idx, participant_id in enumerate(adata.obs.index):
                for time_point in range(1, 6):  # 5 time points
                    age_col = f'AGE{time_point}'
                    reshaped_data.append({
                        'participant_id': participant_id,
                        'age': adata.obs[age_col][participant_idx],
                        'measurement': adata.layers[key][participant_idx, time_point-1],
                        'marker': adata.obs[marker][participant_idx]
                    })
            
            df = pd.DataFrame(reshaped_data)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Prepare the data
            df['participant_id'] = df['participant_id'].astype('category')
            
            # Center variables for easier interpretation of main effects
            df['age_centered'] = df['age'] - df['age'].mean()
            df['marker_centered'] = df['marker'] - df['marker'].mean()
            
            # Specify the model formula
            model_formula = "measurement ~ age_centered + marker_centered + age_centered:marker_centered"
            
            # Fit the Linear Mixed-Effects Model
            mixed_model = smf.mixedlm(formula=model_formula, data=df, groups='participant_id')
            longitudinal_marker_results[key] = mixed_model.fit()

    # Extract results and create plots
    long_coefficient_df = pd.DataFrame({
        marker: {
            'effect_size': fit.params['age_centered:marker_centered'],
            'ci_lower': fit.conf_int().loc['age_centered:marker_centered', 0],
            'ci_upper': fit.conf_int().loc['age_centered:marker_centered', 1],
            'p_value': fit.pvalues['age_centered:marker_centered']
        } for marker, fit in longitudinal_marker_results.items()
    }).T

    single_coefficient_df = pd.DataFrame({
        marker: {
            'effect_size': fit.params['marker_centered'],
            'ci_lower': fit.conf_int().loc['marker_centered', 0],
            'ci_upper': fit.conf_int().loc['marker_centered', 1],
            'p_value': fit.pvalues['marker_centered']
        } for marker, fit in single_marker_results.items()
    }).T

    # Filter for significant interactions
    long_df_significant = long_coefficient_df[long_coefficient_df['p_value'] < p_value_threshold].sort_values('effect_size')
    single_df_significant = single_coefficient_df[single_coefficient_df['p_value'] < p_value_threshold].sort_values('effect_size')

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Effect Sizes and Confidence Intervals for Size Prediction (p < {p_value_threshold})', fontsize=16)

    # Plot longitudinal markers
    ax1.errorbar(long_df_significant['effect_size'], range(len(long_df_significant)), 
                 xerr=[long_df_significant['effect_size'] - long_df_significant['ci_lower'], 
                       long_df_significant['ci_upper'] - long_df_significant['effect_size']], 
                 fmt='o')
    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.set_yticks(range(len(long_df_significant)))
    ax1.set_yticklabels(long_df_significant.index)
    ax1.set_title(f'Longitudinal Markers\n{len(long_df_significant)} / {len(long_coefficient_df)} significant')
    ax1.set_xlabel('Effect Size')

    # Plot single time point markers
    ax2.errorbar(single_df_significant['effect_size'], range(len(single_df_significant)), 
                 xerr=[single_df_significant['effect_size'] - single_df_significant['ci_lower'], 
                       single_df_significant['ci_upper'] - single_df_significant['effect_size']], 
                 fmt='o')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_yticks(range(len(single_df_significant)))
    ax2.set_yticklabels(single_df_significant.index)
    ax2.set_title(f'Single Time Point Markers\n{len(single_df_significant)} / {len(single_coefficient_df)} significant')
    ax2.set_xlabel('Effect Size')

    plt.tight_layout()
    plt.show()

    return longitudinal_marker_results, single_marker_results
# %%

# Usage example:
# marker = 'norm_log_max_size_prediction_120'
marker = 'max_fitness'
longitudinal_results, single_results = run_analysis_and_plot(marker, adata, all_markers, p_value_threshold=0.05)
# %%
