# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

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
single_markers = []
# single measurement associations
for key, d in adata.layers.items():
    d_dropped_nan_time_points = d[:,~np.all(np.isnan(d), axis=0)].copy()
    if d_dropped_nan_time_points.shape[1] == 1:
        # sns.regplot(x=d_dropped_nan_time_points.flatten(),
        #                 y=adata.obs.norm_log_max_size_prediction_120)
        # plt.title(key)
        # plt.show()
        # plt.clf()
        single_markers.append(key)
        adata.obs[key] = d_dropped_nan_time_points.flatten()

# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#longitudinal markers
longitudinal_markers = [m for m in all_markers if m not in single_markers]

def run_lmm_for_marker(adata, marker):
    if adata.layers[marker][:,~np.all(np.isnan(adata.layers[marker]), axis=0)].shape[1]<2:
        print(f"Error fitting model for marker: {marker}")
        return None, None

    # Reshape the data
    reshaped_data = []
    for participant_idx, participant_id in enumerate(adata.obs.index):
        for time_point in range(1, 6):  # 5 time points
            age_col = f'AGE{time_point}'
            reshaped_data.append({
                'participant_id': participant_id,
                'age': adata.obs[age_col][participant_idx],
                'measurement': adata.layers[marker][participant_idx, time_point-1],
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
    model_formula = "measurement ~ age_centered + fitness_centered + age_centered:fitness_centered"
    
    # Fit the Linear Mixed-Effects Model
    mixed_model = smf.mixedlm(
        formula=model_formula,
        data=df,
        groups='participant_id'
    )
    
    # Fit the model and get results
    try:
        model_fit = mixed_model.fit()
        return model_fit, df
    except:
        print(f"Error fitting model for marker: {marker}")
        return None, df

# Run LMM for each marker
results = {}
for marker in longitudinal_markers:
    print(f"Running LMM for marker: {marker}")
    model_fit, df = run_lmm_for_marker(adata, marker)
    if model_fit is not None:
        results[marker] = model_fit

# Extract specific coefficients and p-values
coefficient_df = pd.DataFrame({
    marker: {
        'age_coef': fit.params['age_centered'],
        'age_pvalue': fit.pvalues['age_centered'],
        'fitness_coef': fit.params['fitness_centered'],
        'fitness_pvalue': fit.pvalues['fitness_centered'],
        'interaction_coef': fit.params['age_centered:fitness_centered'],
        'interaction_pvalue': fit.pvalues['age_centered:fitness_centered']
    } for marker, fit in results.items()
}).T

print("\nCoefficients and p-values for all markers:")
print(coefficient_df)

# Sort markers by the p-value of the interaction term
sorted_markers = coefficient_df.sort_values('interaction_pvalue').index

# Print detailed results for top 5 markers with the most significant interaction
print("\nDetailed results for top 5 markers with most significant interaction:")
for marker in sorted_markers[:5]:
    print(f"\nResults for {marker}:")
    print(results[marker].summary())
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_interaction_effect_plot(results, figsize=(12, len(results)*0.3), show_significant_only=False):
    # Extract interaction coefficients and standard errors
    coef_data = []
    for marker, model in results.items():
        coef = model.params['age_centered:fitness_centered']
        std_err = model.bse['age_centered:fitness_centered']
        lower = coef - 1.96 * std_err
        upper = coef + 1.96 * std_err
        is_significant = (lower > 0) or (upper < 0)  # Check if CI doesn't include 0
        coef_data.append({
            'marker': marker,
            'coef': coef,
            'lower': lower,
            'upper': upper,
            'abs_coef': abs(coef),
            'is_significant': is_significant
        })
    
    df = pd.DataFrame(coef_data)
    
    # Filter for significant interactions if requested
    if show_significant_only:
        df = df[df['is_significant']]
    
    df = df.sort_values('abs_coef', ascending=False)  # Sort by absolute coefficient value
    
    # Adjust figure size based on number of markers to display
    adjusted_figsize = (figsize[0], min(figsize[1], len(df)*0.3))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=adjusted_figsize)
    
    # Plot the coefficients and CIs
    ax.scatter(df['coef'], range(len(df)), color='blue', s=50, zorder=2)
    ax.hlines(range(len(df)), df['lower'], df['upper'], color='blue', alpha=0.5, linewidth=2, zorder=1)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, zorder=0)
    
    # Customize the plot
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['marker'])
    ax.set_xlabel('Interaction Coefficient (Age * Fitness)')
    title = 'Interaction Effect Sizes with 95% Confidence Intervals\n(Ordered by Absolute Effect Size)'
    if show_significant_only:
        title += '\nSignificant Interactions Only'
    ax.set_title(title)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Return the sorted dataframe for further analysis if needed
    return df

# Example usage:
# Show all interactions
all_interactions_df = create_interaction_effect_plot(results, show_significant_only=False)

# Show only significant interactions
significant_interactions_df = create_interaction_effect_plot(results, show_significant_only=True)

# Print the number of significant interactions
num_significant = significant_interactions_df.shape[0]
print(f"\nNumber of significant interactions: {num_significant}")

# Print the top 10 significant markers with the strongest interaction effects
print("\nTop 10 significant markers with strongest interaction effects:")
print(significant_interactions_df[['marker', 'coef', 'lower', 'upper']].head(10))
# %%


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def analyze_single_markers(adata, single_markers):
    results = []
    for marker in single_markers:
        if marker in adata.obs.columns:
            # Drop NaN values for both the marker and fitness
            valid_data = adata.obs[[marker, 'norm_max_fitness']].dropna()
            
            if len(valid_data) > 2:  # Ensure we have at least 3 valid data points
                corr, p_value = stats.pearsonr(valid_data['norm_max_fitness'], valid_data[marker])
                n = len(valid_data)
                
                # Calculate confidence interval
                z = np.arctanh(corr)
                se = 1/np.sqrt(n-3)
                z_lower, z_upper = z - 1.96*se, z + 1.96*se
                lower, upper = np.tanh(z_lower), np.tanh(z_upper)
                
                results.append({
                    'marker': marker,
                    'coef': corr,
                    'p_value': p_value,
                    'lower': lower,
                    'upper': upper,
                    'abs_coef': abs(corr),
                    'is_significant': p_value < 0.05,
                    'is_longitudinal': False,
                    'n_samples': n
                })
            else:
                print(f"Warning: Insufficient valid data for marker {marker}. Skipping.")
    
    return pd.DataFrame(results)

def create_combined_effect_plot(results, single_markers_df, adata, figsize=(12, 0.3), show_significant_only=False):
    # Extract interaction coefficients and standard errors for longitudinal markers
    coef_data = []
    for marker, model in results.items():
        coef = model.params['age_centered:fitness_centered']
        std_err = model.bse['age_centered:fitness_centered']
        lower = coef - 1.96 * std_err
        upper = coef + 1.96 * std_err
        is_significant = (lower > 0) or (upper < 0)
        coef_data.append({
            'marker': marker,
            'coef': coef,
            'lower': lower,
            'upper': upper,
            'abs_coef': abs(coef),
            'is_significant': is_significant,
            'is_longitudinal': True,
            'n_samples': model.nobs
        })
    
    # Combine longitudinal and single marker results
    df = pd.DataFrame(coef_data)
    df = pd.concat([df, single_markers_df])
    
    # Filter for significant interactions if requested
    if show_significant_only:
        df = df[df['is_significant']]
    
    df = df.sort_values('abs_coef', ascending=False)  # Sort by absolute coefficient value
    
    # Adjust figure size based on number of markers to display
    adjusted_figsize = (figsize[0], max(figsize[1] * len(df), 5))  # Ensure a minimum height
    
    # Create the plot
    fig, ax = plt.subplots(figsize=adjusted_figsize)
    
    # Plot the coefficients and CIs
    for i, row in df.iterrows():
        color = 'blue' if row['is_longitudinal'] else 'green'
        ax.scatter(row['coef'], i, color=color, s=50, zorder=2)
        ax.hlines(i, row['lower'], row['upper'], color=color, alpha=0.5, linewidth=2, zorder=1)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, zorder=0)
    
    # Customize the plot
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{row['marker']} (n={row['n_samples']})" for _, row in df.iterrows()])
    ax.set_xlabel('Effect Size (Interaction Coefficient for Longitudinal / Correlation for Single Time Point)')
    title = 'Effect Sizes with 95% Confidence Intervals\n(Ordered by Absolute Effect Size)'
    if show_significant_only:
        title += '\nSignificant Effects Only'
    ax.set_title(title)
    
    # Add legend
    ax.scatter([], [], color='blue', label='Longitudinal (Interaction)')
    ax.scatter([], [], color='green', label='Single Time Point (Correlation)')
    ax.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    return df

# Analyze single markers
single_markers_df = analyze_single_markers(adata, single_markers)

# Create combined plot
all_effects_df = create_combined_effect_plot(results, single_markers_df, adata, show_significant_only=False)

# Create plot with only significant effects
significant_effects_df = create_combined_effect_plot(results, single_markers_df, adata, show_significant_only=True)

# Print summary statistics
num_longitudinal = all_effects_df['is_longitudinal'].sum()
num_single = len(all_effects_df) - num_longitudinal
num_significant = significant_effects_df.shape[0]

print(f"\nTotal number of markers analyzed: {len(all_effects_df)}")
print(f"Number of longitudinal markers: {num_longitudinal}")
print(f"Number of single time point markers: {num_single}")
print(f"Number of significant effects: {num_significant}")

# Print the top 10 significant markers with the strongest effects
print("\nTop 10 significant markers with strongest effects:")
print(significant_effects_df[['marker', 'coef', 'lower', 'upper', 'is_longitudinal', 'n_samples']].head(10))
# %%
