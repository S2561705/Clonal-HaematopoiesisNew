# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

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

# %%
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

def analyze_cohorts_with_age(data, marker_map):
    results = []
    
    for key in marker_map.global_name:
        try:
            
            mask = ~np.isnan(data.layers[key]).all(axis=1)
            filtered_data = data[mask, :]

            # Concatenate data
            key_mean_data = np.nanmean(filtered_data.layers[key], axis=1)
            cohort_data = np.array(filtered_data.obs.cohort)
            age_data = np.array(filtered_data.obs.age_wave_1)  # Assuming age is stored in data.obs.age
            
            # Check if we have at least two cohorts
            unique_cohorts = np.unique(cohort_data)
            if len(unique_cohorts) < 2:
                print(f"Skipping {key}: Only one cohort after filtering")
                continue
            
            # Create a DataFrame for the ANCOVA
            df = pd.DataFrame({
                'value': key_mean_data,
                'cohort': cohort_data,
                'age': age_data
            })
             
            # Perform ANCOVA
            model = ols('value ~ C(cohort) + age', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Extract results
            f_statistic = anova_table.loc['C(cohort)', 'F']
            p_value = anova_table.loc['C(cohort)', 'PR(>F)']
            
            # Calculate effect size (partial eta-squared)
            ssn = anova_table.loc['C(cohort)', 'sum_sq']
            sse = anova_table.loc['Residual', 'sum_sq']
            eta_squared = ssn / (ssn + sse)
            
            results.append({
                'key': key,
                'f_statistic': f_statistic,
                'p_value': p_value,
                '-log10(p-value)': -np.log10(p_value),
                'effect_size': eta_squared,
                'significant': p_value < 0.05
            })
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
    
    # results_df['-log10(p-value)'] = results_df['-log10(p-value)'].replace(np.inf, 200)

    return pd.DataFrame(results)

def plot_results(results_df):
    if results_df.empty:
        print("No results to plot. All keys were skipped or encountered errors.")
        return
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=results_df, x='effect_size', y='-log10(p-value)', 
                    hue='significant', size='significant',
                    sizes={True: 100, False: 50}, alpha=0.7)
    
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    plt.title('Effect Size vs Significance of Cohort Differences (Controlling for Age)')
    plt.xlabel('Effect Size (Partial Eta-squared)')
    plt.ylabel('-log10(p-value)')
    
    for _, row in results_df.iterrows():
        if row['significant']:
            plt.annotate(row['key'], (row['effect_size'], -np.log10(row['p_value'])),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Example usage:
results_df = analyze_cohorts_with_age(data, marker_map)
plot_results(results_df)
# %%

key = 'BLD_HAEMATOCRIT'

def z_score_by_cohort(adata, layer_key, cohort_column='cohort'):
    """
    Convert values in adata.layers[layer_key] to z-scores, calculated separately for each cohort.
    
    Parameters:
    - adata: AnnData object
    - layer_key: String, key for the layer to be normalized
    - cohort_column: String, column name in adata.obs containing cohort information
    
    Returns:
    - None (modifies adata in-place)
    """
    for cohort in adata.obs[cohort_column].unique():
        mask = adata.obs[cohort_column] == cohort
        cohort_data = adata[mask].layers[layer_key]
        
        cohort_mean = np.nanmean(cohort_data)
        cohort_std = np.nanstd(cohort_data)
        
        adata.layers[layer_key][mask] = (cohort_data - cohort_mean) / cohort_std

    print(f"Z-score normalization completed for {key} in {cohort_column}")
# %%

for key in marker_map.global_name:
    z_score_by_cohort(data, key)

# %%
key = 'BLD_HBA2'
sns.histplot(x=np.nanmean(data.layers[key], axis=1), hue=data.obs.cohort)

# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def longitudinal_analysis(data, key, marker):
    # Reshape the data
    df = pd.DataFrame({
        'participant_id': np.repeat(data.obs.participant_id, 8),
        'age': data.X.flatten(),
        'measurement': data.layers[key].flatten(),
        'marker': np.repeat(data.obs[marker], 8)
    })

    # Remove rows with NaN values
    df = df.dropna(subset=['age', 'measurement']).copy()
    if len(df) == 0:
        return 
    
    # Prepare the data
    df['participant_id'] = df['participant_id'].astype('category')

    # Center variables for easier interpretation of main effects
    df['age_centered'] = df['age']
    df['marker_centered'] = (df['marker']- df['marker'].mean()) / df['marker'].std()

    # Specify the model formula
    model_formula = "measurement ~ age_centered + marker_centered + age_centered:marker_centered"

    # Fit the Linear Mixed-Effects Model
    mixed_model = smf.mixedlm(formula=model_formula, data=df, groups='participant_id')

    # Fit the model
    try:
        results = mixed_model.fit()
        
        # Extract relevant information
        interaction_coef = results.params['age_centered:marker_centered']
        interaction_pvalue = results.pvalues['age_centered:marker_centered']
        ci = results.conf_int().loc['age_centered:marker_centered']
        
        return {
            'key': key,
            'interaction_coef': interaction_coef,
            'p_value': interaction_pvalue,
            '-log10(p-value)': -np.log10(interaction_pvalue),
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    except Exception as e:
        print(f"Error fitting model for {key}: {str(e)}")
        return None


# Usage
results = []
for key in marker_map.global_name:
    result = longitudinal_analysis(data, key, 'max_fitness')  # Assuming 'max_fitness' is the marker you want to use
    if result:
        results.append(result)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='p_value')
results_df = results_df[results_df.p_value<0.05]
print(results_df)

# %%
# Plotting code (as before)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.errorbar(results_df.index, results_df['interaction_coef'], 
             yerr=[results_df['interaction_coef'] - results_df['ci_lower'], 
                   results_df['ci_upper'] - results_df['interaction_coef']],
             fmt='o', capsize=5, capthick=2, ecolor='gray', alpha=0.5)

scatter = plt.scatter(results_df.index, results_df['interaction_coef'], 
                      c=results_df['p_value'], cmap='viridis_r', 
                      s=100, zorder=5)

plt.colorbar(scatter, label='p-value')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xticks(results_df.index, results_df['key'], rotation=90)
plt.ylabel('Interaction Coefficient (Age * Max Fitness)')
plt.title('Interaction Coefficients with 95% CI for Each Marker')

plt.tight_layout()
plt.show()
# %%
results_df.sort_values(by='p_value')