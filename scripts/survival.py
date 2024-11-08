# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
import matplotlib as mpl
import anndata as ad

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import statsmodels.formula.api as smf

cohort_color_dict = {'sardiNIA':  '#f8cbad',
                    'WHI':  '#9dc3e6' ,
                    'LBC':  '#70ad47'}

# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)
# %%
# LBC Dataset
processed_lbc = [part for part in cohort if 'LBC' in part.uns['cohort']]

lbc21_meta = pd.read_csv('../data/LBC/LBC1921_ExtraVariables.csv')
lbc36_meta = pd.read_csv('../data/LBC/LBC1936_ExtraVariables.csv')

lbc21_meta = lbc21_meta.rename(columns={'studyno': 'ID',
                      'agedaysApx_LastCensor':'agedays_lastcensor'})

lbc36_meta = lbc36_meta.rename(columns={'lbc36no': 'ID',
                      'AgedaysApx_LastCensor':'agedays_lastcensor'})

lbc_df = pd.merge(lbc21_meta, lbc36_meta, how='outer')

# filter merged dataset to only fitted participants
lbc_df = lbc_df[lbc_df.ID.isin([part.uns['participant_id'] for part in processed_lbc])].copy()

age_wave_1_dict = {part.uns['participant_id']:part.var.time_points.min() for part in processed_lbc}
lbc_df['age_wave_1'] = lbc_df['ID'].map(age_wave_1_dict)

# find maximum between death and censorhip
lbc_df['ageyrs_lastcensor'] = lbc_df['agedays_lastcensor']/365.2422

lbc_df['age_death'] = lbc_df[['ageyrs_death', 'ageyrs_lastcensor']].max(axis=1)
lbc_df['from_wave_1'] = lbc_df['age_death'] - lbc_df['age_wave_1']
lbc_df['cohort'] = 'LBC'
lbc_df['death_cause'] = 'NaN'
lbc_df['death_cause_num'] = np.nan
# %%

# Uddin Dataset
processed_WHI = [part for part in cohort if part.uns['cohort']=='WHI']

# load survival data
# aging_df = pd.read_csv('../data/Uddin/outc_aging_ctos_inv.dat', sep='\t')
WHI_df = pd.read_csv('../data/WHI/outc_death_all_discovered_inv.dat', sep='\t')

with open('../resources/WHI_death_cause.json', 'r') as f:
    death_cause_dict = json.load(f)

death_cause_dict = {int(k):v for k,v in death_cause_dict.items()}
# filter survival data to sequencing ids
ids = [part.uns['participant_id'] for part in processed_WHI]
# aging_df = aging_df[aging_df.ID.isin(ids)].copy()
WHI_df = WHI_df[WHI_df.ID.isin([part.uns['participant_id'] for part in processed_WHI])].copy()

WHI_df = WHI_df.rename(columns={'DEATHALL': 'dead',
                            'DEATHALLCAUSE': 'death_cause_num',
                            'ENDFOLLOWALLDY': 'from_wave_1_days'})

# create age_wave_1 dict
age_wave_1_dict = {part.uns['participant_id']:part.var.time_points.min() for part in processed_WHI}
WHI_df['age_wave_1'] = WHI_df['ID'].map(age_wave_1_dict)

WHI_df['from_wave_1'] = WHI_df['from_wave_1_days']/365.2422
WHI_df['age_death'] = WHI_df['age_wave_1'] + WHI_df['from_wave_1']
# Filter WHI_df and add cause of death
WHI_df = WHI_df[['ID', 'dead', 'age_death', 'death_cause_num', 'from_wave_1', 'age_wave_1']].copy()
WHI_df['death_cause'] = WHI_df['death_cause_num'].map(death_cause_dict)
WHI_df['cohort'] = 'WHI'

# %%

# sardiNIA
processed_sardinia = [part for part in cohort if part.uns['cohort']=='sardiNIA']

sardinia_df = pd.read_csv('../data/sardiNIA/fabre_deaths upd 2024.csv')


sardinia_df = sardinia_df[sardinia_df.ID.isin([
                              part.uns['participant_id']
                              for part in processed_sardinia])].copy()

sardinia_df['dead'] = ~np.isnan(sardinia_df.Death_AGE)*1
sardinia_df['age_death'] = np.nanmax(
    sardinia_df[['AGE1', 'AGE2', 'AGE3',
    'AGE4', 'AGE5', 'Death_AGE']],
    axis=1)

# create age_wave_1 dict
age_wave_1_dict = {part.uns['participant_id']:part.var.time_points.min() for part in processed_sardinia}
sardinia_df['age_wave_1'] = sardinia_df['ID'].map(age_wave_1_dict)

# sardinia_df['age_wave_1'] = np.nanmin(
#     sardinia_df[['AGE1', 'AGE2', 'AGE3',
#     'AGE4', 'AGE5', 'Death_AGE']],
#     axis=1)

sardinia_df['dead'] = sardinia_df['dead'].astype('int')
sardinia_df['from_wave_1'] = (sardinia_df['age_death'] 
            - sardinia_df['age_wave_1'])

sardinia_df['cohort'] = 'sardiNIA'
sardinia_df['death_cause'] = 'NaN'
sardinia_df['death_cause_num'] = np.nan

# %%
# Merge cohorts and append fitness information

keep_columns=['ID', 'dead', 'from_wave_1', 'cohort', 'death_cause',
              'age_death', 'age_wave_1']
data_frames = [lbc_df[keep_columns],
               WHI_df[keep_columns],
               sardinia_df[keep_columns]]


merged_survival = pd.concat(data_frames)

cohort_list = [processed_lbc, processed_WHI, processed_sardinia]
cohort_name = ['LBC', 'WHI', 'sardiNIA']

# extract clonal information for each participant
max_fitness_dict = dict()
max_vaf_dict = dict()
max_clonal_grad_dict = dict()

for name, cohort in  zip(cohort_name, cohort_list):
    for part in cohort:
        dict_id = (name, part.uns['participant_id'])
        
        # maximum fitness
        max_fitness_dict[dict_id] = part.obs.fitness.max()
        
        # max vaf at first time point
        max_vaf_dict[dict_id] = part.X[:,0].max()
        
        # maximum predicted clonal gradient at first time point 
        max_clonal_grad_dict[dict_id] = np.array(part.obs.fitness*part.X[:,0]).max()


merged_survival['max_fitness'] = merged_survival.apply(lambda x: 
                max_fitness_dict[(x['cohort'], 
                                       x['ID'])], axis=1)

merged_survival['max_vaf'] = merged_survival.apply(lambda x: 
                max_vaf_dict[(x['cohort'], 
                                       x['ID'])], axis=1) 

merged_survival['max_clonal_grad'] = merged_survival.apply(lambda x: 
                max_clonal_grad_dict[(x['cohort'], 
                                       x['ID'])], axis=1) 


# %%


survival_df = merged_survival[merged_survival.cohort.isin(cohort_choice)].copy()
survival_df['norm_max_clonal_grad'] = (survival_df['max_clonal_grad']-survival_df['max_clonal_grad'].mean())/survival_df['max_clonal_grad'].std()
survival_df['norm_max_vaf'] = (survival_df['max_vaf']- survival_df['max_vaf'].mean())/survival_df['max_clonal_grad'].std()
survival_df['norm_max_fitness'] = (survival_df['max_fitness']-survival_df['max_fitness'].mean())/survival_df['max_fitness'].std()

ylabel_dict = {'norm_max_vaf': 'Max VAF\n(scaled)',
               'norm_max_fitness':'Max Fitness\n(scaled)',
               'age_wave_1': 'Age'}

survival_df.to_csv('../exports/survival_all.csv')

# %%

cohort_choice = ['LBC', 'WHI', 'sardiNIA']

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(survival_df, 
        duration_col='from_wave_1',
        event_col='dead',
        formula="age_wave_1 +  norm_max_vaf +  norm_max_fitness")

cph.print_summary()
# %%
# Get the summary data
summary = cph.summary

summary = summary.loc[['age_wave_1', 'norm_max_vaf', 'norm_max_fitness']]
summary = summary.sort_values(by='exp(coef)')

# Create a new figure with a larger size
fig, ax = plt.subplots()  # Reduced height for closer lines

# Calculate y positions
y_pos = range(len(summary))

plt.errorbar(summary['coef'], y_pos,
             xerr=[summary['coef'] - summary['coef lower 95%'],
                   summary['coef upper 95%'] - summary['coef']],
             fmt='o', capsize=5, color=sns.color_palette('tab10')[0])

# Add labels and title
plt.xlabel('log HR (95% CI)')
plt.title('Combined Cohorts')

# Add a vertical line at x=0
ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)


summary.index
# Set y-ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(summary.index.map(ylabel_dict))
sns.despine()
# Adjust layout
plt.tight_layout()
plt.savefig('../results/survival/combined cohorts.png', dpi=1000)
plt.show()
plt.clf()
# %%

fig, ax = plt.subplots()
# Define the fitness values you want to plot
fitness_values = [-2, 0, 2] # 5 distinct levels of fitness

# Plot partial effects with confidence intervals
for i, value in enumerate(fitness_values):
    color = sns.color_palette('deep')[(i+6)%10]
    cph.plot_partial_effects_on_outcome(covariates='norm_max_fitness', 
                                        values=[value],
                                        plot_baseline=False,
                                        ax=ax,
                                        color=color,
                                        label=f'Fitness: {value:.2f}')
    
ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel('Survival Years')
plt.ylabel('Proportion Survivors (%)')
plt.title('Partial Effect of Fitness on Survival')

plt.legend(['-2 SD', 'baseline', '2 SD'], loc='best')
sns.despine()
plt.savefig('../results/survival/partial_effects_on_outcome_fitness.png', dpi=1000)
plt.show()
plt.clf()
# %%

# Show survival discrepancy on dead people
dead_df = survival_df[survival_df.dead == 1]
sns.regplot(dead_df, x='age_wave_1', y='from_wave_1', scatter=False)
sns.scatterplot(dead_df, x='age_wave_1', y='from_wave_1', hue='cohort',
                palette=cohort_color_dict)

plt.title('Correlation between Age and Years to Death')
plt.xlabel('Age')
plt.ylabel('Years to Death')

sns.despine()
plt.savefig('../results/survival/corr_age_years_to_death_by_cohort.png', dpi=1000)

# %%

from scipy.stats import linregress
age_regress = linregress(dead_df['age_wave_1'], dead_df['from_wave_1'])

dead_df['survival_residual'] = dead_df['from_wave_1'] - (dead_df['age_wave_1']*age_regress.slope + age_regress.intercept)
survival_df['survival_residual'] = survival_df['from_wave_1'] - (survival_df['age_wave_1']*age_regress.slope + age_regress.intercept)
sns.regplot(dead_df, 
    x='survival_residual',
    y='max_fitness',
    scatter=False,
    line_kws={'color':sns.color_palette('tab10')[3]})

sns.scatterplot(dead_df, 
    x='survival_residual',
    y='max_fitness', hue='cohort', alpha=0.7, palette=cohort_color_dict)

plt.ylabel('Maximum Fitness')
plt.xlabel('Survival Discrepancy (Years)')
plt.title('Correlation between Fitness and Survival Discrepancy')
sns.despine()

# save survival figure
plt.savefig('../results/survival/fitness_vs_survival_discrepancy.png')
plt.show()
plt.clf()
# %%



# Compute residual ages for full cohort
survival_df['survival_residual'] = survival_df['from_wave_1'] - (survival_df['age_wave_1']*age_regress.slope + age_regress.intercept)

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(survival_df, 
        duration_col='survival_residual',
        event_col='dead',
        formula="norm_max_fitness")

cph.print_summary()  # Access the individual results using cph.summary

# Plot the coefficients
summary = cph.summary

summary = summary.loc[['norm_max_fitness']]
summary = summary.sort_values(by='exp(coef)')

# Create a new figure with a larger size
fig, ax = plt.subplots(figsize=(5,2))  # Reduced height for closer lines

# Calculate y positions
y_pos = range(len(summary))

plt.errorbar(summary['coef'], y_pos,
             xerr=[summary['coef'] - summary['coef lower 95%'],
                   summary['coef upper 95%'] - summary['coef']],
             fmt='o', capsize=5, color=sns.color_palette('tab10')[0])

# Add labels and title
plt.xlabel('log HR (95% CI)')
plt.title('Combined Cohorts')

# Add a vertical line at x=0
ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)


summary.index
# Set y-ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(summary.index.map(ylabel_dict))
sns.despine()
# Adjust layout
plt.tight_layout()
plt.savefig('../results/survival/survival_discrepancy_combined.png', dpi=1000)
plt.show()
plt.clf()

# %%

fig, ax = plt.subplots()
# Define the fitness values you want to plot
fitness_values = [-2, 0, 2] # 5 distinct levels of fitness

# Plot partial effects with confidence intervals
for i, value in enumerate(fitness_values):
    color = sns.color_palette('tab10')[(i+6)%10]
    cph.plot_partial_effects_on_outcome(covariates='norm_max_fitness', 
                                        values=[value],
                                        plot_baseline=False,
                                        ax=ax,
                                        color=color,
                                        label=f'Fitness: {value:.2f}')
    
ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel('Survival Discrepancy (Years)')
plt.ylabel('Proportion Survivors (%)')
plt.title('Fitness on Survival Discrepancy')

plt.legend(['-2 SD', 'baseline', '2 SD'], loc='best')
sns.despine()
plt.savefig('../results/survival/Fitness on survival discrepancy.png', dpi=1000)
plt.show()
plt.clf()
# %%
# Create fitness categories
survival_df['fitness_category'] = pd.cut(survival_df['norm_max_fitness'], bins=[-2,0,2, np.inf], labels=['-2 SD', 'Baseline', '2 SD'])

for i, category in enumerate(survival_df.fitness_category.unique()):
    mask = survival_df['fitness_category'] == category
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_df.loc[mask, 'survival_residual'], 
            event_observed=survival_df.loc[mask, 'dead'], 
            label=f'Fitness {category}')
    kmf.plot(color=sns.color_palette()[(i+6)%10])

plt.title('Kaplan-Meier Survival Curves by Fitness Level')
plt.xlabel('Survival Discrepancy (Years)')
plt.ylabel('Survival Probability')
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()

sns.despine()
plt.savefig('../results/survival/kaplan_meyer_discrepancy_by_fitness_range.png')
plt.show()
plt.clf()
# %%


# survival by cohort

cohort_choice = ['LBC', 'WHI', 'sardiNIA']
# Initialize a dictionary to store results
results = {}
# Fit separate models for each cohort
for cohort in cohort_choice:
    cohort_data = merged_survival[merged_survival['cohort'] == cohort].copy()
    cohort_data['norm_max_clonal_grad'] = (cohort_data['max_clonal_grad']-cohort_data['max_clonal_grad'].mean())/cohort_data['max_clonal_grad'].std()
    cohort_data['norm_max_vaf'] = (cohort_data['max_vaf']- cohort_data['max_vaf'].mean())/cohort_data['max_clonal_grad'].std()
    cohort_data['norm_max_fitness'] = (cohort_data['max_fitness']-cohort_data['max_fitness'].mean())/cohort_data['max_fitness'].std()
    
    cph = CoxPHFitter()
    cph.fit(cohort_data, duration_col='from_wave_1', event_col='dead', 
            formula="age_wave_1 + norm_max_vaf + norm_max_fitness")
    results[cohort] = cph.summary

# Combine results
combined_results = pd.concat(results.values(), keys=results.keys(), names=['Cohort', 'Covariate'])
combined_results = combined_results.reset_index()

combined_results = combined_results.sort_values(by='coef', ascending=True)

# Calculate y-positions to avoid overlap
y_positions = {cov: i for i, cov in enumerate(combined_results['Covariate'].unique())}
offset = 0.2  # Offset for different cohorts

fig, ax = plt.subplots()
for i, cohort in enumerate(cohort_choice):
    cohort_data = combined_results[combined_results['Cohort'] == cohort]
    y_pos = [y_positions[cov] + (i - 0.5) * offset for cov in cohort_data['Covariate']]
    
    plt.errorbar(cohort_data['coef'], y_pos, 
                 xerr=[cohort_data['coef'] - cohort_data['coef lower 95%'],
                       cohort_data['coef upper 95%'] - cohort_data['coef']],
                 fmt='o', capsize=5, label=cohort, color=cohort_color_dict[cohort])

sns.despine()

plt.axvline(x=0, color='red', linestyle='--')
ax.grid(axis='x', color='grey', linestyle='--', alpha=0.5)
plt.xlabel('log HR (95% CI)')

plt.yticks(list(y_positions.values()), list(y_positions.keys()))
ax.set_yticklabels([ylabel_dict[cov] for cov in combined_results.Covariate.unique()])

plt.ylabel('Covariate')
plt.title('Survival HR by Cohort')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('../results/survival/survival_by_cohort.png')
plt.show()
plt.clf()
# %%

# Survival Analysis by age bins

# Step 1: Create age bins
merged_survival['age_bin'] = pd.cut(merged_survival['age_wave_1'], 
                                    bins=[49, 60, 70, 80, np.inf], 
                                    labels=['50-59', '60-69', '70-79', '80+'])
results = {}

for age_group in merged_survival['age_bin'].unique():
    group_data = merged_survival[merged_survival['age_bin'] == age_group].copy()
    group_data['norm_max_clonal_grad'] = (group_data['max_clonal_grad']-group_data['max_clonal_grad'].mean())/group_data['max_clonal_grad'].std()
    group_data['norm_max_vaf'] = (group_data['max_vaf']- group_data['max_vaf'].mean())/group_data['max_clonal_grad'].std()
    group_data['norm_max_fitness'] = (group_data['max_fitness']-group_data['max_fitness'].mean())/group_data['max_fitness'].std()
    
    
    cph = CoxPHFitter()
    cph.fit(group_data, duration_col='from_wave_1', event_col='dead', 
            formula="norm_max_vaf + norm_max_fitness")
    
    results[age_group] = cph.summary

# Combine results
combined_results = pd.concat(results.values(), keys=results.keys(), names=['Age Group', 'Covariate'])
combined_results = combined_results.reset_index()
combined_results.sort_values(by='Age Group', ascending=False)

bin_to_y_offfset_dict = {'80+': 0.2,
                 '70-79':0.1,
                  '60-69':0,
                '50-59':-0.1}
y_positions = {cov: i for i, cov in enumerate(combined_results['Covariate'].unique())}
offset = 0.1  # Offset for different cohorts

fig , ax = plt.subplots()
for i, group in enumerate(combined_results['Age Group'].unique()):
    group_data = combined_results[combined_results['Age Group'] == group]
    y_pos = [y_positions[cov] + bin_to_y_offfset_dict[group] for cov in group_data['Covariate']]
    
    plt.errorbar(group_data['coef'], y_pos, 
                 xerr=[group_data['coef'] - group_data['coef lower 95%'],
                       group_data['coef upper 95%'] - group_data['coef']],
                 fmt='o', capsize=5,
                 label=group,
                 color=sns.color_palette('tab10')[(i+5)%10])
    
plt.xlabel('log HR (95% CI)')
plt.yticks(list(y_positions.values()), list(y_positions.keys()))

ax.set_yticklabels([ylabel_dict[cov] for cov in combined_results.Covariate.unique()])
plt.ylabel('Covariate')
plt.title('Survival HR by Age Group')
plt.tight_layout()

# Get the current handles and labels
handles, labels = ax.get_legend_handles_labels()

# Define a custom sorting key function
def sort_key(label):
    if label == '80+':
        return 80
    else:
        return int(label.split('-')[0])

# Sort the handles and labels
sorted_pairs = sorted(zip(handles, labels), key=lambda pair: sort_key(pair[1]), reverse=True)
handles, labels = zip(*sorted_pairs)

# Create the legend with the sorted handles and labels
ax.legend(handles, labels)

# plt.legend(combined_results['Age Group'].unique(), loc='best')

plt.axvline(x=0, color='red', linestyle='--')
ax.grid(axis='x', color='grey', linestyle='--', alpha=0.5)
plt.xlabel('log HR (95% CI)')

sns.despine()
plt.savefig('../results/survival/survival_by_age_bin.png')
plt.show()
plt.clf()
# %%
# cause of death analysis
survival_WHI = merged_survival[merged_survival.cohort=='WHI'].copy()

# Analize death cause
cause_df = survival_WHI[survival_WHI['death_cause'].notna()].copy()
death_cause_counts = cause_df.death_cause.value_counts().to_dict()

results = []
for cause in cause_df.death_cause.unique():
    if death_cause_counts[cause] < 2:
        continue
    cause_num = cause + f' ({death_cause_counts[cause]})'

    if cause in ['Lymphoma (NHL only)', 'Unknown Cause']:
        continue
    cause_df['cause'] = (cause_df['death_cause'].isin([cause]))*1


    model_logit = smf.logit(formula="cause ~ scale(max_fitness) + age_wave_1", data=cause_df)
    res = model_logit.fit()

    conf_int = res.conf_int().loc['scale(max_fitness)']
    beta = res.params['scale(max_fitness)']

    # Append the results to the list
    results.append({
        'cause': cause_num,
        'beta': beta,
        'conf_low': conf_int[0],
        'conf_high': conf_int[1]
    })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)
results_df  = results_df.sort_values(by='beta')

fig, ax = plt.subplots()  # Reduced height for closer lines

# Create a seaborn plot
plt.errorbar(results_df['beta'],
             results_df['cause'],
             xerr=(results_df['beta'] - results_df['conf_low'],
                   results_df['conf_high'] - results_df['beta']),
                   fmt='o', capsize=3,
                   color=sns.color_palette('tab10')[0],
            )

# Add a dotted vertical line at x=0
plt.axvline(x=0, color='red', linestyle='--')
ax.grid(axis='x', color='grey', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel('Fitness Effect (Beta Value)')
plt.ylabel('Death Cause (number of observations)')
plt.title('Correlation Between Fitness and Death Cause')
sns.despine()
plt.tight_layout()
plt.savefig('../results/survival/fitness_vs_death_cause.png', dpi=1000)
plt.show()
plt.clf()
# %%


sns.scatterplot(survival_df, x='age_wave_1', y='age_death', hue='dead')
# %%

