# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.aux import *

import pandas as pd
import matplotlib as mpl
import anndata as ad

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import statsmodels.formula.api as smf


survival_df = pd.read_csv('../results/participant_df.csv', index_col=0)

# %%

cohort_color_dict = {'sardiNIA':  '#f8cbad',
                    'WHI':  '#9dc3e6' ,
                    'LBC':  '#70ad47'}
survival_df = survival_df[survival_df['dead'].notna()]

ylabel_dict = {'max_VAF_z_score': 'Max VAF',
               'max_fitness_z_score':'Max Fitness',
               'age_wave_1': 'Age',
               'Female': 'Sex',
               'max_size_prediction_120_z_score': 'Maximum clone size \n predicted at age 120',
               'max_size_prediction_next_30_z_score': 'Maximum clone size predicted in 30 years'}



# %%

cohort_choice = ['LBC', 'WHI', 'sardiNIA']

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(survival_df, 
        duration_col='from_wave_1',
        event_col='dead',
        formula="max_VAF_z_score + max_fitness_z_score + age_wave_1 + Female + max_size_prediction_120_z_score")
      #   formula="max_VAF_z_score + max_fitness_z_score + age_wave_1 + Female ")
      #   formula="max_VAF_z_score + max_size_prediction_120_z_score + age_wave_1 + Female ")

cph.print_summary()

# Get the summary data
summary = cph.summary


summary = summary.loc[summary.index.to_list()]
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

covariate = 'max_size_prediction_120_z_score'
fig, ax = plt.subplots()
# Define the fitness values you want to plot
fitness_values = [-2, 0, 2] # 5 distinct levels of fitness

# Plot partial effects with confidence intervals
for i, value in enumerate(fitness_values):
    color = sns.color_palette('deep')[(i+6)%10]
    cph.plot_partial_effects_on_outcome(covariates=covariate, 
                                        values=[value],
                                        plot_baseline=False,
                                        ax=ax,
                                        color=color,
                                        label=f'Fitness: {value:.2f}')
    
ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel('Survival Years')
plt.ylabel('Proportion Survivors (%)')
plt.title(f'Partial Effect of {ylabel_dict[covariate]} on Survival')

plt.legend(['-2 SD', 'baseline', '2 SD'], loc='best')
sns.despine()
plt.savefig(f'../results/survival/partial_effects_on_outcome_{ylabel_dict[covariate]}.png', dpi=1000)
plt.show()
plt.clf()
# %%
# Age vs norm_log_max_size_prediction_120
sns.regplot(survival_df, x='age_wave_1', y='max_size_prediction_120_z_score', scatter=False)
sns.scatterplot(survival_df, x='age_wave_1', y='max_size_prediction_120_z_score', hue='cohort',
                palette=cohort_color_dict)

plt.title('Correlation between Age and Predicted maximum clone size at 120')
plt.xlabel('Age')
plt.ylabel('Predited clone size \n (log and scaled)')

sns.despine()
plt.savefig('../results/survival/corr_age_years_to_predicted_clone_size.png', dpi=1000)
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
# %%
from scipy.stats import linregress
age_regress = linregress(dead_df['age_wave_1'], dead_df['from_wave_1'])

dead_df['survival_residual'] = dead_df['from_wave_1'] - (dead_df['age_wave_1']*age_regress.slope + age_regress.intercept)
survival_df['survival_residual'] = survival_df['from_wave_1'] - (survival_df['age_wave_1']*age_regress.slope + age_regress.intercept)
sns.regplot(dead_df, 
    x='survival_residual',
    y='max_size_prediction_120_z_score',
    scatter=False,
    line_kws={'color':sns.color_palette('tab10')[3]})

sns.scatterplot(dead_df, 
    x='survival_residual',
    y='max__120_z_score', hue='cohort', alpha=0.7, palette=cohort_color_dict)

plt.ylabel('Predited clone size \n (log and scaled)')
plt.xlabel('Survival Discrepancy (Years)')
plt.title('Correlation between Fitness and Survival Discrepancy')
sns.despine()

# save survival figure
plt.savefig('../results/survival/fitness_vs_survival_discrepancy.png')
plt.show()
plt.clf()
# # %%

# # Compute residual ages for full cohort
# survival_df['survival_residual'] = survival_df['from_wave_1'] - (survival_df['age_wave_1']*age_regress.slope + age_regress.intercept)

# # Fit the Cox proportional hazards model
# cph = CoxPHFitter()
# cph.fit(survival_df, 
#         duration_col='survival_residual',
#         event_col='dead',
#         formula="norm_max_fitness")

# cph.print_summary()  # Access the individual results using cph.summary

# # Plot the coefficients
# summary = cph.summary

# summary = summary.loc[['norm_max_fitness']]
# summary = summary.sort_values(by='exp(coef)')

# # Create a new figure with a larger size
# fig, ax = plt.subplots(figsize=(5,2))  # Reduced height for closer lines

# # Calculate y positions
# y_pos = range(len(summary))

# plt.errorbar(summary['coef'], y_pos,
#              xerr=[summary['coef'] - summary['coef lower 95%'],
#                    summary['coef upper 95%'] - summary['coef']],
#              fmt='o', capsize=5, color=sns.color_palette('tab10')[0])

# # Add labels and title
# plt.xlabel('log HR (95% CI)')
# plt.title('Combined Cohorts')

# # Add a vertical line at x=0
# ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)
# plt.axvline(x=0, color='red', linestyle='--', linewidth=2)


# summary.index
# # Set y-ticks and labels
# ax.set_yticks(y_pos)
# ax.set_yticklabels(summary.index.map(ylabel_dict))
# sns.despine()
# # Adjust layout
# plt.tight_layout()
# plt.savefig('../results/survival/survival_discrepancy_combined.png', dpi=1000)
# plt.show()
# plt.clf()

# # %%

# fig, ax = plt.subplots()
# # Define the fitness values you want to plot
# fitness_values = [-2, 0, 2] # 5 distinct levels of fitness

# # Plot partial effects with confidence intervals
# for i, value in enumerate(fitness_values):
#     color = sns.color_palette('tab10')[(i+6)%10]
#     cph.plot_partial_effects_on_outcome(covariates='norm_max_fitness', 
#                                         values=[value],
#                                         plot_baseline=False,
#                                         ax=ax,
#                                         color=color,
#                                         label=f'Fitness: {value:.2f}')
    
# ax.grid(axis='both', color='grey', linestyle='--', alpha=0.5)

# # Add labels and title
# plt.xlabel('Survival Discrepancy (Years)')
# plt.ylabel('Proportion Survivors (%)')
# plt.title('Fitness on Survival Discrepancy')

# plt.legend(['-2 SD', 'baseline', '2 SD'], loc='best')
# sns.despine()
# plt.savefig('../results/survival/Fitness on survival discrepancy.png', dpi=1000)
# plt.show()
# plt.clf()

# # %%
# # Create fitness categories
# survival_df['fitness_category'] = pd.cut(survival_df['norm_max_fitness'], bins=[-2,0,2, np.inf], labels=['-2 SD', 'Baseline', '2 SD'])

# for i, category in enumerate(survival_df.fitness_category.unique()):
#     mask = survival_df['fitness_category'] == category
#     kmf = KaplanMeierFitter()
#     kmf.fit(durations=survival_df.loc[mask, 'survival_residual'], 
#             event_observed=survival_df.loc[mask, 'dead'], 
#             label=f'Fitness {category}')
#     kmf.plot(color=sns.color_palette()[(i+6)%10])

# plt.title('Kaplan-Meier Survival Curves by Fitness Level')
# plt.xlabel('Survival Discrepancy (Years)')
# plt.ylabel('Survival Probability')
# plt.grid(True, alpha=0.5)
# plt.legend()
# plt.tight_layout()

# sns.despine()
# plt.savefig('../results/survival/kaplan_meyer_discrepancy_by_fitness_range.png')
# plt.show()
# plt.clf()
# # %%

# # survival by cohort

# cohort_choice = ['LBC', 'WHI', 'sardiNIA']
# # Initialize a dictionary to store results
# results = {}
# # Fit separate models for each cohort
# for cohort in cohort_choice:
#     cohort_data = survival_df[survival_df['cohort'] == cohort].copy()
#     cohort_data['norm_max_VAF'] = (cohort_data['max_VAF']- cohort_data['max_VAF'].mean())/cohort_data['max_VAF'].std()
#     cohort_data['norm_max_fitness'] = (cohort_data['max_fitness']-cohort_data['max_fitness'].mean())/cohort_data['max_fitness'].std()
    
#     cph = CoxPHFitter()
#     cph.fit(cohort_data, duration_col='from_wave_1', event_col='dead', 
#             formula="age_wave_1 +  norm_max_VAF +  norm_max_fitness + Female+ norm_log_max_size_prediction_next_30 + norm_log_max_size_prediction_120")
#     results[cohort] = cph.summary

# # Combine results
# combined_results = pd.concat(results.values(), keys=results.keys(), names=['Cohort', 'Covariate'])
# combined_results = combined_results.reset_index()

# combined_results = combined_results.sort_values(by='coef', ascending=True)

# # Calculate y-positions to avoid overlap
# y_positions = {cov: i for i, cov in enumerate(combined_results['Covariate'].unique())}
# offset = 0.2  # Offset for different cohorts

# fig, ax = plt.subplots()
# for i, cohort in enumerate(cohort_choice):
#     cohort_data = combined_results[combined_results['Cohort'] == cohort]
#     y_pos = [y_positions[cov] + (i - 0.5) * offset for cov in cohort_data['Covariate']]
    
#     plt.errorbar(cohort_data['coef'], y_pos, 
#                  xerr=[cohort_data['coef'] - cohort_data['coef lower 95%'],
#                        cohort_data['coef upper 95%'] - cohort_data['coef']],
#                  fmt='o', capsize=5, label=cohort, color=cohort_color_dict[cohort])

# sns.despine()

# plt.axvline(x=0, color='red', linestyle='--')
# ax.grid(axis='x', color='grey', linestyle='--', alpha=0.5)
# plt.xlabel('log HR (95% CI)')

# plt.yticks(list(y_positions.values()), list(y_positions.keys()))
# ax.set_yticklabels([ylabel_dict[cov] for cov in combined_results.Covariate.unique()])

# plt.ylabel('Covariate')
# plt.title('Survival HR by Cohort')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('../results/survival/survival_by_cohort.png')
# plt.show()
# plt.clf()

# # %%
# # Survival Analysis by age bins

# # Step 1: Create age bins
# survival_df['age_bin'] = pd.cut(survival_df['age_wave_1'], 
#                                     bins=[49, 60, 70, 80, np.inf], 
#                                     labels=['50-59', '60-69', '70-79', '80+'])
# results = {}

# for age_group in survival_df['age_bin'].unique():
#     group_data = survival_df[survival_df['age_bin'] == age_group].copy()
#     group_data['norm_max_VAF'] = (group_data['max_VAF']- group_data['max_VAF'].mean())/group_data['max_VAF'].std()
#     group_data['norm_max_fitness'] = (group_data['max_fitness']-group_data['max_fitness'].mean())/group_data['max_fitness'].std()
    
    
#     cph = CoxPHFitter()
#     cph.fit(group_data, duration_col='from_wave_1', event_col='dead', 
#             formula="norm_max_VAF +  norm_max_fitness + Female + norm_log_max_size_prediction_next_30 + norm_log_max_size_prediction_120")
    
#     results[age_group] = cph.summary

# # Combine results
# combined_results = pd.concat(results.values(), keys=results.keys(), names=['Age Group', 'Covariate'])
# combined_results = combined_results.reset_index()
# combined_results.sort_values(by='Age Group', ascending=False)

# bin_to_y_offfset_dict = {'80+': 0.2,
#                  '70-79':0.1,
#                   '60-69':0,
#                 '50-59':-0.1}
# y_positions = {cov: i for i, cov in enumerate(combined_results['Covariate'].unique())}
# offset = 0.1  # Offset for different cohorts

# fig , ax = plt.subplots()
# for i, group in enumerate(combined_results['Age Group'].unique()):
#     group_data = combined_results[combined_results['Age Group'] == group]
#     y_pos = [y_positions[cov] + bin_to_y_offfset_dict[group] for cov in group_data['Covariate']]
    
#     plt.errorbar(group_data['coef'], y_pos, 
#                  xerr=[group_data['coef'] - group_data['coef lower 95%'],
#                        group_data['coef upper 95%'] - group_data['coef']],
#                  fmt='o', capsize=5,
#                  label=group,
#                  color=sns.color_palette('tab10')[(i+5)%10])
    
# plt.xlabel('log HR (95% CI)')
# plt.yticks(list(y_positions.values()), list(y_positions.keys()))

# ax.set_yticklabels([ylabel_dict[cov] for cov in combined_results.Covariate.unique()])
# plt.ylabel('Covariate')
# plt.title('Survival HR by Age Group')
# plt.tight_layout()

# # Get the current handles and labels
# handles, labels = ax.get_legend_handles_labels()

# # Define a custom sorting key function
# def sort_key(label):
#     if label == '80+':
#         return 80
#     else:
#         return int(label.split('-')[0])

# # Sort the handles and labels
# sorted_pairs = sorted(zip(handles, labels), key=lambda pair: sort_key(pair[1]), reverse=True)
# handles, labels = zip(*sorted_pairs)

# # Create the legend with the sorted handles and labels
# ax.legend(handles, labels)

# # plt.legend(combined_results['Age Group'].unique(), loc='best')

# plt.axvline(x=0, color='red', linestyle='--')
# ax.grid(axis='x', color='grey', linestyle='--', alpha=0.5)
# plt.xlabel('log HR (95% CI)')

# sns.despine()
# plt.savefig('../results/survival/survival_by_age_bin.png')
# plt.show()
# plt.clf()
# # %%
# # cause of death analysis
# survival_WHI = survival_df[survival_df.cohort=='WHI'].copy()

# # Analize death cause
# cause_df = survival_WHI[survival_WHI['death_cause'].notna()].copy()
# death_cause_counts = cause_df.death_cause.value_counts().to_dict()

# results = []
# for cause in cause_df.death_cause.unique():
#     if death_cause_counts[cause] < 2:
#         continue
#     cause_num = cause + f' ({death_cause_counts[cause]})'

#     if cause in ['Lymphoma (NHL only)', 'Unknown Cause']:
#         continue
#     cause_df['cause'] = (cause_df['death_cause'].isin([cause]))*1


#     model_logit = smf.logit(formula="cause ~ scale(max_fitness) + age_wave_1", data=cause_df)
#     res = model_logit.fit()

#     conf_int = res.conf_int().loc['scale(max_fitness)']
#     beta = res.params['scale(max_fitness)']

#     # Append the results to the list
#     results.append({
#         'cause': cause_num,
#         'beta': beta,
#         'conf_low': conf_int[0],
#         'conf_high': conf_int[1]
#     })

# # Convert the results list to a DataFrame
# results_df = pd.DataFrame(results)
# results_df  = results_df.sort_values(by='beta')

# fig, ax = plt.subplots()  # Reduced height for closer lines

# # Create a seaborn plot
# plt.errorbar(results_df['beta'],
#              results_df['cause'],
#              xerr=(results_df['beta'] - results_df['conf_low'],
#                    results_df['conf_high'] - results_df['beta']),
#                    fmt='o', capsize=3,
#                    color=sns.color_palette('tab10')[0],
#             )

# # Add a dotted vertical line at x=0
# plt.axvline(x=0, color='red', linestyle='--')
# ax.grid(axis='x', color='grey', linestyle='--', alpha=0.5)

# # Add labels and title
# plt.xlabel('Fitness Effect (Beta Value)')
# plt.ylabel('Death Cause (number of observations)')
# plt.title('Correlation Between Fitness and Death Cause')
# sns.despine()
# plt.tight_layout()
# plt.savefig('../results/survival/fitness_vs_death_cause.png', dpi=1000)
# plt.show()
# plt.clf()
# # %%
# %%
