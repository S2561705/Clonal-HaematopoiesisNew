# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

from src.aux import *
from src.blood_markers_aux import *


import matplotlib.gridspec as gridspec
import pickle as pk

from scipy import stats
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import statsmodels.formula.api as smf
# # Import results
# with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
#     cohort = pk.load(f)

summary = pd.read_csv('../results/mutation_df.csv')
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)

part_summary = pd.read_csv('../results/participant_df.csv')
part_summary = part_summary.sort_values(by='cohort')
survival_df = part_summary[part_summary['dead'].notna()].copy()
ylabel_dict = {'max_VAF_z_score': 'Max VAF',
               'max_fitness_z_score':'Max Fitness',
               'scale(age_wave_1)': 'Age',
               'Female': 'Sex',
               'max_size_prediction_120_z_score': 'MACS 120'}

# %%

fitness = 0.1
ATMA = np.array([70, 50])

t = np.linspace(ATMA, 120, 100)
cs = np.exp(fitness*(t.T-ATMA[:, None]))

fig, ax = plt.subplots()
sns.lineplot(x=t[:, 0], y=cs[0])
sns.lineplot(x=t[:, 1], y=cs[1])

ax.axvline(x=ATMA[0], linestyle='--', color=sns.color_palette()[0])
ax.text(ATMA[0]+1, ax.get_ylim()[1], f"ATMA = {ATMA[0]}", 
         rotation=90, va='top', ha='left')

ax.axvline(x=ATMA[1], linestyle='--', color=sns.color_palette()[1])
ax.text(ATMA[1]+1, ax.get_ylim()[1], f"ATMA = {ATMA[1]}", 
         rotation=90, va='top', ha='left')

ax.text(120+1, cs[0, -1], f"Max Predicted\nCS at age 120", 
        color=sns.color_palette()[0],
         va='top', ha='left')


ax.text(120+1, cs[1, -1], f"Max Predicted\nCS at age 120", 
        color=sns.color_palette()[1],
         va='top', ha='left')

# ax.axvline(x=ATMA[1], linestyle='--', color=sns.color_palette()[1])
# ax.text(ATMA[1]+1, ax.get_ylim()[1], f"ATMA = {ATMA[1]}", 
#          rotation=90, va='top', ha='left')
# plt.xlim(50, 130)
plt.title('Evolution of two mutations with same fitness\n')
sns.despine()
plt.ylabel('Clone Size')
plt.xlabel('Participant Age')
plt.savefig('../plots/Figure 2/predicted_CS.png', dpi=1_000, bbox_inches = 'tight')
plt.savefig('../plots/Figure 2/predicted_CS.svg',bbox_inches='tight')

# %%

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(survival_df, 
        duration_col='from_wave_1',
        event_col='dead',
        formula="max_VAF_z_score + max_fitness_z_score + scale(age_wave_1) + Female + max_size_prediction_120_z_score")
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

# Set y-ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(summary.index.map(ylabel_dict))
sns.despine()
# Adjust layout
plt.tight_layout()
plt.savefig('../plots/Figure 2/combined cohorts.png', dpi=200)
plt.show()
plt.clf()

# %%

for covariate in ['max_VAF_z_score', 'max_fitness_z_score',
                   'scale(age_wave_1)', 'Female', 'max_size_prediction_120_z_score']:

    fig, ax = plt.subplots()
    # Define the fitness values you want to plot
    fitness_values = np.quantile(survival_df[covariate], [0.25,0.5, 0.75])
    
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

    plt.legend(['25% quantile', 'mean', '75% quantile'], loc='best')
    sns.despine()
    plt.savefig(f'../plots/Figure 2/partial_effects_on_outcome_{ylabel_dict[covariate]}.png', transparent=True, dpi=200)
    plt.show()
    plt.clf()

# %%
# survival by cohort
cohort_choice = ['LBC', 'WHI', 'sardiNIA']
# Initialize a dictionary to store results
results = {}
# Fit separate models for each cohort

for cohort in cohort_choice:
    cohort_data = survival_df[survival_df['cohort'] == cohort].copy()
    normalise_parameter(cohort_data, 'max_fitness')
    normalise_parameter(cohort_data, 'max_VAF')
    normalise_parameter(cohort_data, 'max_size_prediction_120')
    normalise_parameter(cohort_data, 'max_size_prediction_next_30')

    if cohort == 'WHI':
        formula = "max_VAF_z_score + max_fitness_z_score + scale(age_wave_1) + max_size_prediction_120_z_score"
    else:
        formula = "max_VAF_z_score + max_fitness_z_score + scale(age_wave_1) + Female + max_size_prediction_120_z_score"
    cph = CoxPHFitter()
    cph.fit(cohort_data, duration_col='from_wave_1', event_col='dead', 
            formula=formula)
    
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
plt.savefig('../plots/Supp Figure 2/survival_by_cohort.png',  transparent=True, dpi=200)
plt.savefig('../plots/Supp Figure 2/survival_by_cohort.svg')
plt.show()
plt.clf()

# %%
# Survival Analysis by age bins

# Step 1: Create age bins
survival_df['age_bin'] = pd.cut(survival_df['age_wave_1'], 
                                    bins=[49, 60, 70, 80, np.inf], 
                                    labels=['50-59', '60-69', '70-79', '80+'])
results = {}

for age_group in survival_df['age_bin'].unique():
    group_data = survival_df[survival_df['age_bin'] == age_group].copy()
    
    cph = CoxPHFitter()
    cph.fit(group_data, duration_col='from_wave_1', event_col='dead', 
            formula="max_VAF_z_score + max_fitness_z_score + Female + max_size_prediction_120_z_score")
    
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
plt.savefig('../plots/Supp Figure 2/survival_by_age_bin.png', transparent=True, dpi=200)
plt.savefig('../plots/Supp Figure 2/survival_by_age_bin.svg')

plt.show()
plt.clf()
# %%
# cause of death analysis
survival_WHI = survival_df[survival_df.cohort=='WHI'].copy()

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


    model_logit = smf.logit(formula="cause ~ max_size_prediction_120_z_score + age_wave_1", data=cause_df)
    res = model_logit.fit()

    conf_int = res.conf_int().loc['max_size_prediction_120_z_score']
    beta = res.params['max_size_prediction_120_z_score']

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
plt.savefig('../plots/Figure 2/fitness_vs_death_cause.png', dpi=1000)
plt.show()
plt.clf()
# %%


dead_df = survival_df[survival_df.dead==1].copy()
dead_df['surv_offset'] = 0
survival_df['surv_offset'] = 0

for cohort in cohort_choice:
    dead_mask = dead_df.cohort == cohort
    survival_mask = survival_df.cohort == cohort

    cohort_data = dead_df[dead_mask]
    regr = stats.linregress(x=cohort_data.age_wave_1,
                            y=cohort_data.from_wave_1)

    dead_df.loc[dead_mask, 'surv_offset'] = (dead_df[dead_mask].from_wave_1 
                            - dead_df[dead_mask].age_wave_1*regr.slope 
                            - regr.intercept)
    
    survival_df.loc[survival_mask, 'surv_offset'] = (survival_df[survival_mask].from_wave_1 
                            - survival_df[survival_mask].age_wave_1*regr.slope 
                            - regr.intercept)


stats.linregress(x=dead_df.surv_offset, y=dead_df.max_size_prediction_120_z_score)   
stats.linregress(x=dead_df.surv_offset, y=dead_df.max_fitness_z_score)   


sns.scatterplot(dead_df, x='max_size_prediction_120_z_score', y='max_fitness_z_score', hue='surv_offset')
# %%

sns.regplot(x=dead_df.max_size_prediction_120_z_score - dead_df.max_fitness_z_score,
                y=dead_df.surv_offset)
# %%
