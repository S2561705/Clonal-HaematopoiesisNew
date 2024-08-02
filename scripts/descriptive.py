# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

from scipy import stats
sns.set_palette(custom_palette['plotly_d3'])


cohort_color_dict = {'sardiNIA':  '#f8cbad',
                    'WHI':  '#9dc3e6' ,
                    'LBC':  '#70ad47'}


# Import results
with open('../exports/WHI/WHI_fitted_1percent.pk', 'rb') as f:
    processed_WHI = pk.load(f)

# Import results
with open('../exports/LBC/merged_cohort_fitted.pk', 'rb') as f:
    processed_lbc = pk.load(f)

# Import results
with open('../exports/sardiNIA/sardiNIA_1percent_fitted.pk', 'rb') as f:
    processed_sardinia = pk.load(f)

for cohort_name, cohort in zip(['WHI','LBC','sardiNIA'],
                  [processed_WHI, processed_lbc, processed_sardinia]):
    for part in cohort:
        part.obs['Cohort'] = cohort_name
        part.obs['participant_id'] = part.uns['participant_id']
        part.obs['Age'] = part.var.time_points[0]
        part.obs['max_VAF'] = part.X[0].max()

cohort = processed_WHI + processed_lbc + processed_sardinia
# drop participants with warnings
cohort = [part for part in cohort if part.uns['warning'] is None]

# %%

# Find overlapping columns
overlapping_obs_columns = list(set(processed_WHI[0].obs.columns) 
                                & set(processed_lbc[0].obs.columns) 
                                & set(processed_sardinia[0].obs.columns))

summary = pd.concat([part.obs[overlapping_obs_columns] for part in cohort])
summary = summary[summary['fitness'].notna()]
summary = summary.sort_values(by='fitness', ascending=False)


summary.to_csv('../results/summary.csv')

# %%

# numer of mutations by fitness
sns.histplot(summary, x='fitness', hue='Cohort', multiple='stack', palette=cohort_color_dict)
sns.despine()
plt.ylabel('Mutation Counts')
plt.xlabel('Fitness')
plt.savefig('../results/descriptive/Counts vs fitness.png', dpi=1000)
plt.show()
plt.clf()


sns.histplot(summary, x='max_VAF', hue='Cohort', multiple='stack', palette=cohort_color_dict)
sns.despine()
plt.ylabel('Mutation Counts')
plt.xlabel('Maximum VAF')
plt.savefig('../results/descriptive/Counts vs max VAF.png', dpi=1000)
plt.show()
plt.clf()
# %%

fig, ax = plt.subplots()
for i, cohort in enumerate(summary.Cohort.unique()):
    sns.boxplot(x=i, y=summary[summary.Cohort==cohort].fitness,
                color=cohort_color_dict[cohort])
ax.set_xticklabels(summary.Cohort.unique())
plt.ylabel('Fitness')
sns.despine()
plt.savefig('../results/descriptive/Fitness distribution by cohort.png', dpi=1000)
plt.show()
plt.clf()

fig, ax = plt.subplots()
for i, cohort in enumerate(summary.Cohort.unique()):
    sns.boxplot(x=i, y=summary[summary.Cohort==cohort].max_VAF,
                color=cohort_color_dict[cohort])
ax.set_xticklabels(summary.Cohort.unique())
plt.ylabel('Max VAF')
sns.despine()
plt.savefig('../results/descriptive/Max VAF distribution by cohort.png', dpi=1000)

# %%
# Age distribution
part_summary = summary.groupby(by='participant_id')[['Age','Cohort', 'fitness','max_VAF']].max()
sns.histplot(part_summary, x='Age', hue='Cohort', multiple='stack', palette=cohort_color_dict)
plt.ylabel('Number of Participants')
sns.despine()
plt.savefig('../results/descriptive/age distribution.png', dpi=1000)
plt.show()
            
# %%

# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(part_summary['Age'], part_summary['fitness'])

# Calculate R-squared
r_squared = r_value**2

# Create the plot
plt.figure(figsize=(10, 6))

sns.scatterplot(data=part_summary, x='Age', y='fitness', hue='Cohort',
            alpha=0.7, palette=cohort_color_dict)
sns.regplot(data=part_summary, x='Age', y='fitness',
            scatter=False,
            order=1, color=sns.color_palette(custom_palette['plotly_d3'])[3])

sns.despine()
plt.xlabel('Age')
plt.ylabel('Max Fitness')

# Add R-squared and p-value to the title
plt.title(f'Fitness vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})')
plt.savefig('../results/descriptive/age_vs_fitness_scatter.png', dpi=1000)
plt.show()
plt.clf()

# %%
# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['Age'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary,
            x='Age_Group', y='fitness', color=sns.color_palette('tab10')[9])

sns.despine()
plt.xlabel('Age Group')
plt.ylabel('Max Fitness')
# plt.title('Distribution of Fitness by Age Group')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.savefig('../results/descriptive/age_vs_fitness_boxplot.png', dpi=1000)
plt.show()
plt.clf()

# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['Age'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# %%


# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary, hue='Cohort', 
            x='Age_Group', y='fitness',
            palette=cohort_color_dict)

# sns.despine()
plt.xlabel('Age Group')
plt.ylabel('Max Fitness')
# plt.title('Distribution of Fitness by Age Group and Cohort')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
sns.despine()
plt.savefig('../results/descriptive/age_vs_fitness_boxplot_cohort.png', dpi=1000)
plt.show()
plt.clf()

# %%

# Age vs VAF
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(part_summary['Age'], part_summary['max_VAF'])

# Calculate R-squared
r_squared = r_value**2

# Create the plot
plt.figure(figsize=(10, 6))

sns.scatterplot(data=part_summary, x='Age', y='max_VAF', hue='Cohort', alpha=0.7,
                palette=cohort_color_dict)
sns.regplot(data=part_summary, x='Age', y='max_VAF',
            scatter=False,
            order=1, color=custom_palette['plotly_d3'][3])

sns.despine()
plt.xlabel('Age')
plt.ylabel('Max VAF')

# Add R-squared and p-value to the title
plt.title(f'max_VAF vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})')
plt.savefig('../results/descriptive/age_vs_max_VAF_scatter.png', dpi=1000)
plt.show()
plt.clf()

# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['Age'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary, color=sns.color_palette('tab10')[9],
            x='Age_Group', y='max_VAF')

sns.despine()
plt.xlabel('Age Group')
plt.ylabel('Max VAF')
# plt.title('Distribution of max_VAF by Age Group')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.savefig('../results/descriptive/age_vs_max_VAF_boxplot.png', dpi=1000)
plt.show()
plt.clf()

# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['Age'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary, hue='Cohort', palette=cohort_color_dict,
            x='Age_Group', y='max_VAF')

# sns.despine()
plt.xlabel('Age Group')
plt.ylabel('Max VAF')
# plt.title('Distribution of max_VAF by Age Group and Cohort')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
sns.despine()
plt.savefig('../results/descriptive/age_vs_max_VAF_boxplot_cohort.png', dpi=1000)
plt.show()
plt.clf()
# %%

fitness_threshold = 0.02
gene_count_threshold = 1

gene_counts = summary[summary.fitness>fitness_threshold].PreferredSymbol.value_counts()
gene_keep = gene_counts[gene_counts > gene_count_threshold].index


# summary_filtered
summary_filtered = summary[(summary.fitness>fitness_threshold)
                            & (summary.PreferredSymbol.isin(gene_keep))]


# Sort dataframe by gene mean
# Step 1: Calculate the mean fitness for each PreferredSymbol
means = summary_filtered.groupby('PreferredSymbol')['fitness'].mean()

# Step 2: Sort the means
sorted_means = means.sort_values(ascending=False)

# Step 3: Reorder the categories in PreferredSymbol based on the sorted means
summary_filtered['PreferredSymbol'] = pd.Categorical(
    summary_filtered['PreferredSymbol'],
    categories=sorted_means.index,
    ordered=True
)


plt.figure(figsize=(14,6))

# plot boxplot
sns.boxplot(summary_filtered, x='PreferredSymbol', y='fitness',
            hue='PreferredSymbol',
            showfliers='suspectedoutliers',
            # gap=1,
            palette = sns.color_palette('muted'))

# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=55)
plt.ylabel('Fitness')
sns.despine()
# save and show 
plt.savefig('../results/descriptive/combined_fitness_box.png', dpi=1000)
plt.show()
# %%


# Cosmic

import pandas as pd
cosmic_df = pd.read_csv('../data/Site_mutsWed Apr 3 10 00 12 2024.csv')

cosmic_df['p_key'] = cosmic_df['Gene Name'] + ' ' + cosmic_df['AA Mutation']

# Assuming cosmic_df and summary_filtered dataframes are already loaded
fitness_threshold = 0.02
gene_count_threshold = 1

gene_counts = summary[summary.fitness > fitness_threshold].PreferredSymbol.value_counts()
gene_keep = gene_counts[gene_counts > gene_count_threshold].index

# summary_filtered
summary_filtered = summary[(summary.fitness > fitness_threshold)
                           & (summary.PreferredSymbol.isin(gene_keep))]

# Sort dataframe by gene mean
means = summary_filtered.groupby('PreferredSymbol')['fitness'].mean()
sorted_means = means.sort_values(ascending=False)

# Reorder the categories in PreferredSymbol based on the sorted means
summary_filtered['PreferredSymbol'] = pd.Categorical(
    summary_filtered['PreferredSymbol'],
    categories=sorted_means.index,
    ordered=True
)

plt.figure(figsize=(14, 6))

# plot boxplot
sns.boxplot(data=summary_filtered, x='PreferredSymbol', y='fitness',
            showfliers=False,
            palette=sns.color_palette('deep'))

# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=55)

outliers= []
# Identify outliers
for preferred_symbol in summary_filtered['PreferredSymbol'].cat.categories:
    subset = summary_filtered[summary_filtered['PreferredSymbol'] == preferred_symbol]
    q1 = subset['fitness'].quantile(0.25)
    q3 = subset['fitness'].quantile(0.75)
    iqr = q3 - q1
    # lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers.append(subset[subset['fitness'] > upper_bound])

outliers = pd.concat(outliers)

outliers_dict = {'SF3B1 p.K666M': 'AML/MDS',
                'TET2 p.I1873T': 'AML/MDS',
                'TET2 p.Y1245C': 'CML',
                'TET2 p.L1332P': 'AML',
                'DNMT3A p.M682Cfs*23': 'AML',
                'DNMT3A p.S770L': 'AML/MDS',
                'U2AF1 p.Q84R': 'AML',
                'BCORL1 p.Leu236Arg': 'Not Reported',
                'CBL c.2252-2A>G': 'Meningioma',
                'TET2 p.T1883K': 'AML/CML',
                'TET2 p.P401fs': 'AML',
                'TET2 p.I274fs': 'CML',
                'TET2 p.Gln321Ter': 'AML/MDS',
                'NOTCH1 p.A1323S': 'Glioma',
                'NaN': 'AML/Lymph Neoplasm',
                'DNMT3A c.1429+1G>A':'AML/Lymph Neoplasm',
                'DNMT3A p.Tyr735Cys': 'AML',
                'DNMT3A p.R379C': 'Carcinoma',
                'ZNF318 p.R275Vfs*6': 'Carcinoma',
                'CUX1 p.V282V': 'Not Reported'}

outliers['reports'] = outliers['p_key'].map(outliers_dict)

# Define the desired order for the hue categories
hue_order = ['AML', 'CML', 'AML/CML', 'AML/MDS', 'AML/Lymph Neoplasm', 
             'Meningioma', 'Glioma', 'Carcinoma', 'Not Reported']

sns.stripplot(data=outliers, x='PreferredSymbol', y='fitness',
              hue='reports', jitter=0.8, hue_order = hue_order, dodge=True, marker='o', 
              size=7, edgecolor='none')

plt.legend(title='Reports')
sns.despine()

# save and show 
plt.ylabel('Fitness')
plt.savefig('../results/descriptive/combined_fitness_box.png', dpi=1500)
plt.show()

# %%