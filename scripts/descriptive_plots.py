# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

from scipy import stats

# Import results
with open('../exports/final_participant_list.pk', 'rb') as f:
    cohort = pk.load(f)

summary = pd.read_csv('../results/mutation_df.csv')
part_summary = pd.read_csv('../results/participant_df.csv')

part_summary = part_summary.sort_values(by='cohort')

# %%
custom_palette['plotly_d3']
# Define color dictionary for different cohorts


# numer of mutations by fitness
sns.histplot(summary, x='fitness', hue='cohort', multiple='stack', palette=cohort_color_dict, bins=50)
sns.despine()
plt.ylabel('Mutation Counts')
plt.xlabel('Fitness')
plt.yscale('log')
plt.savefig('../results/descriptive/Counts vs fitness.png', dpi=1000)
plt.show()
plt.clf()

sns.histplot(summary, x='mut_max_VAF', hue='cohort', multiple='stack', palette=cohort_color_dict, bins=50)
sns.despine()
plt.ylabel('Mutation Counts')
plt.xlabel('Maximum observed VAF')
plt.yscale('log')
plt.savefig('../results/descriptive/Counts vs max VAF.png', dpi=1000)
plt.show()
plt.clf()


# %%

fig, ax = plt.subplots()
for i, cohort in enumerate(summary.cohort.unique()):
    sns.boxplot(x=i, y=summary[summary.cohort==cohort].fitness,
                color=cohort_color_dict[cohort])
ax.set_xticklabels(summary.cohort.unique())
plt.ylabel('Fitness')
sns.despine()
plt.savefig('../results/descriptive/Fitness distribution by cohort.png', dpi=1000)
plt.show()
plt.clf()

fig, ax = plt.subplots()
for i, cohort in enumerate(summary.cohort.unique()):
    sns.boxplot(x=i, y=summary[summary.cohort==cohort].mut_max_VAF,
                color=cohort_color_dict[cohort])
ax.set_xticklabels(summary.cohort.unique())
plt.ylabel('Maximum observed VAF')
sns.despine()
plt.savefig('../results/descriptive/Max VAF distribution by cohort.png', dpi=1000)

# %%
# Age distribution

sns.histplot(part_summary, x='age_wave_1', hue='cohort', multiple='stack', palette=cohort_color_dict)
plt.ylabel('Number of Participants')
plt.xlabel('Age at first observation')
sns.despine()
plt.savefig('../results/descriptive/participant_age distribution.png', dpi=1000)
plt.show()
# %%

# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    part_summary['age_wave_1'], part_summary['max_fitness'])

# Calculate R-squared
r_squared = r_value**2

# Create the plot
plt.figure(figsize=(10, 6))

sns.scatterplot(data=part_summary, x='age_wave_1', y='max_fitness', hue='cohort',
            alpha=0.7, palette=cohort_color_dict)

sns.regplot(data=part_summary, x='age_wave_1', y='max_fitness',
            scatter=False,
            order=1, color=sns.color_palette(custom_palette['plotly_d3'])[3])

sns.despine()
plt.xlabel('Age at first observation')
plt.ylabel('Max Fitness in participant')

# Add R-squared and p-value to the title
plt.title(f'Fitness vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})')
plt.savefig('../results/descriptive/age_vs_fitness_scatter.png', dpi=1000)
plt.show()
plt.clf()


# %%

# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    summary['part_age_tp_0'], summary['fitness'])

# Calculate R-squared
r_squared = r_value**2

# Create the plot
# plt.figure(figsize=(10, 6))

sns.jointplot(data=summary, x='part_age_tp_0', y='fitness', hue='cohort',
            alpha=0.7, palette=cohort_color_dict)

sns.regplot(data=summary, x='part_age_tp_0', y='fitness',
            scatter=False,
            order=1, color='tab:orange')

plt.xlim(np.min(summary.part_age_tp_0)-2, np.max(summary.part_age_tp_0)+2)
plt.ylim(-0.02,1)

sns.despine()
plt.xlabel('Age at first observation')
plt.ylabel('Max Fitness in participant')

# Add R-squared and p-value to the title
# plt.title(f'Fitness vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})')
plt.savefig('../results/descriptive/age_vs_fitness_scatter.png', dpi=1000)
plt.savefig('../results/descriptive/age_vs_fitness_scatter.svg')

plt.show()
plt.clf()
# %%
import matplotlib.gridspec as gridspec

# Create main figure
fig = plt.figure(figsize=(8,7))

# First divide the figure into top and bottom sections with matching margins
outer_gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[6, 1],
                           height_ratios=[1, 6],
                           hspace=0.1, wspace=0.1)

# Create the broken axis subplot for main scatter
gs_scatter = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                            subplot_spec=outer_gs[1, 0],
                                            height_ratios=[1, 6], 
                                            hspace=0.05)

# Create broken axis subplot for right KDE
gs_kde_right = gridspec.GridSpecFromSubplotSpec(2, 1,
                                              subplot_spec=outer_gs[1, 1],
                                              height_ratios=[1, 6],
                                              hspace=0.05)

# Create all axes
ax_kde_top = fig.add_subplot(outer_gs[0, 0])      # Top KDE
ax_kde_right_top = fig.add_subplot(gs_kde_right[0])    # Right KDE top
ax_kde_right_bottom = fig.add_subplot(gs_kde_right[1])  # Right KDE bottom
ax_main_top = fig.add_subplot(gs_scatter[0])      # Top part of broken axis
ax_main_bottom = fig.add_subplot(gs_scatter[1])    # Bottom part of broken axis

# Plot KDEs
sns.kdeplot(data=summary, x='part_age_tp_0', hue='cohort',
            palette=cohort_color_dict, 
            fill=True, legend=False, ax=ax_kde_top)

# Plot right KDE in both top and bottom parts
for ax in [ax_kde_right_top, ax_kde_right_bottom]:
    sns.kdeplot(data=summary, y='fitness', hue='cohort',
                palette=cohort_color_dict, 
                fill=True, legend=False, ax=ax)

# Plot scatter plots
for ax in [ax_main_top, ax_main_bottom]:
    sns.scatterplot(data=summary, x='part_age_tp_0', y='fitness', 
                   hue='cohort', alpha=0.7, 
                   palette=cohort_color_dict, ax=ax)
        
    sns.regplot(data=summary, x='part_age_tp_0', y='fitness',
                scatter=False, order=1, color='tab:grey', ax=ax)

# Style the top KDE
ax_kde_top.set_xlim(48, 100)
ax_kde_top.set_ylabel('')
ax_kde_top.set_xlabel('')
ax_kde_top.tick_params(labelbottom=False, labelleft=False, left=False)
sns.despine(ax=ax_kde_top, top=True, left=True, right=True)

# Style the right KDEs
ax_kde_right_top.set_ylim(0.94, 1.0)
ax_kde_right_bottom.set_ylim(-0.02, 0.65)
for ax in [ax_kde_right_top, ax_kde_right_bottom]:
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False)
    sns.despine(ax=ax, top=True, right=True, bottom=True)

# Configure broken right KDE appearance
ax_kde_right_top.spines.bottom.set_visible(False)
ax_kde_right_bottom.spines.top.set_visible(False)

# Set different y-axis limits for broken main axis
ax_main_top.set_ylim(0.94, 1.0)
ax_main_bottom.set_ylim(-0.02, 0.65)
ax_main_top.set_xlim(48, 100)
ax_main_bottom.set_xlim(48, 100)

# Configure broken main axis appearance
ax_main_top.spines.bottom.set_visible(False)
ax_main_top.tick_params(labeltop=False,labelbottom=False, bottom=False)
ax_main_top.spines.top.set_visible(False)
ax_main_top.spines.right.set_visible(False)
ax_main_top.set_yticks([0.95, 1])
ax_main_top.set_ylabel("")
ax_main_top.set_xlabel("")

ax_main_top.get_legend().remove()

ax_main_bottom.spines.top.set_visible(False)
ax_main_bottom.xaxis.tick_bottom()
ax_main_bottom.spines.right.set_visible(False)
ax_main_bottom.set_ylabel("Fitness", fontsize=12)
ax_main_bottom.set_xlabel("Age at first observation", fontsize=12)

# Add broken axis indicators for main plot
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
             linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_main_top.plot([0], [0], transform=ax_main_top.transAxes, **kwargs)
ax_main_bottom.plot([0], [1], transform=ax_main_bottom.transAxes, **kwargs)

# Add broken axis indicators for right KDE
ax_kde_right_top.plot([0], [0], transform=ax_kde_right_top.transAxes, **kwargs)
ax_kde_right_bottom.plot([0], [1], transform=ax_kde_right_bottom.transAxes, **kwargs)

# Handle legend
ax_main_bottom.legend(title='Cohort', bbox_to_anchor=(1.02, 1), loc='best')


# Add R-squared and p-value to the title
ax_main_bottom.text(x=60, y=0.57, 
                    s=f'Fitness vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})',
                    fontsize=12,
                    color='tab:grey'
                    )

plt.savefig('../results/descriptive/age_vs_fitness_scatter.png', dpi=1000)
plt.savefig('../results/descriptive/age_vs_fitness_scatter.svg')

# %%
# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['age_wave_1'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary,
            x='Age_Group', y='max_fitness', color=sns.color_palette('tab10')[9])

sns.despine()
plt.xlabel('Age Group')
plt.ylabel('Max Fitness in Participant')
# plt.title('Distribution omax_f Fitness by Age Group')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.savefig('../results/descriptive/age_vs_fitness_boxplot.png', dpi=1000)
plt.show()
plt.clf()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

age_groups = ['50-60', '60-70', '70-80', '80-90+']
# Assuming part_summary is your dataframe
# Create age groups (as in your original code)
part_summary['Age_Group'] = pd.cut(part_summary['age_wave_1'],
                                   bins=[50, 60, 70, 80, 100],
                                   labels=age_groups)

group1 = part_summary[part_summary.Age_Group == '60-70'].max_fitness
group2 = part_summary[part_summary.Age_Group == '70-80'].max_fitness

stats.ttest_ind(group1, group2)
# Function to calculate significance
def calculate_significance(group1, group2):
    statistic, p_value = stats.ttest_ind(group1, group2)
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""  # Return empty string for non-significant results

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=part_summary, x='Age_Group', y='max_fitness', hue='cohort', palette=cohort_color_dict)

# Add significance stars between age groups
age_groups = part_summary['Age_Group'].unique()
y_max = part_summary['max_fitness'].max()

from itertools import combinations

for comb in combinations(age_groups, 2):
    group1 = part_summary[part_summary['Age_Group'] == comb[0]]['max_fitness']
    group2 = part_summary[part_summary['Age_Group'] == comb[1]]['max_fitness']  
    significance = calculate_significance(group1, group2)
    if significance:  # Only add annotation if significant
        print(comb)
        plt.text((i + i+1)/2, y_max * 1.05, significance, ha='center', va='bottom')
        plt.plot([i, i+1], [y_max * 1.03, y_max * 1.03], color='black')

# for i in range(len(age_groups) - 1):
#     group1 = part_summary[part_summary['Age_Group'] == age_groups[i]]['max_fitness']
#     group2 = part_summary[part_summary['Age_Group'] == age_groups[i+1]]['max_fitness']
#     significance = calculate_significance(group1, group2)
#     if significance:  # Only add annotation if significant
#         plt.text((i + i+1)/2, y_max * 1.05, significance, ha='center', va='bottom')
#         plt.plot([i, i+1], [y_max * 1.03, y_max * 1.03], color='black')

plt.xlabel('Age Group')
plt.ylabel('Max Fitness')
plt.xticks(rotation=45)
plt.tight_layout()
sns.despine()

plt.savefig('../results/descriptive/age_vs_fitness_boxplot_cohort_with_significance.png', dpi=1000)
plt.show()
plt.clf()

# %%
# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['age_wave_1'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary, hue='cohort', 
            x='Age_Group', y='max_fitness',
            palette=cohort_color_dict)

# sns.despine()
plt.xlabel('Age Group')
plt.ylabel('Max Fitness')

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
slope, intercept, r_value, p_value, std_err = stats.linregress(
    part_summary['age_wave_1'], part_summary['max_VAF_tp_0'])

# Calculate R-squared
r_squared = r_value**2

# Create the plot
plt.figure(figsize=(10, 6))

sns.scatterplot(data=part_summary, x='age_wave_1', y='max_VAF_tp_0', hue='cohort', alpha=0.7,
                palette=cohort_color_dict)
sns.regplot(data=part_summary, x='age_wave_1', y='max_VAF_tp_0',
            scatter=False,
            order=1, color=custom_palette['plotly_d3'][3])

sns.despine()
plt.xlabel('Age')
plt.ylabel('VAF at first observation')

# Add R-squared and p-value to the title
plt.title(f'max_VAF vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})')
plt.savefig('../results/descriptive/age_vs_max_VAF_scatter.png', dpi=1000)
plt.show()
plt.clf()

# Create age groups
part_summary['Age_Group'] = pd.cut(part_summary['age_wave_1'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary, color=sns.color_palette('tab10')[9],
            x='Age_Group', y='max_VAF_tp_0')

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
part_summary['Age_Group'] = pd.cut(part_summary['age_wave_1'],
            bins=[50, 60, 70, 80, 100],
            labels=['50-60', '60-70', '70-80', '80-90+'])

# Create the plot
plt.figure(figsize=(12, 6))

sns.boxplot(data=part_summary, hue='cohort', palette=cohort_color_dict,
            x='Age_Group', y='max_VAF_tp_0')

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
            palette=sns.color_palette())

# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=55)

sns.stripplot(data=outliers, x='PreferredSymbol', y='fitness',
              hue='reports', jitter=0.8, hue_order = hue_order, dodge=True, marker='o', 
              size=7, edgecolor='none')

plt.legend(title='Reports')
sns.despine()
plt.xlabel('')
plt.xticks(ha='right', fontsize=12)
plt.ylabel('Fitness', fontsize=20)
plt.yticks(ha='right', fontsize=12)
# save and show
plt.savefig('../results/descriptive/cosmic_combined_fitness_box.png', dpi=1500)
plt.show()


# %%

filtered_df = summary[summary.PreferredSymbol.isin(['DNMT3A', 'TET2', 'JAK2', 'PPM1D', 'ASXL1'])]
filtered_df = filtered_df.sort_values(by='PreferredSymbol')
sns.boxplot(filtered_df, x='overall_order', y='fitness', hue='PreferredSymbol')


# Customize the plot
plt.legend(title='Gene')
plt.xlabel('Mutation order', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.ylim(0, 0.6)
sns.despine()
# Show the plot
plt.tight_layout()
plt.savefig('../results/descriptive/fitness_by_overall_order_selected_genes.png', dpi=1500)
plt.show()
plt.clf()
# %%

sns.boxplot(summary, x='overall_order', y='fitness')

# Customize the plot
plt.xlabel('Mutation order', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.ylim(0, 0.6)
sns.despine()

# Show the plot
plt.tight_layout()
plt.savefig('../results/descriptive/fitness_by_overall_order.png', dpi=1500)
plt.show()
plt.clf()

# %%

sns.regplot(summary, x='overall_order', y='clipped_init_age')

# %%
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)
sns.scatterplot(summary, x='clipped_init_age', y='fitness', hue='overall_order')

# Customize the plot
plt.xlabel('ATMA', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.ylim(0, 0.6)
sns.despine()

max_age = part_summary.age_wave_1.max()
N = 100_000
ATMA = np.linspace(1, 70, 70)
y = np.log(0.01*N)/(max_age - ATMA)
sns.lineplot(x=ATMA, y=y, color='red', linestyle='--', label='min detectable fitness')

# Show the plot
plt.tight_layout()
plt.savefig('../results/descriptive/fitness_by_overall_order.png', dpi=200)
plt.show()
plt.clf()
# %%

# Perform linear regression
result = stats.linregress(summary['overall_order'], summary['fitness'])

# Create the plot
plt.figure()
sns.scatterplot(data=summary, x='overall_order', y='fitness', color=sns.color_palette()[0], alpha=0.6)
sns.regplot(data=summary, x='overall_order', y='fitness', scatter=False, color=sns.color_palette()[1])

# Customize the plot
plt.xlabel('Mutation order')
plt.ylabel('Fitness')
plt.ylim(0, 0.6)
p_value_formatted = f'{result.pvalue:.0e}'
plt.title(f'Fitness vs Mutation Order (p-value: {p_value_formatted})', fontsize=14)
sns.despine()

# Show the plot
plt.tight_layout()
plt.savefig('../results/descriptive/fitness_vs_overall_order.png', dpi=1500)
plt.show()
plt.clf()
# %%
# init age vs fitness
summary['clipped_init_age'] = np.clip(summary.init_age, 0, summary.init_age)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(summary['clipped_init_age'], summary['fitness'])

# Calculate R-squared
r_squared = r_value**2

# Create the plot
plt.figure(figsize=(10, 6))

sns.scatterplot(data=summary, x='clipped_init_age', y='fitness', hue='cohort', alpha=0.7,
                palette=cohort_color_dict)

sns.regplot(data=summary, x='clipped_init_age', y='fitness',
            scatter=False,
            order=1, color=custom_palette['plotly_d3'][3])

sns.despine()
plt.xlabel('Age at mutation acquisition')
plt.ylabel('Fitness')

# Add R-squared and p-value to the title
plt.title(f'max_VAF vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})')
plt.savefig('../results/descriptive/age_mutation_vs_fitness_regplot.png', dpi=1000)
plt.show()
plt.clf()

# %%
# Create the plot
plt.figure()

sns.scatterplot(data=summary, x='clipped_init_age', y='fitness', hue='cohort', alpha=0.7,
                palette=cohort_color_dict)

plt.xlabel('Age at mutation acquisition')
plt.ylabel('Fitness')
sns.despine()


# Add R-squared and p-value to the title
plt.savefig('../results/descriptive/age_mutation_vs_fitness_scatter.png', dpi=1000)
plt.show()
plt.clf()

# %%
# Create the plot
plt.figure()
summary['TET2'] = summary.PreferredSymbol == 'TET2'
summary['TET2'] = summary.PreferredSymbol == 'TET2'
summary['JAK'] = summary.PreferredSymbol == 'TET2'

sns.scatterplot(data=summary[summary['TET2']], x='clipped_init_age', y='fitness', hue='TET2', alpha=0.7)


summary_filtered= summary[summary.PreferredSymbol.isin(['TET2', 'DNMT3A', 'TP53', 'PPM1D'])].copy()
sns.histplot(summary_filtered[summary_filtered.clipped_init_age>0], x='clipped_init_age', hue='PreferredSymbol', multiple='dodge', stat='density')
sns.despine()

plt.savefig('../results/descriptive/age_mutation_vs_f.png', dpi=1000)
plt.show()
plt.clf()


# %%
clonal_str_df = summary.groupby(['participant_id', 'clonal_structure']).max(numeric_only=True)

clonal_str_df = clonal_str_df[clonal_str_df.clonal_structure_size<3]
# Perform linear regression
result = stats.linregress(clonal_str_df['clonal_structure_size'], 
                          clonal_str_df['fitness'])

# Create the plot
plt.figure()
sns.scatterplot(data=clonal_str_df, x='clonal_structure_size', y='fitness', color=sns.color_palette()[0], alpha=0.6)
sns.regplot(data=clonal_str_df, x='clonal_structure_size', y='fitness', scatter=False, color=sns.color_palette()[1])

# Customize the plot
plt.xlabel('Clone size')
plt.ylabel('Fitness')
plt.ylim(0, 0.6)
p_value_formatted = f'{result.pvalue:.0e}'
plt.title(f'Fitness vs Mutation Order (p-value: {p_value_formatted})', fontsize=14)
sns.despine()

# Show the plot
plt.tight_layout()
plt.savefig('../results/descriptive/fitness_vs_clonal_structure_size.png', dpi=1500)
plt.show()
plt.clf()

# %%
numeric_cols = ['fitness', 'init_age', 'part_age_tp_0', 'mut_VAF_tp_0', 'mut_max_VAF', 'overall_order']
correlation = summary[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Mutation Features')

plt.savefig('../results/descriptive/correlation_heatmap.png', dpi=1500)
plt.show()

plt.clf()

# %%
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Assuming your dataset is called 'df'
# df = pd.read_csv('your_dataset.csv')

def kruskal_wallis_test(group1, group2):
    if len(group1) < 2 or len(group2) < 2:
        return 0, 1  # Return statistic 0 and p-value 1
    if group1.equals(group2):  # Skip if the groups are identical
        return 0, 1  # Return statistic 0 and p-value 1
    try:
        statistic, p_value = stats.kruskal(group1, group2)
        return statistic, p_value
    except ValueError:
        return 0, 1  # Return statistic 0 and p-value 1

# Get genes with at least 2 instances
gene_counts = summary['PreferredSymbol'].value_counts()
valid_genes = gene_counts[gene_counts >= 2].index.tolist()


# Calculate median fitness for each gene
median_fitness = summary.groupby('PreferredSymbol')['fitness'].median().sort_values(ascending=False)
ordered_genes = median_fitness.index.intersection(valid_genes).tolist()

# Create empty DataFrames to store results
p_values = pd.DataFrame(1.0, index=ordered_genes, columns=ordered_genes)

# Perform Kruskal-Wallis test for each pair of genes
for i, gene1 in enumerate(ordered_genes):
    for j, gene2 in enumerate(ordered_genes):
        if i < j:  # To avoid redundant comparisons
            group1 = summary[summary['PreferredSymbol'] == gene1]['fitness']
            group2 = summary[summary['PreferredSymbol'] == gene2]['fitness']
            _, p_value = kruskal_wallis_test(group1, group2)
            p_values.loc[gene1, gene2] = p_value
            p_values.loc[gene2, gene1] = p_value


# Filter out genes without any significant interactions
significant_genes = p_values.index[
    (p_values < 0.05).any(axis=1) | (p_values < 0.05).any(axis=0)
].tolist()
p_values_filtered = p_values.loc[significant_genes, significant_genes]
log_p_values = -np.log10(p_values_filtered)
log_p_values_clipped = log_p_values.where(log_p_values<3, 3)
fig, ax = plt.subplots(figsize=(12, 10))
rdgn = sns.diverging_palette(h_neg=220, h_pos=20, s=70, l=60, sep=1, as_cmap=True)
divnorm = TwoSlopeNorm(vmin=np.min(log_p_values_clipped), vcenter=1.3, vmax=np.max(log_p_values_clipped))
sns.heatmap(log_p_values_clipped, cmap=rdgn, norm=divnorm, 
            cbar=True, ax=ax)
x_ticks = log_p_values_clipped.index
# ax.tick_params(axis='x', labelsize=12)
# ax.xaxis.set_tick_params(rotation=60)
# ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(x_ticks, rotation=60, fontsize=12, ha='right') # rotate the labels with proper anchoring.
ax.set_yticklabels(x_ticks, fontsize=12, ha='right') # rotate the labels with proper anchoring.


# Add significance level line to colorbar
colorbar = ax.collections[0].colorbar
# Increase tick label size
colorbar.ax.tick_params(labelsize=12)  # Increase tick label size

# Add significance level
colorbar.ax.axhline(y=1.3, color='black', linestyle='--', linewidth=1)
colorbar.ax.text(2, 1.5, 'Significance level \n-log10 (p-value)', 
                va='center', 
                ha='left',
                color='black',
                rotation=90,
                fontsize=12,
                )

plt.savefig('../results/descriptive/kruskal_wallis_heatmap.png', transparent=True, dpi=300)
plt.savefig('../results/descriptive/kruskal_wallis_heatmap.svg')

# %%
# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)


cohort_high = [part for part in cohort if (part.shape[0]>2) & (part.uns['cohort']=='sardiNIA')]
combined_posterior_plot(cohort_high[9])
plt.savefig('../results/deterministic/example_sardiNIA.png')
plt.savefig('../results/deterministic/example_sardiNIA.svg')
plt.show()
plt.clf()
 
cohort_high = [part for part in cohort if (part.shape[0]>2) & (part.uns['cohort']=='WHI')]
combined_posterior_plot(cohort_high[14])
plt.savefig('../results/deterministic/example_WHI.png')
plt.savefig('../results/deterministic/example_WHI.svg')
plt.show()
plt.clf()

cohort_high = [part for part in cohort if (part.shape[0]>2) & (part.uns['cohort']=='LBC')]
combined_posterior_plot(cohort_high[3])
plt.savefig('../results/deterministic/example_LBC.png')
plt.savefig('../results/deterministic/example_LBC.svg')

plt.show()
plt.clf()

# %%

summary['log_size_prediction_120'] = np.log(summary.size_prediction_120)
sns.scatterplot(summary, x='init_age', y='fitness', hue='log_size_prediction_120')
sns.scatterplot(summary, x='init_age', y='fitness', hue='VAF_prediction_120')

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

ax.text(120+1, cs[0, -1], f"Predicted CS\n{int(cs[0, -1])}", 
        color=sns.color_palette()[0],
         va='top', ha='left')


ax.text(120+1, cs[1, -1], f"Predicted CS\n{int(cs[1, -1])}", 
        color=sns.color_palette()[1],
         va='top', ha='left')

# ax.axvline(x=ATMA[1], linestyle='--', color=sns.color_palette()[1])
# ax.text(ATMA[1]+1, ax.get_ylim()[1], f"ATMA = {ATMA[1]}", 
#          rotation=90, va='top', ha='left')
plt.yscale('log')
# plt.xlim(50, 130)
plt.title('Evolution of two mutations with same fitness\n')
sns.despine()
plt.ylabel('Clone Size')
plt.xlabel('Participant Age')
plt.savefig('../results/deterministic/predicted_CS.png', bbox_inches = 'tight')
# %%


sns.scatterplot(x=summary.fitness, y=summary.size_prediction_120_z_score)
plt.title('fitness vs predicted size at 120')
sns.despine()
plt.savefig('../results/descriptive/fitness vs size.png')
# %%
