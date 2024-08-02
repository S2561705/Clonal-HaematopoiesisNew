# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
# %%

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

cohort = processed_WHI + processed_lbc + processed_sardinia
# drop participants with warnings
cohort = [part for part in cohort if part.uns['warning'] is None]

# for part in cohort:
#     part.obs['participant_id'] = part.uns['participant_id']

# %%

# Find overlapping columns
overlapping_obs_columns = list(set(processed_WHI[0].obs.columns) 
                                & set(processed_lbc[0].obs.columns) 
                                & set(processed_sardinia[0].obs.columns))

summary = pd.concat([part.obs[overlapping_obs_columns] for part in cohort])
summary = summary[summary['fitness'].notna()]
summary = summary.sort_values(by='fitness', ascending=False)

summary.to_csv('../results/summary.csv', sep=';')

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
            palette = sns.color_palette('deep'))

# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=55)

sns.despine()
# save and show 
plt.savefig('../results/descriptive/combined_fitness_box.png', dpi=1000)
plt.show()


# %%
new = [part for part in processed_sardinia if part.uns['warning'] is None]
for part in new:
    if part.obs.fitness.max()>0.8:
        break

fig, axes = plt.subplots(1, 2, figsize=(10,5))
for mut in part:
 sns.lineplot(x=mut.var.time_points,
    y=mut.X.flatten(),
    label=mut.obs.p_key,
    ax=axes[0])
axes[0].set_ylabel('Age (years)')
axes[0].set_xlabel('VAF')

model = part.uns['optimal_model']
output = model['posterior']
cs = model['clonal_structure']
ms = model['mutation_structure']

ps = []
for structure in cs:
    ps.append([part[i].obs.p_key.values[0] for i in structure])


s_range = model['s_range']
# normalisation constant
norm_max = np.max(output, axis=0)
# Plot
i =1

mut_colors = [sns.color_palette()[0], sns.color_palette()[2]]
for i in range(len(cs)):
    label = f'clone {i+1}\n' + '\n'.join(map(str, ps[i]))
    
    sns.lineplot(x=s_range,
    y=output[:, i]/ norm_max[i],
    label=label,
    ax=axes[1],
    color=mut_colors[i])
    axes[1].set_xlabel('Fitness')
    axes[1].set_ylabel('Normalised probability')

sns.despine()

axes[0].axhline(y=0.02, color='grey', linestyle='--')

plt.savefig('../results/sample_participant_2.png', dpi=1000)


# %%
i = 171
part = cohort[i]
fig, axes = plt.subplots(1, 2, figsize=(10,5))
for mut in part:
    sns.lineplot(x=mut.var.time_points,
    y=mut.X.flatten(),
    label=mut.obs.index,
    ax=axes[0])

axes[0].set_ylabel('Age (years)')
axes[0].set_xlabel('VAF')

model = part.uns['optimal_model']
output = model['posterior']
cs = model['clonal_structure']
ms = model['mutation_structure']

ps = []
for structure in cs:
    ps.append([part[i].obs.p_key.values[0] for i in structure])

s_range = model['s_range']
# normalisation constant
norm_max = np.max(output, axis=0)
# Plot
i =1
for i in range(len(cs)):
    label = f'clone {i+1}\n' + '\n'.join(map(str, ms[i]))
    
    sns.lineplot(x=s_range,
    y=output[:, i]/ norm_max[i],
    label=label,
    ax=axes[1])
    axes[1].set_xlabel('Fitness')
    axes[1].set_ylabel('Normalised probability')

sns.despine()

axes[0].axhline(y=0.02, color='grey', linestyle='--')

plt.savefig('../results/sample_participant.png', dpi=1000)
