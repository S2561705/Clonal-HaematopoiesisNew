# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *
from src.blood_markers_aux import *
from src.aux import *

import matplotlib.gridspec as gridspec
import pickle as pk

from scipy import stats

# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

summary = pd.read_csv('../results/mutation_df.csv')
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)

part_summary = pd.read_csv('../results/participant_df.csv')

part_summary = part_summary.sort_values(by='cohort')

# %%

JAK2_ids = summary[summary.PreferredSymbol =='JAK2'].participant_id.unique()
JAK2_parts = [part for part in cohort if str(part.uns['participant_id']) in JAK2_ids]

# Figure 1A
# Plot one participant from each cohort

part= JAK2_parts[0]


# fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [2, 1], 'hspace':0.3})

# Create a deterministic plot of the model results
N = 100_000
cs = part.uns['optimal_model']['clonal_structure']

init = part.obs.init_age.to_numpy()
time_points = np.linspace(part.var.time_points.min()-10, part.var.time_points.max()+5)

# Compute time from mutation to observation
t_from_init = np.array(time_points) - init[:, None]
clipped_ages = np.where(part.obs.init_age.values<0, 0 , np.floor(part.obs.init_age))

# Calculate deterministic evolution of clone sizes
fitness = np.array(part.obs.fitness)[:, None]
sizes = np.exp(t_from_init * fitness)

sizes = np.where(sizes==0, 0, sizes)
max_ids = []
for i, clone in enumerate(cs):
    max_within_clone = np.argmax(sizes[clone].sum(axis=1))
    max_ids.append(clone[max_within_clone])
    # max_size = np.array(sizes[clone]).max(axis=0)
    # clone_size.append(max_size)

leading_sizes = sizes[max_ids, :]

# Calculate deterministic size of Variant Allele Frequency (VAF)
vaf_sizes = sizes / (2*(N + leading_sizes.sum(axis=0)))

# Plot the results
for i, size in enumerate(vaf_sizes):
    label = part[i].obs.p_key.values[0]
    if type(label) != str:
        label = part[i].obs.PreferredSymbol.values[0] + ' Splicing'

    color = sns.color_palette()[i]
    # add data points
    sns.scatterplot(x=part[i].var.time_points, y=part[i].X.flatten(), color=color,  label=label, ax=ax1)
    
    # add deterministic line
    sns.lineplot(x=time_points, y=size, color=color, linestyle='--', ax=ax1)
    # Add clipped age text at the beginning of the line
    first_point_x = time_points[15]
    first_point_y = size[0]
    ax1.text(first_point_x, first_point_y, f'ATMA: {clipped_ages[i]:.0f}', color=color, 
            ha='right', va='bottom', fontweight='bold', fontsize=10,
            bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', pad=1))
if part.uns['warning'] is not None:
    print('WARNING: ' + part.uns['warning'])

model = part.uns['optimal_model']
output = model['posterior']
cs = model['clonal_structure']
s_range = model['s_range']

# normalisation constant
norm_max = np.max(output, axis=0)

# Plot
for i in range(len(cs)):
    first_mut_in_cs_idx = cs[i][0]
    p_key_str = f''
    for k, j in enumerate(cs[i]):
        label = part[j].obs.p_key.values[0]
        if type(label) != str:
            label = part[j].obs.PreferredSymbol.values[0] + ' Splicing'
        if k == 0:
            p_key_str += f'{label}'
        if k > 0:
            p_key_str += f'\n{label}'

    color = sns.color_palette()[first_mut_in_cs_idx]
    sns.lineplot(x=s_range,
                y=output[:, i]/ norm_max[i],
                label=p_key_str,
                color=color,
                ax=ax2)

            # Fill the area under the line
    ax2.fill_between(s_range, 
                        output[:, i]/ norm_max[i], 
                        alpha=0.3,  # Adjust alpha for transparency
                        color=color)

sns.despine()  
ax1.title.set_text('Clonal dynamics inference')
ax2.title.set_text('Clonal Fitness')
ax1.set_xlabel('Age (yrs)')
ax1.set_ylabel('VAF')
ax2.set_xlabel('Fitness')
ax2.set_ylabel('Normalised likelihood')
# ax2.set_xscale('log')
# ax2.legend(loc='upper right', bbox_to_anchor=(1.7,1))
# ax2.set_yscale('log')


plt.savefig('../plots/Figure 3/JAK2_part.png')
plt.savefig('../plots/Figure 3/JAK2_part.svg')
plt.show()
plt.clf()
# %%
max_age = 0
for part in cohort:
    max_age = max(max_age, part.var.time_points.max())

for part in cohort:
    N = 100_000
    cs = part.uns['optimal_model']['clonal_structure']

    init = part.obs.init_age.to_numpy()
    time_points = max_age
    # Compute time from mutation to observation
    t_from_init = np.array(time_points) - init[:, None]
    clipped_ages = np.where(part.obs.init_age.values<0, 0 , np.floor(part.obs.init_age))

    # Calculate deterministic evolution of clone sizes
    fitness = np.array(part.obs.fitness)[:, None]
    sizes = np.exp(t_from_init * fitness)

    sizes = np.where(sizes==0, 0, sizes)
    max_ids = []
    for i, clone in enumerate(cs):
        max_within_clone = np.argmax(sizes[clone].sum(axis=1))
        max_ids.append(clone[max_within_clone])
        # max_size = np.array(sizes[clone]).max(axis=0)
        # clone_size.append(max_size)

    leading_sizes = sizes[max_ids, :]

    # Calculate deterministic size of Variant Allele Frequency (VAF)
    vaf_sizes = sizes / (2*(N + leading_sizes.sum(axis=0)))

    vaf_sizes_no_comp = sizes / (2*(N + sizes))

    part.obs['det_VAF_at_120'] = vaf_sizes
    part.obs['no_comp_VAF_at_120'] = vaf_sizes_no_comp
    part.obs['comp_vaf_diff'] = vaf_sizes_no_comp - vaf_sizes

# Export mutation-level dataframe
overlaping_columns = [set(part.obs.columns) for part in cohort]
overlaping_columns = list(set.intersection(*overlaping_columns))

summary = pd.concat([part.obs[overlaping_columns] for part in cohort])
summary = summary.sort_values(by='fitness', ascending=False)
normalise_parameter(summary, 'fitness')

sns.histplot(summary, x='comp_vaf_diff' )

# %%
for part in cohort:
    if part.uns['participant_id'] == 'LBC360636':
        break
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]})
# fig, ax = plt.subplots()
# Create a deterministic plot of the model results
N = 100_000
cs = part.uns['optimal_model']['clonal_structure']

init = part.obs.init_age.to_numpy()
time_points = np.linspace(part.var.time_points.min()-10, max_age, 25)

# Compute time from mutation to observation
t_from_init = np.array(time_points) - init[:, None]
clipped_ages = np.where(part.obs.init_age.values<0, 0 , np.floor(part.obs.init_age))

# Calculate deterministic evolution of clone sizes
fitness = np.array(part.obs.fitness)[:, None]
sizes = np.exp(t_from_init * fitness)

sizes = np.where(sizes==0, 0, sizes)
max_ids = []
for i, clone in enumerate(cs):
    max_within_clone = np.argmax(sizes[clone].sum(axis=1))
    max_ids.append(clone[max_within_clone])
    # max_size = np.array(sizes[clone]).max(axis=0)
    # clone_size.append(max_size)

leading_sizes = sizes[max_ids, :]

# Calculate deterministic size of Variant Allele Frequency (VAF)
vaf_sizes = sizes / (2*(N + leading_sizes.sum(axis=0)))

vaf_sizes_no_comp = sizes / (2*(N + sizes))


# Plot the results
for i, size in enumerate(vaf_sizes):
    label = part[i].obs.p_key.values[0]
    if type(label) != str:
        label = part[i].obs.PreferredSymbol.values[0] + ' Splicing'

    color = sns.color_palette()[i]
    # add data points
    sns.scatterplot(x=part[i].var.time_points, y=part[i].X.flatten(), color=color, s=100, label=label, ax=ax1)
    
    # add deterministic line
    sns.lineplot(x=time_points, y=size, color=color, linestyle='-', ax=ax1, label=label)
    # sns.lineplot(x=time_points, y=vaf_sizes_no_comp[i], color=color,  ax=ax, label=label)
    sns.scatterplot(x=time_points, y=vaf_sizes_no_comp[i], marker='+', color=color,  ax=ax1, label=label)
 


model = part.uns['optimal_model']
output = model['posterior']
cs = model['clonal_structure']
s_range = model['s_range']

# normalisation constant
norm_max = np.max(output, axis=0)

# Plot
for i in range(len(cs)):
    first_mut_in_cs_idx = cs[i][0]
    p_key_str = f''
    for k, j in enumerate(cs[i]):
        label = part[j].obs.p_key.values[0]
        if type(label) != str:
            label = part[j].obs.PreferredSymbol.values[0] + ' Splicing'
        if k == 0:
            p_key_str += f'{label}'
        if k > 0:
            p_key_str += f'\n{label}'

    color = sns.color_palette()[first_mut_in_cs_idx]
    sns.lineplot(x=s_range,
                y=output[:, i]/ norm_max[i],
                label=p_key_str,
                color=color,
                ax=ax2)

            # Fill the area under the line
    ax2.fill_between(s_range, 
                        output[:, i]/ norm_max[i], 
                        alpha=0.3,  # Adjust alpha for transparency
                        color=color)

sns.despine()  
ax1.title.set_text('Clonal dynamics inference')
ax2.title.set_text('Clonal Fitness')
ax1.set_xlabel('Age (yrs)')
ax1.set_ylabel('VAF')
ax2.set_xlabel('Fitness')
ax2.set_ylabel('Normalised likelihood')
plt.savefig('../plots/Figure 3/vaf_competition.svg')

# %%
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)
summary['log_fitness'] = np.log(summary.fitness)
sns.scatterplot(summary, x='log_fitness', y='comp_vaf_diff')
# %%

part_df = summary.groupby(by='participant_id').max(numeric_only=True)
mean_diff = part_df.comp_vaf_diff.mean()

# %%

fig, ax = plt.subplots()
# sns.kdeplot(part_df, x='comp_vaf_diff', ax=ax )
sns.histplot(part_df, x='comp_vaf_diff')

#mean_vaf_diff
plt.axvline(x=mean_diff, color='green', linestyle='--')
ax.text(x=5+0.02, y=50, s=f'mean VAF difference is {np.round(mean_diff,2)}', color='green')

# VAF_diff >0.1
plt.axvline(x=0.1, color=sns.color_palette()[1], linestyle='--')
percentage_part = 100*np.sum(part_df.comp_vaf_diff>0.1)/len(part_df)
ax.text(x=0.1+0.02, y=50, s=(f'{int(percentage_part)}% of participants show\n'
                                    f'differences in VAF > 0.1'), color=sns.color_palette()[1])

plt.yscale('log')
sns.despine()
plt.savefig('../plots/Figure 3/vaf_diff_histplot.svg')
# %%

gene_mask = summary.PreferredSymbol =='DNMT3A'
gene_data = summary[gene_mask].copy()
# gene_data[gene_data.clonal_structure_size == 1]

# Create separate KDE plots for each mutation, normalize, then combine
g = plt.figure()
unique_mutations = gene_data['clonal_structure_size'].unique()

for mutation in unique_mutations:
    subset = gene_data[gene_data['clonal_structure_size'] == mutation]
    
    # Get KDE values
    kde = stats.gaussian_kde(subset['fitness'])
    x_range = np.linspace(gene_data['fitness'].min(), gene_data['fitness'].max(), 200)
    kde_values = kde(x_range)
    
    # Normalize to maximum of 1
    kde_values_normalized = kde_values / kde_values.max()
    
    # Plot normalized KDE
    plt.plot(x_range, kde_values_normalized, label=str(mutation))

sns.despine()
plt.xlabel('Fitness')
plt.ylabel('Normalized Density')
plt.legend(title='Clonal Structure Size')
# %%
sns.regplot(gene_data , x='clonal_structure_size', y='fitness_z_score')
sns.despine()

# %%
import statsmodels.formula.api as smf
res = smf.ols(data =gene_data, formula='fitness_z_score ~ clonal_structure_size').fit()
res.summary()
# %%
# Assuming cosmic_df and summary_filtered dataframes are already loaded
fitness_threshold = 0.02
gene_count_threshold = 1

gene_counts = summary[summary.fitness > fitness_threshold].PreferredSymbol.value_counts()
gene_keep = gene_counts[gene_counts > gene_count_threshold].index

# summary_filtered
summary_filtered = summary[(summary.fitness > fitness_threshold)
                           & (summary.PreferredSymbol.isin(gene_keep))]


grouped_gene = summary_filtered.groupby(by='PreferredSymbol').mean(numeric_only=True)

import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text


# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    grouped_gene['clipped_init_age'], grouped_gene['fitness'])

# Calculate R-squared
r_squared = r_value**2
    # Create figure and axis
fig, ax = plt.subplots(figsize=(7,7))
    
# Create scatter plot
sns.scatterplot(data=grouped_gene, x='clipped_init_age', y='fitness')
sns.regplot(data=grouped_gene, x='clipped_init_age', y='fitness',
            color='orange', scatter=False)

plt.xlim(-20,70)
# Create list to store text objects
texts = []

# Add labels for each point
for x, y, label in zip(grouped_gene['clipped_init_age'], grouped_gene['fitness'], grouped_gene.index):
    texts.append(plt.text(x, y, label))

# Adjust text positions to avoid overlap
adjust_text(texts,
            arrowprops=dict(arrowstyle='-', color='gray', lw=1),
            expand_points=(1.5, 1.5),
            prevent_crossings=True,
            force_text=1,
        #    expand_axes=True,
            max_move=15,
            force_points=(0.1, 0.1))

# Customize plot
plt.xlabel('clipped_init_age')
plt.ylabel('fitness')
sns.despine()

# Add R-squared and p-value to the title
ax.text(x=-10, y=0.35,
                    s=f'Fitness vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})',
                    fontsize=12,
                    color='tab:grey'
                    )


plt.ylabel('Fitness')
plt.xlabel('ATMA')
plt.savefig('../plots/Supp Figure 3/fitness_vs_ATMA_by_gene.svg')
plt.savefig('../plots/Supp Figure 3/fitness_vs_ATMA_by_gene.png', transparent=True, dpi=200)
plt.show()
# %%

slope, intercept, r_value, p_value, std_err = stats.linregress(
    summary['clipped_init_age'], summary['fitness'])