# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

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
# Figure 1A
# Plot one participant from each cohort
cohort_high = [part for part in cohort if (part.shape[0]>2) & (part.uns['cohort']=='sardiNIA')]
combined_posterior_plot(cohort_high[9])
plt.savefig('../plots/Figure 1/example_sardiNIA.png')
plt.savefig('../plots/Figure 1/example_sardiNIA.svg')
plt.show()
plt.clf()
 
cohort_high = [part for part in cohort if (part.shape[0]>2) & (part.uns['cohort']=='WHI')]
combined_posterior_plot(cohort_high[14])
plt.savefig('../plots/Figure 1/example_WHI.png')
plt.savefig('../plots/Figure 1/example_WHI.svg')
plt.show()
plt.clf()

cohort_high = [part for part in cohort if (part.shape[0]>2) & (part.uns['cohort']=='LBC')]
combined_posterior_plot(cohort_high[3])
plt.savefig('../plots/Figure 1/example_LBC.png')
plt.savefig('../plots/Figure 1/example_LBC.svg')

plt.show()
plt.clf()

# %%
# Figure 1B
def plot_normalized_kde(data, x=None, y=None, hue=None, ax=None, palette=None, **kwargs):
    """
    Plot KDE where each curve's maximum height is normalized to 1.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    x : str, optional
        Column name for x-axis (for horizontal KDE)
    y : str, optional
        Column name for y-axis (for vertical KDE)
    hue : str
        Column name for grouping
    ax : matplotlib.axes.Axes
        Axes object to plot on
    palette : dict
        Color mapping for hue groups
    """
    if ax is None:
        ax = plt.gca()
        
    # Get unique values in hue column
    hue_vals = data[hue].unique()
    
    for hue_val in hue_vals:
        subset = data[data[hue] == hue_val]
        
        if x is not None:
            # Horizontal KDE
            density = sns.kdeplot(data=subset, x=x, ax=ax, color=palette[hue_val],
                                fill=False, legend=False, **kwargs)
            line = density.lines[-1]
            # Get the y values and normalize them
            y_values = line.get_ydata()
            y_values = y_values / y_values.max()
            line.set_ydata(y_values)
            
        if y is not None:
            # Vertical KDE
            density = sns.kdeplot(data=subset, y=y, ax=ax, color=palette[hue_val],
                                fill=False, legend=False, **kwargs)
            line = density.lines[-1]
            # Get the x values and normalize them
            x_values = line.get_xdata()
            x_values = x_values / x_values.max()
            line.set_xdata(x_values)

# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    summary['part_age_tp_0'], summary['fitness'])

# Calculate R-squared
r_squared = r_value**2
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
# sns.kdeplot(data=summary, x='part_age_tp_0', hue='cohort',
#             palette=cohort_color_dict, 
#             fill=False, legend=False, ax=ax_kde_top)
plot_normalized_kde(data=summary, x='part_age_tp_0', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_top)

# Plot right KDE in both top and bottom parts
for ax in [ax_kde_right_top, ax_kde_right_bottom]:
    # sns.kdeplot(data=summary, y='fitness', hue='cohort',
    #             palette=cohort_color_dict, 
    #             fill=False, legend=False, ax=ax)
    plot_normalized_kde(data=summary, y='fitness', hue='cohort',
                   palette=cohort_color_dict, ax=ax)
    
# Plot scatter plots
for ax in [ax_main_top, ax_main_bottom]:
    sns.scatterplot(data=summary, x='part_age_tp_0', y='fitness', 
                   hue='cohort', alpha=0.7, 
                   palette=cohort_color_dict, ax=ax)
        
    sns.regplot(data=summary, x='part_age_tp_0', y='fitness',
                scatter=False, order=1, color='tab:grey', ax=ax)

# Style the top KDE
ax_kde_top.set_xlim(48, 100)
ax_kde_top.set_ylim(0.05, 1.05)

ax_kde_top.set_ylabel('')
ax_kde_top.set_xlabel('')
ax_kde_top.tick_params(labelbottom=False, labelleft=False, left=False)
sns.despine(ax=ax_kde_top, top=True, left=True, right=True)

# Style the right KDEs
ax_kde_right_top.set_ylim(0.94, 1.0)
ax_kde_right_top.set_xlim(0.05, 1.05)
ax_kde_right_bottom.set_ylim(-0.02, 0.65)
ax_kde_right_bottom.set_xlim(0.05, 1.05)

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

plt.savefig('../plots/Figure 1/age_vs_fitness_scatter.png', transparent=True, dpi=1000)
plt.savefig('../plots/Figure 1/age_vs_fitness_scatter.svg')

# %%

# Figure 1C
# import cosmic data
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

# Create figure with broken axis
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 6), 
                            gridspec_kw={'height_ratios': [1,   6], 'hspace': 0.05})

# Plot the same data on both axes
sns.boxplot(data=summary_filtered, x='PreferredSymbol', y='fitness',
            showfliers=False, palette=sns.color_palette(), legend=False,
            ax=ax1)

sns.boxplot(data=summary_filtered, x='PreferredSymbol', y='fitness',
            showfliers=False, palette=sns.color_palette(), ax=ax2)

# Plot stripplot for outliers on both axes
sns.stripplot(data=outliers, x='PreferredSymbol', y='fitness',
             hue='reports', jitter=0.8, hue_order=hue_order, dodge=True, 
             marker='o', size=7, edgecolor='none', ax=ax1)

sns.stripplot(data=outliers, x='PreferredSymbol', y='fitness',
             hue='reports', jitter=0.8, hue_order=hue_order, dodge=True, 
             marker='o', size=7, edgecolor='none', ax=ax2)

# Set different y-axis limits for each subplot
ax1.set_ylim(0.94, 1.0)  # Upper subplot
ax2.set_ylim(0, 0.65)     # Lower subplot
ax1.set_ylim(0.94, 1.0)  # Upper subplot
ax2.set_ylim(0, 0.65)     # Lower subplot

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax1.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax1.set_yticks([0.95, 1])
ax1.set_ylabel("")

ax2.spines.top.set_visible(False)
ax2.xaxis.tick_bottom()
ax2.spines.right.set_visible(False)
ax2.set_ylabel("Fitness", fontsize=15)
ax2.set_xlabel("")
ax1.tick_params(axis='both', which='major', labelsize=12)  # Increase top subplot tick size
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(title='Reports', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set x-tick labels with right anchor
plt.setp(ax2.get_xticklabels(), rotation=55, ha='right', rotation_mode="anchor")


d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)

plt.xticks(rotation=55)
# Move legend to top right of ax1
ax1.legend(title='Reports', bbox_to_anchor=(1.02, 1), loc='best')
ax2.get_legend().remove()  # Remove legend from bottom plot

change_boxplot_opacity(ax2, 0.7)

# Adjust layout to prevent label cutoff
# plt.tight_layout()
plt.xlim(-1)

# Save the figure
plt.savefig('../plots/Figure 1/cosmic_combined_fitness_box_broken.png', 
            dpi=1000, transparent=True, bbox_inches='tight')

# Save the figure
plt.savefig('../plots/Figure 1/cosmic_combined_fitness_box_broken.svg', 
            bbox_inches='tight')
plt.show()

# %%

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

plt.savefig('../plots/Figure 1/kruskal_wallis_heatmap.png', transparent=True, dpi=300)
plt.savefig('../plots/Figure 1/kruskal_wallis_heatmap.svg')

# %%

max_order = 5

# Create filtered dataset for only relevant genes
filtered_df = summary[summary.PreferredSymbol.isin(['DNMT3A', 'TET2','PPM1D', 'ASXL1'])]
summary_copy = summary.copy()
summary_copy['PreferredSymbol'] = 'ALL'
filtered_df = pd.concat([filtered_df, summary_copy])

filtered_df = filtered_df[filtered_df.overall_order<max_order]
filtered_df = filtered_df.sort_values(by='PreferredSymbol')
filtered_df = filtered_df[filtered_df.fitness<0.6]


# %%
fig, ax = plt.subplots()

# Create boxplot
new_palette = [sns.color_palette()[7], *sns.color_palette()]
sns.boxplot(data=filtered_df, 
            x='overall_order', 
            y='fitness',
            showfliers=False,
            palette=new_palette,
            hue='PreferredSymbol', 
            ax=ax)

# Add statistical annotations
# Filter only 'ALL' data for statistics
all_data = filtered_df[filtered_df.PreferredSymbol == 'ALL']

for i in range(1,4):
    subset = filtered_df[filtered_df.overall_order == i]
    q1 = subset['fitness'].quantile(0.25)
    q3 = subset['fitness'].quantile(0.75)
    iqr = q3 - q1
    upper_bound_0 = q3 + 1.5 * iqr


    subset = filtered_df[filtered_df.overall_order == i+1]
    q1 = subset['fitness'].quantile(0.25)
    q3 = subset['fitness'].quantile(0.75)
    iqr = q3 - q1
    upper_bound_1 = q3 + 1.5 * iqr

    x1, x2 = -0.33+i-1, -0.33+i
    y, h, col = min(upper_bound_0, upper_bound_1)+0.01, 0.01, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

# Add connecting lines between ALL boxplots
# Get positions of 'ALL' boxplots
all_medians = []
for order in range(max_order):
    order_data = all_data[all_data.overall_order == order].fitness
    if not order_data.empty:
        all_medians.append(order_data.median())

# Plot connecting lines
x_positions = np.array(range(len(all_medians)))-0.
ax.plot(x_positions-0.33, all_medians, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc=(-0.1,1.1), ncol=3)
sns.despine()


for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.4))

ax.set_ylabel('Fitness')
ax.set_xlabel('Overall mutation order')
ax.set_xticklabels(['1st', '2nd', '3rd', '4th'])
plt.savefig('../plots/Figure 1/mutation_order.svg')
# %%
# Create the subplot layout
fig, ax = plt.subplots(1, 2, gridspec_kw={'wspace': 0.05, 'width_ratios': [5, 1]})

# Create boxplot
new_palette = [sns.color_palette()[7], *sns.color_palette()]
sns.boxplot(data=filtered_df, 
            x='overall_order', 
            y='fitness',
            showfliers=False,
            palette=new_palette,
            hue='PreferredSymbol', 
            ax=ax[0])

# Add statistical annotations
# Filter only 'ALL' data for statistics
all_data = filtered_df[filtered_df.PreferredSymbol == 'ALL']

x1, x2 = 0, 1
y, h, col = all_data[all_data.overall_order.isin([0,1])].fitness.max()-0.01, 0.01, 'k'
ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax[0].text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

# Add connecting lines between ALL boxplots
# Get positions of 'ALL' boxplots
all_medians = []
for order in range(max_order):
    order_data = all_data[all_data.overall_order == order].fitness
    if not order_data.empty:
        all_medians.append(order_data.median())

# Plot connecting lines
x_positions = range(len(all_medians))
ax[0].plot(x_positions, all_medians, color='gray', linestyle='--', alpha=0.5)

# Create density plot
div_palette = sns.diverging_palette(h_neg=240, h_pos=20, s=70, l=35, sep=1, center='light', n=max_order-1)
sns.kdeplot(filtered_df[filtered_df.overall_order < max_order],
            y='fitness',
            hue='overall_order',
            palette=div_palette,
            common_norm=False,
            fill=True, 
            ax=ax[1])

# Adjust plot aesthetics
ax[0].set_ylim(-0.01, 0.6)
ax[1].set_ylim(-0.01, 0.6)
sns.despine()

# Hide the spines between ax and ax2
ax[1].tick_params(labelleft=False, labelbottom=False, bottom=False)
ax[1].set_ylabel('')
ax[0].legend(loc="upper left", ncol=3)
# Adjust layout
# plt.tight_layout()
plt.savefig('../plots/Figure 1/fitness_by_overall_order_selected_genes.png', transparent=True, dpi=200)
plt.savefig('../plots/Figure 1/fitness_by_overall_order_selected_genes.svg')

# %%
sns.boxplot(data=filtered_df, x='overall_order', y='fitness',
            showfliers=False,
            hue='PreferredSymbol', ax=ax[0])

import statsmodels.api as sm
import statsmodels.formula.api as smf

mod = smf.ols(formula='fitness ~ scale(overall_order) + scale(clipped_init_age)', data=summary)
res = mod.fit()
print(res.summary())



# %%

cohort_color_dict
# %%
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)
sns.scatterplot(summary, x='clipped_init_age', y='fitness', hue='cohort', palette=cohort_color_dict)
# sns.scatterplot(summary, x='clipped_init_age', y='fitness', hue='part_age_tp_0')

# Customize the plot
plt.xlabel('ATMA', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.ylim(0, 0.6)
sns.despine()

max_age = part_summary.age_wave_1.max()
N = 100_000
ATMA = np.linspace(1, 70, 70)
y = np.log(0.01*N)/(max_age - ATMA)
sns.lineplot(x=ATMA, y=y, color='grey', linestyle='--', label='Minimum\ndetectable fitness')

# Add the filled area under the curve
plt.fill_between(ATMA, y, 0, color='grey', alpha=0.1)  # alpha controls transparency


# Show the plot
plt.tight_layout()
plt.savefig('../plots/Figure 1/fitness_vs_ATMA.png', transparent=True, dpi=200)
plt.savefig('../plots/Figure 1/fitness_vs_ATMA.svg')

plt.show()
plt.clf()
# %%
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)
sns.scatterplot(summary, x='clipped_init_age', y='fitness', hue='part_age_tp_0', palette='coolwarm')
# sns.scatterplot(summary, x='clipped_init_age', y='fitness', hue='part_age_tp_0')

# Customize the plot
plt.xlabel('ATMA', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.ylim(0, 0.6)
sns.despine()

max_age = part_summary.age_wave_1.max()
N = 100_000
ATMA = np.linspace(1, 70, 70)
y = np.log(0.01*N)/(max_age - ATMA)
sns.lineplot(x=ATMA, y=y, color='grey', linestyle='--', label='Minimum\ndetectable fitness')

# Add the filled area under the curve
plt.fill_between(ATMA, y, 0, color='grey', alpha=0.1)  # alpha controls transparency


# Show the plot
plt.tight_layout()
plt.savefig('../plots/Figure 1/fitness_vs_ATMA_by age.png', transparent= True, dpi=200)
plt.savefig('../plots/Figure 1/fitness_vs_ATMA_by_age.svg')

plt.show()
plt.clf()

# %%
