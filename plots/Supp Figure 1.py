# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

import matplotlib.gridspec as gridspec
import pickle as pk

from scipy import stats

# # Import results
# with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
#     cohort = pk.load(f)

summary = pd.read_csv('../results/mutation_df.csv')
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)

part_summary = pd.read_csv('../results/participant_df.csv')
part_summary = part_summary.sort_values(by='cohort')

# %%

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
    summary['part_age_tp_0'], summary['mut_max_VAF'])

# Calculate R-squared
r_squared = r_value**2

# Create main figure
fig = plt.figure(figsize=(8,7))

# Divide the figure into main and margin sections
outer_gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[6, 1],
                           height_ratios=[1, 6],
                           hspace=0.1, wspace=0.1)

# Create main subplot and KDE subplots
ax_kde_top = fig.add_subplot(outer_gs[0, 0])      # Top KDE
ax_kde_right = fig.add_subplot(outer_gs[1, 1])    # Right KDE
ax_main = fig.add_subplot(outer_gs[1, 0])         # Main scatter plot

# Plot KDEs
plot_normalized_kde(data=summary, x='part_age_tp_0', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_top)

# Plot right KDE
plot_normalized_kde(data=summary, y='mut_max_VAF', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_right)

# Plot scatter plot
sns.scatterplot(data=summary, x='part_age_tp_0', y='mut_max_VAF', 
               hue='cohort', alpha=0.7, 
               palette=cohort_color_dict, ax=ax_main)
    
sns.regplot(data=summary, x='part_age_tp_0', y='mut_max_VAF',
            scatter=False, order=1, color='tab:grey', ax=ax_main)

# Style the top KDE
ax_kde_top.set_xlim(48, 100)
ax_kde_top.set_ylim(0.05, 1.05)
ax_kde_top.set_ylabel('')
ax_kde_top.set_xlabel('')
ax_kde_top.tick_params(labelbottom=False, labelleft=False, left=False)
sns.despine(ax=ax_kde_top, top=True, left=True, right=True)

# Style the right KDE
ax_kde_right.set_xlim(0.05, 1.05)
ax_kde_right.set_ylim(-0.02, 1.0)  # Adjusted to show full range
ax_kde_right.set_ylabel('')
ax_kde_right.set_xlabel('')
ax_kde_right.tick_params(labelbottom=False, labelleft=False, bottom=False)
sns.despine(ax=ax_kde_right, top=True, right=True, bottom=True)

# Style main plot
ax_main.set_xlim(48, 100)
# ax_main.set_ylim(-0.02, 1.0)  # Adjusted to show full range
ax_main.spines.right.set_visible(False)
ax_main.spines.top.set_visible(False)
ax_main.set_ylabel("Max VAF", fontsize=12)
ax_main.set_xlabel("Age at first observation", fontsize=12)

# Remove duplicate legend from main plot
ax_main.legend(title='Cohort', bbox_to_anchor=(1.02, 1), loc='best')

# Add R-squared and p-value to the plot
ax_main.text(x=60, y=0.3,  # Adjusted y position for new scale
             s=f'Max VAF vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})',
             fontsize=12,
             color='tab:grey')

plt.savefig('../plots/Supp Figure 1/age_vs_VAF_scatter.png', 
            transparent=True, dpi=1000)
plt.savefig('../plots/Supp Figure 1/age_vs_VAF_scatter.svg')

# %%
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)
rdgn = sns.diverging_palette(h_neg=220, h_pos=20, s=70, l=60, sep=1, as_cmap=True)

sns.scatterplot(summary, x='clipped_init_age', y='fitness', hue='part_age_tp_0', palette=rdgn)
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
plt.savefig('../plots/Supp Figure 1/fitness_vs_ATMA_by age.png', transparent= True, dpi=200)
plt.savefig('../plots/Supp Figure 1/fitness_vs_ATMA_by_age.svg')

plt.show()
plt.clf()

# %%
