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
# Supp Figure 2A
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
# %%
# Age vs MACS
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    part_summary['age_wave_1'], part_summary['max_size_prediction_120_z_score'])

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
plot_normalized_kde(data=part_summary, x='age_wave_1', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_top)

# Plot right KDE
plot_normalized_kde(data=part_summary, y='max_size_prediction_120_z_score', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_right)

# Plot scatter plot
sns.scatterplot(data=part_summary, x='age_wave_1', y='max_size_prediction_120_z_score', 
               hue='cohort', alpha=0.7, 
               palette=cohort_color_dict, ax=ax_main)
    
sns.regplot(data=part_summary, x='age_wave_1', y='max_size_prediction_120_z_score',
            scatter=False, order=1, color='tab:grey', ax=ax_main)

# Style the top KDE
ax_kde_top.set_xlim(48, 95)
ax_kde_top.set_ylim(0.05, 1.05)
ax_kde_top.set_ylabel('')
ax_kde_top.set_xlabel('')
ax_kde_top.tick_params(labelbottom=False, labelleft=False, left=False)
sns.despine(ax=ax_kde_top, top=True, left=True, right=True)

# Style the right KDE
ax_kde_right.set_xlim(0.05, 1.05)
ax_kde_right.set_ylim(-2.1, 2.5)  # Adjusted to show full range
ax_kde_right.set_ylabel('')
ax_kde_right.set_xlabel('')
ax_kde_right.tick_params(labelbottom=False, labelleft=False, bottom=False)
sns.despine(ax=ax_kde_right, top=True, right=True, bottom=True)

# Style main plot
ax_main.set_xlim(48, 95)
ax_main.set_ylim(-2.1, 2.5)  # Adjusted to show full range
ax_main.spines.right.set_visible(False)
ax_main.spines.top.set_visible(False)
ax_main.set_ylabel("MACS 120", fontsize=12)
ax_main.set_xlabel("Age at first observation", fontsize=12)

# Remove duplicate legend from main plot
ax_main.legend(title='Cohort', bbox_to_anchor=(1.02, 1), loc='best')

# Add R-squared and p-value to the plot
ax_main.text(x=60, y=0.3,  # Adjusted y position for new scale
             s=f'MACS 120 vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})',
             fontsize=12,
             color='tab:grey')

plt.savefig('../plots/Supp Figure 2/age_vs_MACS_scatter.png', 
            transparent=True, dpi=1000)
plt.savefig('../plots/Supp Figure 2/age_vs_MACS_scatter.svg', transparent=True)

# %%
# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    part_summary['age_wave_1'], part_summary['max_fitness_z_score'])

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
plot_normalized_kde(data=part_summary, x='age_wave_1', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_top)

# Plot right KDE
plot_normalized_kde(data=part_summary, y='max_fitness_z_score', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_right)

# Plot scatter plot
sns.scatterplot(data=part_summary, x='age_wave_1', y='max_fitness_z_score', 
               hue='cohort', alpha=0.7, 
               palette=cohort_color_dict, ax=ax_main)
    
sns.regplot(data=part_summary, x='age_wave_1', y='max_fitness_z_score',
            scatter=False, order=1, color='tab:grey', ax=ax_main)

# Style the top KDE
ax_kde_top.set_xlim(48, 95)
ax_kde_top.set_ylim(0.05, 1.05)
ax_kde_top.set_ylabel('')
ax_kde_top.set_xlabel('')
ax_kde_top.tick_params(labelbottom=False, labelleft=False, left=False)
sns.despine(ax=ax_kde_top, top=True, left=True, right=True)

# Style the right KDE
ax_kde_right.set_xlim(0.05, 1.05)
ax_kde_right.set_ylim(-2.1, 2.5)  # Adjusted to show full range
ax_kde_right.set_ylabel('')
ax_kde_right.set_xlabel('')
ax_kde_right.tick_params(labelbottom=False, labelleft=False, bottom=False)
sns.despine(ax=ax_kde_right, top=True, right=True, bottom=True)

# Style main plot
ax_main.set_xlim(48, 95)
ax_main.set_ylim(-2.1, 2.5)  # Adjusted to show full range
ax_main.spines.right.set_visible(False)
ax_main.spines.top.set_visible(False)
ax_main.set_ylabel("Max Fitness (z-score)", fontsize=12)
ax_main.set_xlabel("Age at first observation", fontsize=12)

# Remove duplicate legend from main plot
ax_main.legend(title='Cohort', bbox_to_anchor=(1.02, 1), loc='best')

# Add R-squared and p-value to the plot
ax_main.text(x=60, y=0.3,  # Adjusted y position for new scale
             s=f'Max fitness vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})',
             fontsize=12,
             color='tab:grey')

plt.savefig('../plots/Supp Figure 2/age_vs_max_fitness_scatter.png', 
            transparent=True, dpi=1000)
plt.savefig('../plots/Supp Figure 2/age_vs_max_fitness_scatter.svg', transparent=True)

# %%
# Supp Figure 2B

dead_df = part_summary[part_summary.dead==1].copy()

# Age vs Fitness
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    dead_df['age_wave_1'], dead_df['max_size_prediction_120_z_score'])

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
plot_normalized_kde(data=dead_df, x='age_wave_1', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_top)

# Plot right KDE
plot_normalized_kde(data=dead_df, y='max_size_prediction_120_z_score', hue='cohort',
                   palette=cohort_color_dict, ax=ax_kde_right)

# Plot scatter plot
sns.scatterplot(data=dead_df, x='age_wave_1', y='max_size_prediction_120_z_score', 
               hue='cohort', alpha=0.7, 
               palette=cohort_color_dict, ax=ax_main)
    
sns.regplot(data=dead_df, x='age_wave_1', y='max_size_prediction_120_z_score',
            scatter=False, order=1, color='tab:grey', ax=ax_main)

# Style the top KDE
ax_kde_top.set_xlim(48, 95)
ax_kde_top.set_ylim(0.05, 1.05)
ax_kde_top.set_ylabel('')
ax_kde_top.set_xlabel('')
ax_kde_top.tick_params(labelbottom=False, labelleft=False, left=False)
sns.despine(ax=ax_kde_top, top=True, left=True, right=True)

# Style the right KDE
ax_kde_right.set_xlim(0.05, 1.05)
ax_kde_right.set_ylim(-2.1, 2.5)  # Adjusted to show full range
ax_kde_right.set_ylabel('')
ax_kde_right.set_xlabel('')
ax_kde_right.tick_params(labelbottom=False, labelleft=False, bottom=False)
sns.despine(ax=ax_kde_right, top=True, right=True, bottom=True)

# Style main plot
ax_main.set_xlim(48, 95)
ax_main.set_ylim(-2.1, 2.5)  # Adjusted to show full range
ax_main.spines.right.set_visible(False)
ax_main.spines.top.set_visible(False)
ax_main.set_ylabel("MACS", fontsize=12)
ax_main.set_xlabel("Age at first observation", fontsize=12)

# Remove duplicate legend from main plot
ax_main.legend(title='Cohort', bbox_to_anchor=(1.02, 1), loc='best')

# Add R-squared and p-value to the plot
ax_main.text(x=60, y=0.3,  # Adjusted y position for new scale
             s=f'MACS vs Age (R² = {r_squared:.3f}, p = {p_value:.3e})',
             fontsize=12,
             color='tab:grey')

plt.savefig('../plots/Supp Figure 2/age_vs_MACS_scatter.png', 
            transparent=True, dpi=1000)
plt.savefig('../plots/Supp Figure 2/age_vs_MACS_scatter.svg', transparent=True)

# %%
discr_regr = stats.linregress(dead_df.age_wave_1, dead_df.from_wave_1)

dead_df['surv_discrepancy'] = dead_df.from_wave_1 - (discr_regr.slope*dead_df.age_wave_1 + discr_regr.intercept)

dead_df['MACS_120_bins'] = pd.cut(dead_df.max_size_prediction_120_z_score, [-np.inf,-1,0,1, np.inf])
sns.jointplot(dead_df, x='age_wave_1', y='surv_discrepancy', hue='MACS_120_bins')



# %%
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_anova(df):
    """
    Perform one-way ANOVA test and create visualizations for the results
    
    Parameters:
    df: DataFrame containing MACS_120_bins and surv_discrepancy columns
    """
    # Perform one-way ANOVA
    groups = [group for _, group in df.groupby('MACS_120_bins')['surv_discrepancy']]
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    sns.boxplot(data=df, x='MACS_120_bins', y='surv_discrepancy', ax=ax1)
    ax1.set_title('Distribution of Z-Scores Across MACS Bins')
    ax1.set_xlabel('MACS 120 Bins')
    ax1.set_ylabel('Survival discrepancy')
    
    # Violin plot with individual points
    sns.violinplot(data=df, x='MACS_120_bins', y='surv_discrepancy', ax=ax2)
    sns.swarmplot(data=df, x='MACS_120_bins', y='surv_discrepancy', 
                  color='white', alpha=0.5, size=3, ax=ax2)
    ax2.set_title('Distribution with Individual Points')
    ax2.set_xlabel('MACS 120 Bins')
    ax2.set_ylabel('Survival_discrepancy')
    
    # Add ANOVA results as text
    plt.figtext(0.02, 0.02, f'One-way ANOVA: F-statistic = {f_stat:.3f}, p-value = {p_value:.3e}',
                fontsize=10, ha='left')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics by Bin:")
    print(df.groupby('MACS_120_bins')['surv_discrepancy'].describe())
    
    # Perform Tukey's HSD test if ANOVA is significant
    if p_value < 0.05:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(df['surv_discrepancy'], 
                                 df['MACS_120_bins'])
        print("\nTukey's HSD Test Results:")
        print(tukey)

# Usage example:
analyze_and_plot_anova(dead_df)
# %%

sns.regplot(grouped_df, y='fitness', x='clipped_init_age')
# %%
grouped_df = summary.groupby('PreferredSymbol').mean(numeric_only=True).sort_values(by='fitness')

grouped_df[grouped_df.index=='U2AF1']