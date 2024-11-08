# Import necessary libraries
import anndata as ad
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import pickle as pk
import json
import warnings

from anndata._core.aligned_df import ImplicitModificationWarning

warnings.filterwarnings("ignore", category=ImplicitModificationWarning, message="Transforming to str index.")
pd.set_option('display.max_columns', 500)

# Note: The following line is commented out, consider removing if unused
# # set color palette
# sns.set_palette('tab10')

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path) as json_file:
        return json.load(json_file)

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as pkl_file:
        return pk.load(pkl_file)

def dump_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as f:
        pk.dump(data, f)

def plot_part(part, cohort=None, corrected=False):
    """
    Plot data for each item in the 'part' iterable.
    
    Args:
    part: Iterable containing data to plot
    cohort: Optional parameter (unused in current implementation)
    corrected: Boolean flag to determine which data to plot (default or corrected)
    """
    for i in part:
        if corrected is False:
            # Plot uncorrected data
            sns.lineplot(x=i.var.time_points, y=i.X.flatten(), label=i.obs.index)
        
        if corrected is True:
            # Plot corrected data
            sns.lineplot(x=i.var.time_points, y=i.layers['corrected_VAF'].flatten(), label=list(i.obs.index))
    plt.show()

def plot_prior(prior):
    """
    Plot prior data.
    
    Args:
    prior: A tuple or list containing x and y values for the prior
    """
    sns.lineplot(x=prior[0], y=prior[1])

# Define custom color palettes
custom_palette = {
    'plotly_t10': px.colors.qualitative.T10,
    'plotly_d3': px.colors.qualitative.D3
}

cohort_color_dict = {
    # 'sardiNIA': '#f8cbad',
    # 'WHI': '#9dc3e6',
    # 'LBC': '#70ad47'
    'sardiNIA':sns.color_palette('deep')[3],
    'LBC': sns.color_palette('deep')[4],
    'WHI': sns.color_palette('deep')[9]
    # 'sardiNIA': '#0072B2',
    # 'LBC': '#D55E00',
    # 'WHI': '#009E73'

}

def change_boxplot_opacity(ax, alpha=0.8):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))
