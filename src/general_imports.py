import anndata as ad
import pickle as pk

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# # set color palette
# sns.set_palette('tab10')

def plot_part(part, cohort=None):
   for i in part:
      if cohort == 'Fabre':
            sns.lineplot(x=i.var.time_points, y =i.X.flatten(), label=i.obs.PreferredSymbol.values)
      else:
       sns.lineplot(x=i.var.time_points, y =i.X.flatten(), label=list(i.obs.index))
   plt.show()

import plotly.express as px

custom_palette= {'plotly_t10': px.colors.qualitative.T10,
                 'plotly_d3':px.colors.qualitative.D3}
