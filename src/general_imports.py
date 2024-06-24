import anndata as ad
import pickle as pk

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



def plot_part(part, cohort=None):
   for i in part:
      if cohort == 'Fabre':
            sns.lineplot(x=i.var.time_points, y =i.X.flatten(), label=i.obs.PreferredSymbol.values)
      else:
       sns.lineplot(x=i.var.time_points, y =i.X.flatten(), label=list(i.obs.index))
   plt.show()