import anndata as ad
import pickle as pk

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



def plot_part(part):
    for i in part:
       sns.lineplot(x=i.var.time_points, y =i.X.flatten(), label=i.obs.PreferredSymbol.values)
    plt.show()