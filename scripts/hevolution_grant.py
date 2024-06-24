# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import networkx as nx

import plotly.express as px
import pandas as pd
import pickle

with open('../exports/Uddin_processed_2024_04_02.pk', 'rb') as f:
    participant_list = pk.load(f)


# %%
for i, part in enumerate(participant_list):
    if part.shape[0]==4:
        print(i)
        plot_part(part)
# %%

part = participant_list[126]
plot_part(part)
plot_optimal_model(part)
part.uns['optimal_model']
part.uns['model_dict']

compute_clonal_models_prob_vec(part, filter_invalid=False)
# %%


model_dict = part.uns['model_dict'] 
model_dict['model_3'][0]

mutation_dict

import plotly.express as px
import plotly.graph_objects as go

x_labels=[]
for k, v in part.uns['model_dict'].items():
    x_labels.append(str([list(part.obs.index[list_mut]) for list_mut in v[0]]))

x_range = np.arange(len(x_labels))

fig, ax = plt.subplots()
sns.barplot(x=x_range, y=[v[1] for v in part.uns['model_dict'].values()], ax=ax)
ax.set_xticklabels(x_labels)
plt.xticks(rotation=70)



plot_part(part)
plot_optimal_model(part)