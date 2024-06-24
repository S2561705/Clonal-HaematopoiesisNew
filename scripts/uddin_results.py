# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
import plotly.express as px
import pickle

with open('../exports/Uddin_processed_2024_04_02.pk', 'rb') as f:
    participant_list = pk.load(f)

for part in participant_list:
    part.obs['Sample ID'] = part.uns['participant_id']

# %%

summary = pd.concat([part.obs for part in participant_list])
summary = summary[summary['fitness'].notna()]
summary = summary.sort_values(by='fitness', ascending=False)

px.box(summary, x='Gene', y='fitness', points='all', hover_data=['Sample ID', 'Protein Change'])

# %%
# drop zero fitness
summary_drop = summary[summary.fitness>0.01]
px.box(summary_drop , x='Gene', y='fitness', points='all')


# %%
# Check gene
summary[summary.Gene == 'SF3B1']

# %%
# plot part including mutation
mut = "JAK2 p.V617F"
for part in participant_list:
    if mut in part.obs.index:
        plot_part(part)
        plot_optimal_model(part)
# %%
# plot part including mutation
for part in participant_list:
    if part.uns['participant_id'] == 'Sample 162':
        plot_part(part)
        plot_optimal_model(part)




# %%
