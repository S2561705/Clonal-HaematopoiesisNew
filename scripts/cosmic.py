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
    
summary = pd.concat([part.obs for part in participant_list])

cosmic_df = pd.read_csv('../data/Site_mutsWed Apr 3 10 00 12 2024.csv')
# %%

summary = summary[summary['fitness'].notna()]
summary = summary.sort_values(by='fitness', ascending=False)

cosmic_df ['Protein Change'] = cosmic_df['Gene Name'] + ' ' + cosmic_df['AA Mutation']

cosmic_intersection = summary[summary.index.isin(list(cosmic_df['Protein Change']))]

px.box(cosmic_intersection, x='Gene', y='fitness', points='all', hover_data=["Protein Change", 'Sample ID'])

for part in participant_list:
    if part.uns['participant_id'] == "Sample 65":
        plot_part(part)
        part.uns
        break
# %%
