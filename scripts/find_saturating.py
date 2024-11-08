import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

from scipy import stats

sns.set_palette(custom_palette['plotly_d3'])

# Import results
with open('../exports/final_participant_list.pk', 'rb') as f:
    cohort = pk.load(f)


# export participant_data
with open('../exports/final_participant_list.pk', 'rb') as f:
    cohort = pk.load(f)

summary = pd.read_csv('../results/mutation_df.csv', index_col=0)

# %%

JAK2 = summary[(summary.cohort=='LBC') & (summary.PreferredSymbol=='JAK2')].copy()

id_list = JAK2.participant_id.unique()

for part in cohort:
    if part.uns['participant_id'] in id_list:
        plot_part(part)
        plt.title(part.uns['participant_id'])