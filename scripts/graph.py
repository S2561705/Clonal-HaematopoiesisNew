# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import networkx as nx

import plotly.express as px
import pandas as pd
import pickle

with open('../exports/MDS/MDS_cohort_fitted.pk', 'rb') as f:
    participant_list = pk.load(f)

for part in participant_list:
    part.obs['Sample ID'] = part.uns['participant_id']
    

# filter out all participants with nan
nan_filtered = []
for part in participant_list:
    nan_counter = 0
    for idx in part.obs.index:
        if 'nan' in idx:
            nan_counter +=1
    
    if nan_counter ==0:
        nan_filtered.append(part)
    
summary = pd.concat([part.obs for part in nan_filtered])


# %%

summary = summary[summary['fitness'].notna()]
summary = summary.sort_values(by='fitness', ascending=False)

# %%
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations


seed = 13648  # Seed random number generators for reproducibility
# Plot graph network
G = nx.Graph()


# %%
# add nodes
print(summary.columns)

gene_dict = summary['PreferredSymbol'].value_counts().to_dict()


nodes = list(gene_dict.keys())
node_sizes = list(gene_dict.values())

# color map
cmap = plt.cm.plasma
for n, s in zip(nodes, node_sizes):
    G.add_node(n, size=np.log(s))


# %%

# Extract all co-ocurring mutations
all_edges = []

for i, row in summary.iterrows():
    if len(row.clonal_structure)>1:
        all_edges.extend(
            list(set(
                combinations([mut.split(' ')[0] for mut in row.clonal_structure],
                            2))))
        


edges_dict = Counter(all_edges)
edge_list = list(edges_dict.keys())
weights = list(edges_dict.values())

for edge, weight in edges_dict.items():
    weights = nx.get_edge_attributes(G,'weight').values()
    G.add_edge(edge[0], edge[1], weight=np.log(weight))

weights = nx.get_edge_attributes(G,'weight').values()

# %%
nx.draw_random(G,
        # edge_color=colors,
        node_size=node_sizes, 
        width=list(weights),
        with_labels=True)

    # %%

# %%
