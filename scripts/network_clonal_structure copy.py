# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.blood_markers_aux import *
from src.aux import *

import pandas as pd
from pyvis.network import Network
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import Counter
from itertools import combinations
import networkx as nx

min_edge = 1
in_edge = True
contains_gene = 'DNMT3A'

# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

# Create Network plot
for i, part in enumerate(cohort):
    part.obs['clonal_structure_str'] = part.obs.clonal_structure.astype(str)

summary = pd.concat([part.obs for part in cohort])
summary = summary[summary['fitness'].notna()].copy()
summary['log_fitness'] = np.log(summary.fitness)
normalise_parameter(summary, 'fitness')
normalise_parameter(summary, 'log_fitness')

summary['clonal_structure_tuple'] = summary['clonal_structure'].apply(tuple)

summary[summary.PreferredSymbol=='DNMT3A']
cs_df = summary.groupby(['participant_id', 'clonal_structure_tuple']).mean(numeric_only=True)

if in_edge is True:
    cs_df = cs_df[cs_df.index.map(lambda x: len(x[1]) > 1)].copy()
    keep_genes = set([PS.split()[0] for i in cs_df.index for PS in i[1]])
    summary = summary[summary.PreferredSymbol.isin(list(keep_genes))].copy()

# Create colormap
cmap = mpl.colormaps['plasma']

# Initialize NetworkX graph
G = nx.Graph()

# Calculate min and max for normalization
min_log_fitness = np.amin(summary.log_fitness_z_score)
max_log_fitness = np.amax(summary.log_fitness_z_score)

# Add nodes to NetworkX graph
gene_dict = summary.PreferredSymbol.value_counts().to_dict()
nodes = list(gene_dict.keys())
node_sizes = np.array(list(gene_dict.values()), dtype='float')
node_sizes_log = 5 + np.log2(np.array(node_sizes))

# Calculate normalized node colors
node_log_fitness = np.array([summary[summary.PreferredSymbol == gene].log_fitness_z_score.mean() for gene in nodes])
node_log_fitness_scaled = (node_log_fitness - min_log_fitness) / (max_log_fitness - min_log_fitness)
node_color_rgba = cmap(node_log_fitness_scaled)
node_color_hex = [mpl.colors.rgb2hex(color) for color in node_color_rgba]

if contains_gene is not None:
    cs_list = list(summary.clonal_structure_tuple.values)
    keep_idx = []
    for i in range(len(cs_list)):
        gene_counts = 0
        for mut in cs_list[i]:
            if contains_gene in mut:
                keep_idx.append(i)
                continue

    summary = summary.iloc[keep_idx].copy()
    cs_df = summary.groupby(['participant_id', 'clonal_structure_tuple']).mean(numeric_only=True)


keep_genes = summary.PreferredSymbol.unique()

# Add nodes with sizes and colors
for node, size, color in zip(nodes, node_sizes_log, node_color_hex):
    if node in keep_genes:
        G.add_node(node, size=size, color=color, font=100)

# Create edges
edge_dict = dict()
for i in range(len(cs_df)):
    cs = cs_df.index[i][1]
    if len(cs) > 1:
        cs_log_fitness = cs_df.iloc[i].log_fitness_z_score
        for comb in combinations([mut.split(' ')[0] for mut in cs], 2):
            sorted_comb = tuple(sorted(comb))
            if sorted_comb in edge_dict.keys():
                edge_dict[sorted_comb].append(cs_log_fitness)
            else:
                edge_dict[sorted_comb] = [cs_log_fitness]

if contains_gene is not None:
    edge_dict = {k:v for k,v in edge_dict.items() if contains_gene in k}

for key, v in edge_dict.items():
    edge_dict[key] = (np.mean(v), len(v))

# Add edges to NetworkX graph if both nodes exist
for key, v in edge_dict.items():
    if v[1] >= min_edge:
        edge_size = 1 + np.log(1 + np.array(v[1]))
        # Normalize edge color using the same min/max as nodes
        edge_log_fitness_scaled = (v[0] - min_log_fitness) / (max_log_fitness - min_log_fitness)
        edge_color_rgba = cmap(edge_log_fitness_scaled)
        edge_color_hex = mpl.colors.rgb2hex(edge_color_rgba)
        if key[0] in G.nodes and key[1] in G.nodes:
            G.add_edge(key[0], key[1], weight=edge_size, color=edge_color_hex)



# Initialize pyvis Network
net = Network(height="1000px", width="1000px")

# net = Network()
font_size = 20
# Add nodes to pyvis Network
for node in G.nodes(data=True):
    if in_edge is True:
        if node[0] in keep_genes:
            net.add_node(node[0], size=node[1]['size'], color=node[1]['color'],
                        font={'size': font_size})
    else:
        net.add_node(node[0], size=node[1]['size'], color=node[1]['color'],
                    font={'size': font_size})

# Add edges to pyvis Network
for edge in G.edges(data=True):
    net.add_edge(edge[0], edge[1], width=edge[2]['weight'], color=edge[2]['color'])

# Options
net.set_options("""
    const options = {
      "physics": {
        "forceAtlas2Based": {
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
""")

# Save graph
if contains_gene is None:
    net.save_graph(name='../results/clonal_structure_graph.html')
else:
    net.save_graph(name=f'../results/clonal_structure_{contains_gene}_graph.html')

# %%
# Plot legends
fig, ax = plt.subplots()

# Colorbar for node colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_log_fitness, vmax=max_log_fitness))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Normalised log Fitness')

# Width legend for edges
thickness_levels = [1.7, 4.95]  # Example thickness levels
transformed_levels = [1, 50]
edge_legend_elements = [plt.Line2D([0], [0], color='black', lw=thickness, label=f'{transformed} occurences')
                   for thickness, transformed in zip(thickness_levels, transformed_levels)]


# Node size legend
node_sizes = {
    '1 occurences': 100,     # Example values
    '20 occurences': 100,
    '200 occurrences': 100
}
node_size_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markersize=np.sqrt(size) / 10,
                                        markerfacecolor='skyblue', label=label)
                             for label, size in node_sizes.items()]

# Combine legends
edge_legend = ax.legend(handles=edge_legend_elements, title='Edge Thickness ~ Log Occurrences', loc='upper right')
node_size_legend = ax.legend(handles=node_size_legend_elements, title='Node Sizes ~ Log Occurences', loc='lower right')

# Add legends to plot
ax.add_artist(edge_legend)  # Add the edge legend to the plot
ax.add_artist(node_size_legend)  # Add the node size legend to the plot

# ax.legend(handles=legend_elements, title='Edge Thickness ~ Log Occurences', loc='upper right')
# ax.set_title('Network Graph Legends')
if contains_gene is not None:
    plt.savefig(f'../results/clonal_structure_graph_legend_{contains_gene}.svg')

else:
    plt.savefig('../results/clonal_structure_graph_legend.svg')

# %%
