# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
from pyvis.network import Network
import matplotlib as mpl
import ast
from collections import Counter

# Export results
with open('../exports/Uddin/Uddin_processed_2024_05_17.pk', 'rb') as f:
    processed_uddin = pk.load(f)

# Export results
with open('../exports/LBC/merged_cohort_fitted.pk', 'rb') as f:
    processed_lbc = pk.load(f)

rename_col_dict = dict({'Gene':'PreferredSymbol',
                        'Annotation': 'HGVSc',
                        'Position (hg19)': 'position',
                        'CHR': 'chromosome',
                        'REF': 'reference',
                        'ALT': 'mutation',
                        'Protein Change': 'p_key',
                        'fitness':'fitness'})

for part in processed_uddin:
    part.obs = part.obs.rename(columns=rename_col_dict)

cohort = processed_uddin + processed_lbc
for part in cohort:
    part.obs['Sample ID'] = part.uns['participant_id']

# QC filter
# Drop participatns with fit warning
cohort = [part for part in cohort if part.uns['warning']==None]
# Drop MYC participant:
cohort = [part for part in cohort if 'MYC' not in list(part.obs.PreferredSymbol)]

# %%

# extract clonal structures
for i, part in enumerate(cohort):
    part.obs['clonal_structure_str'] = part.obs.clonal_structure.astype(str)
    part.obs['log_fitness'] = np.log(part.obs.fitness)

cs_df = pd.concat([part.obs.groupby('clonal_structure_str').mean(numeric_only=True)
           for part in cohort])

summary = pd.concat([part.obs for part in cohort])
summary = summary[summary['fitness'].notna()]
summary['log_fitness'] = np.log(summary.fitness)
# summary = summary[summary.fitness>0.01]


# %%
# %%
# # keep only genes with >2 mutations
# gene_dict = summary.PreferredSymbol.value_counts().to_dict()
# selected_genes = [gene for gene, v in gene_dict.items() if v>=2]

# summary = summary[summary.PreferredSymbol.isin(selected_genes)].copy()
cmap =  mpl.colormaps['plasma']


# %%
# add nodes
# each node corresponds to a gene
net = Network(height="1000px", width="1000px")

gene_dict = summary.PreferredSymbol.value_counts().to_dict()

nodes = list(gene_dict.keys())

# node size corresponds to the amount of mutations occurring in the gene
node_sizes = np.array(list(gene_dict.values()), dtype='float')
node_sizes_log =  5 + np.log2(np.array(node_sizes))
node_sizes_sqr_transform = np.sqrt(node_sizes)

# node color corresponds to the acerage fitness of mutations in gene
node_log_fitness = np.array([summary[summary.PreferredSymbol == gene].log_fitness.mean()
    for gene in nodes])

# create log_fitness scaler
min_log_fitness = np.amin(summary.log_fitness)
max_log_fitness = np.amax(summary.log_fitness)

node_log_fitness_scaled = (node_log_fitness-min_log_fitness)/(max_log_fitness-min_log_fitness)
node_color_rgba = cmap(node_log_fitness_scaled)
node_color_hex = [mpl.colors.rgb2hex(color) for color in node_color_rgba]

net.add_nodes(nodes, size=node_sizes_log, color=node_color_hex)

# Create edges
# Extract all co-ocurring mutations in the same participant
# Create edges with associated list of log_fitnesses
edge_dict = dict()
for i in range(len(cs_df)):
    # create_nodes associated with clonal structure
    cs = ast.literal_eval(cs_df.index[i])
    if len(cs) > 1:
        cs_log_fitness = cs_df.iloc[i].log_fitness
        for comb in combinations([mut.split(' ')[0] for mut in cs],
                                    2):
            # order combination of genes alphabetically:
            sorted_comb = tuple(sorted(comb))
            
            if sorted_comb in edge_dict.keys():
                edge_dict[sorted_comb].append(cs_log_fitness)
            else:    
                edge_dict[sorted_comb]=[cs_log_fitness]

# compute average log_fitness
for key, v in edge_dict.items():
    edge_dict[key] = (np.mean(v), len(v))

for key, v in edge_dict.items():
    edge_size = 2 + np.log2(1+np.array(v[1]))
    edge_size_log = np.log(1 + np.array(v[1]))
    edge_size_sqr_transform = 10 + np.sqrt(v[1])
    
    edge_log_fitness_scaled = (v[0]-min_log_fitness)/(max_log_fitness-min_log_fitness)
    edge_color_rgba = cmap(edge_log_fitness_scaled)
    edge_color_hex = mpl.colors.rgb2hex(edge_color_rgba)

    print(edge_size_log)
    net.add_edge(key[0], key[1], width=edge_size, color=edge_color_hex) 

net.toggle_physics(True)
# net.show_buttons(filter_=['physics'])

# sns.displot(summary.log_fitness, kde=True)
# sns.displot([np.log(1+ v[1]) for v in edge_dict.values()], bins=100)

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
net.show('1.html')
# %%

import networkx as nx
G = nx.Graph()
for i in range(len(nodes)):
    G.add_node(nodes[i], color=node_color_hex[i], size=node_sizes_log[i])

edge_width_list = []
for key, v in edge_dict.items():
    edge_size = v[1]
    edge_size_log = 5 + np.log(1 + np.array(v[1]))
    edge_size_sqr_transform = 5 + np.sqrt(v[1])
    edge_width_list.append(edge_size_log)

    edge_log_fitness_scaled = (v[0]-min_log_fitness)/(max_log_fitness-min_log_fitness)
    edge_color_rgba = cmap(edge_log_fitness_scaled)
    edge_color_hex = mpl.colors.rgb2hex(edge_color_rgba)

    G.add_edge(key[0], key[1], width = edge_size,
                                  color=edge_color_hex) 


# %%
plt.figure(figsize=(10, 10), dpi=100)
pos = nx.spring_layout(G,k=2, seed=2)
nx.draw_networkx(G, pos, with_labels=True, node_size= node_sizes_log, node_color=node_log_fitness, cmap='plasma')
nx.draw_networkx_edges(G, pos = pos, width=edge_width_list)
# %%
