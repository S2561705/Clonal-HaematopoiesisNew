import pandas as pd
from pyvis.network import Network
import matplotlib as mpl
import ast
from collections import Counter
import numpy as np

import os
def create_gene_part_network(gene, cohort, directory):
    gene_cohort = [part for part in cohort 
                if gene in list(part.obs.PreferredSymbol)]

    summary = pd.concat([part.obs for part in gene_cohort])

    summary = summary[summary['fitness'].notna()]
    summary['log_fitness'] = np.log(summary.fitness)

    # add nodes
    # each node corresponds to a gene
    net = Network(height="1000px", width="1000px")
    cmap =  mpl.colormaps['plasma']

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
    for part in gene_cohort:
        if part.shape[0] > 1:
            obs = part.obs
            # Loop through rows to find gene rows
            for i in obs.index:
                if obs.loc[i].PreferredSymbol == gene:
                    rest_genes = list(obs.drop(i).PreferredSymbol)
                    for g in rest_genes:
                        if g in edge_dict.items():
                            edge_dict[g].append(obs.loc[i].log_fitness)
                        else:
                            edge_dict[g] = [obs.loc[i].log_fitness]

    # compute average log_fitness
    for key, v in edge_dict.items():
        edge_dict[key] = (np.mean(v), len(v))

    for key, v in edge_dict.items():
        print(key, v)
        edge_size = 2 + np.log2(1+np.array(v[1]))
        edge_size_log = np.log(1 + np.array(v[1]))
        edge_size_sqr_transform = 10 + np.sqrt(v[1])
        
        edge_log_fitness_scaled = (v[0]-min_log_fitness)/(max_log_fitness-min_log_fitness)
        edge_color_rgba = cmap(edge_log_fitness_scaled)
        edge_color_hex = mpl.colors.rgb2hex(edge_color_rgba)

        net.add_edge(gene, key, width=edge_size, color=edge_color_hex) 

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

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    net.save_graph(name=f'{directory + gene}_network.html')
