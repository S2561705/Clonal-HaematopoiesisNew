# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
# %%

# Export results
with open('../exports/Uddin/Uddin_processed_2024_05_17.pk', 'rb') as f:
    processed_uddin = pk.load(f)

# Export results
with open('../exports/LBC/merged_cohort_fitted.pk', 'rb') as f:
    processed_lbc = pk.load(f)

processed_uddin[0]
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

# %%

summary = pd.concat([part.obs for part in cohort])
summary = summary[summary['fitness'].notna()]
summary = summary.sort_values(by='fitness', ascending=False)


# plot fitness distribution
sns.displot(x=summary.fitness, kde=True)
plt.title('Fitness distribution')
plt.show()
plt.clf()

px.box(summary, x='PreferredSymbol', y='fitness', points='all', hover_data=['Sample ID', 'p_key'])
px.box(summary[summary.fitness>0.01], x='PreferredSymbol', y='fitness', points='all', hover_data=['Sample ID', 'p_key'])

# %%
summary_filtered = summary[summary.fitness>0.01]

gene_mean_fitness_dict =(
    summary_filtered.groupby('PreferredSymbol')['fitness'].mean().to_dict())

gene_mean_fitness_dict = dict(sorted(gene_mean_fitness_dict.items(), key=lambda item: item[1], reverse=True))

fig = go.Figure()
for gene in gene_mean_fitness_dict.keys():
    summary_gene = summary_filtered[summary_filtered.PreferredSymbol==gene]
    fig.add_trace(
        go.Box(
            x=summary_gene.PreferredSymbol, y=summary_gene.fitness,
            name=gene, boxpoints='all', hovertext=summary_gene.index
        )
    )
fig.update_xaxes(tickangle=-70)
fig.show(width=2000)
fig.write_image('../results/combined_fitness_map.png', width=1000)

# %%
# Plot MYC part
# for part in cohort:
#     if part.uns['participant_id'] == 'LBC360020':
#         plot_part(part)
#         plot_optimal_model(part)

for part in cohort:
    if 'PPM1D' in list(part.obs.PreferredSymbol):
        plot_part(part)
        # print(part.layers['DP'])
        # print(part.layers['AO'])
        print(part.uns['participant_id'])
        # print(part.uns)

# %%
#only genes with 3 mutations
fig = go.Figure()
for gene in gene_mean_fitness_dict.keys():
    summary_gene = summary_filtered[summary_filtered.PreferredSymbol==gene]
    if len(summary_gene)<2:
        continue
    fig.add_trace(
        go.Box(
            x=summary_gene.PreferredSymbol, y=summary_gene.fitness,
            name=gene, boxpoints='all', hovertext=summary_gene.index
        )
    )
fig.update_xaxes(tickangle=-70)
fig.show(width=2000)
fig.write_image('../results/combined_fitness_map_morethan2.png', width=1000)
# %%

summary_filtered.to_csv('../results/summary_filtered.csv', sep=';')
# %%