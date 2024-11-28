# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.aux import *
from src.survival_predictor_aux import *
from statannotations.Annotator import Annotator


import pickle as pk
# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

# Create Network plot
for i, part in enumerate(cohort):
    part.obs['clonal_structure_str'] = part.obs.clonal_structure.astype(str)



filtered_gene = 'ASXL1'
min_mut = 0

summary = pd.concat([part.obs for part in cohort])

data = summary[summary.PreferredSymbol == filtered_gene].copy()

df = pd.DataFrame(columns=['gene', 'fitness'])
pandas_tuples = []
for i in range(len(data)):
    row = data.iloc[i]
    if len(row.clonal_structure) == 1:
        pandas_tuples.append(['single', row.fitness])
        # df = pd.concat([df, 
        #            pd.DataFrame({'gene':'single',
        #                          'fitness':row.fitness})],
        #                          ignore_index=True)

        # if 'single' not in fitness_dict:
        #     fitness_dict['single'] = [row.fitness]
        # else:
        #     fitness_dict['single'].append(row.fitness)
    else:
        row.clonal_structure.remove(data.index[i])
        other_mut = [m.split(' ')[0] for m in row.clonal_structure]
        for g in other_mut:
            pandas_tuples.append([g, row.fitness])

df = pd.DataFrame(columns=['gene', 'fitness'], data=pandas_tuples)

result = df[df.gene.isin(df.gene.value_counts()[df.gene.value_counts() > min_mut].index)]
order = result.groupby(by='gene').mean().sort_values(
    by='fitness', ascending=True).index
ax = sns.boxplot(x=result.gene,
                 y=result.fitness,
                 order=order)


from itertools import combinations
pairs = list(combinations(order,2))
keep_pairs = []
for p in pairs:
    s, p_value = stats.brunnermunzel(result[result.gene==p[0]].fitness,
                        result[result.gene==p[1]].fitness)
    if p_value < 0.05:
        keep_pairs.append(p)

annotator = Annotator(ax, keep_pairs, data=df, x='gene', y='fitness', order=order)
annotator.configure(test='Brunner-Munzel', text_format='simple', loc='inside')
annotator.apply_and_annotate()
sns.despine()
plt.title(f'{filtered_gene} context specific\nfitness differences')
plt.xlabel('')
plt.ylabel('Fitness')
plt.xticks(rotation=45)
plt.savefig(f'../plots/Supp Figure 3/network_differences_{filtered_gene}.png', 
            bbox_inches='tight', transparent=True, dpi=200 )

df.groupby(by='gene').mean()
# %%


summary = pd.read_csv('../results/mutation_df.csv')

# summary = summary[(summary.fitness>0.05)].copy()
# data = summary[summary.PreferredSymbol == filtered_gene].copy()
# sns.catplot(x=data.clonal_structure_size, y=data.fitness, kind='violin', inner=None)

# %%
from itertools import combinations
order = [1,2,3,4]
pairs = list(combinations(order,2))
ax= sns.swarmplot(x=summary.clonal_structure_size, y=summary.fitness)

annotator = Annotator(ax, pairs, x=summary.clonal_structure_size, y=summary.fitness, order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()
sns.despine()
plt.xlabel('')
plt.ylabel('Fitness')

plt.ylim(0, 1.2)
plt.savefig('../plots/Supp Figure 3/clone_size_vs_fitness.png', transparent=True,dpi=200)
plt.savefig('../plots/Supp Figure 3/clone_size_vs_fitness.svg')


# %%

summary[summary.clonal_structure_size ==1].fitness.mean()
summary[summary.clonal_structure_size ==3].fitness.mean()