# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

import matplotlib.gridspec as gridspec
import pickle as pk

from scipy import stats

# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

summary = pd.read_csv('../results/mutation_df.csv')
summary['clipped_init_age'] = np.where(summary.init_age<0, 0, summary.init_age)

whi_variants = pd.read_csv('../data/WHI/uddin_whi.variants.with_effects_vep.tsv', sep='\t')


# Rename columns for consistency
whi_variants = whi_variants.rename(columns={
                        'commonid':'participant_id',
                        'DrawAge': 'age',
                        'Gene.refGene':'PreferredSymbol',
                        'transcriptOI': 'HGVSc',
                        'pos': 'position',
                        'chrom': 'chromosome',
                        'ref': 'reference',
                        'alt': 'mutation',
                        'vaf': 'AF',
                        'count_alt': 'AO',
                        'depth': 'DP'})

# Create unique keys for each mutation
key_list = []
for i, row in whi_variants.iterrows():
    key = row['Gene'] + ' ' + str(row.Protein)
    key_list.append(key)
whi_variants['key'] = key_list

whi_variants_dict = dict(zip(whi_variants.key, whi_variants.Effect))


for part in cohort:
    if part.uns['cohort'] == 'WHI':
        part.obs['Variant_Classification'] = part.obs.index.map(whi_variants_dict)

# %%
from sklearn.cluster import DBSCAN

data = summary[(summary.PreferredSymbol=='TET2') & (summary.fitness>0.05) & (summary.fitness<0.6)].copy()
db_res = DBSCAN(eps=1000, min_samples=5).fit(np.array(data.position)[:, None])
data['cluster_labels'] = db_res.labels_.astype(str)
data = data[data.cluster_labels!= '-1'].copy()

# sns.scatterplot(x=data.position, y=data.fitness, hue=data.cohort)
sns.scatterplot(x=data.position, y=data.fitness, hue=data.cluster_labels)


# sns.kdeplot(data, x='fitness_z_score', hue='cluster_labels')

# # %%
# from itertools import combinations
# from statannotations.Annotator import Annotator

# x = "cluster_labels"
# y = "fitness_z_score"
# order = [f'{i}' for i in range(0,7)]

# ax = sns.swarmplot(data=data, x=x, y=y, order=order)

# pairs = list(combinations(order, 2))

# annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
# annotator.configure(test='Mann-Whitney', text_format='simple', loc='outside')
# annotator.apply_and_annotate()
# %%
tet2_clinvar = pd.read_csv('../data/TET2_clinvar.txt', sep='\t')

classification = []
locations = []

for i in tet2_clinvar.GRCh37Location:
    if str(i) == i:
        split = np.array(i.split(' - '), dtype='float')
        if split.shape[0]==1:
            locations.append(split[0])
        else:
            locations.append(np.nan)
        # locations.append(
        #     np.nanmean(np.array(i.split(' - '), dtype='float')))
    else:
        locations.append(i)
tet2_clinvar['new_locations']=locations


tet2_clinvar = tet2_clinvar[tet2_clinvar.new_locations>1.06e8].copy()

label_dict={'Benign':-3,
            'Benign/Likely benign': -2,
            'Likely benign': -1,
            'Uncertain significance': 0,
            'Likely pathogenic': 1,
            'Conflicting classifications of pathogenicity':2,
            'Pathogenic':3,
            'other': np.nan,
            'not provided': np.nan}

label_dict['Conflicting classifications\nof pathogenicity'] = label_dict.pop('Conflicting classifications of pathogenicity')

tet2_clinvar['numerical_classification'] = tet2_clinvar['Germline classification'].map(label_dict)
tet2_clinvar = tet2_clinvar[tet2_clinvar.numerical_classification.notna()]
# %%

# Create a figure with two vertically stacked subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# First scatter plot on top subplot
sns.scatterplot(data=data, x='position', y='fitness', hue='cohort', ax=ax1)

# Second scatter plot on bottom subplot
sns.scatterplot(tet2_clinvar, x='new_locations', y='numerical_classification', 
                label='clinvar data', color='tab:grey', ax=ax2)

ax1.set_ylabel('Fitness')
ax2.set_xlabel('Chr position')
ax2.set_ylabel('')

# First, let's create an inverse dictionary, excluding NaN values
inverse_dict = {v: k for k, v in label_dict.items() if not pd.isna(v)}

# Now we can modify the yticks for ax2
# Assuming your current numeric yticks are -3 through 3
yticks = list(range(-3, 4))
yticklabels = [inverse_dict.get(val, '') for val in yticks]

# Apply to the axis
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticklabels)
# Adjust the layout to prevent overlap
plt.tight_layout()
sns.despine()
# Show the plot
plt.show()

# %%

# Export mutation-level dataframe
overlaping_columns = [set(part.obs.columns) for part in cohort]
overlaping_columns = list(set.intersection(*overlaping_columns))

summary_filtered = pd.concat([part.obs[overlaping_columns] for part in cohort])
summary_filtered.Variant_Classification.unique()

variant_classification = {
    'Frameshift': 'Frameshift',
    'Frame_Shift_Del': 'Frameshift',
    'Frame_Shift_Ins': 'Frameshift',
    'Splice site': 'Splice_Site',
    'Splice_Site': 'Splice_Site', 
    'Splice_Region': 'Splice_Site',
    'Nonsense': 'Nonsense',
    'Nonsense_Mutation': 'Nonsense',
    'Missense': 'Missense',
    'Missense_Mutation': 'Missense',
    'Synonymous': 'Synonymous',
    'In_Frame_Ins': 'In_Frame',
    'Inframe': 'In_Frame',
    "3'Flank": 'Flank'
}

pathogenic_classification = {
   'Frameshift': 'Likely Pathogenic',
   'Frame_Shift_Del': 'Likely Pathogenic',
   'Frame_Shift_Ins': 'Likely Pathogenic',
   'Splice site': 'Likely Pathogenic',
   'Splice_Site': 'Likely Pathogenic', 
   'Splice_Region': 'Likely Pathogenic',
   'Nonsense': 'Pathogenic',
   'Nonsense_Mutation': 'Pathogenic',
   'Missense': 'Benign',
   'Missense_Mutation': 'Benign',
   'Synonymous': 'Benign',
   'In_Frame_Ins': 'Benign',
   'Inframe': 'Benign',
   "3'Flank": 'Benign'
}
summary_filtered['pathogenic_classification'] = summary_filtered.Variant_Classification.map(pathogenic_classification)
summary_filtered['variant_classification'] = summary_filtered.Variant_Classification.map(variant_classification)

data = summary_filtered.copy()
order = data.groupby('variant_classification').mean(numeric_only=True).sort_values(by='fitness').index

from itertools import combinations
from statannotations.Annotator import Annotator

variant_comb = list(combinations(order, 2))

x='variant_classification'
y='fitness'
ax = sns.boxplot(data=data, x=x, y=y, order=order)
annot = Annotator(ax, variant_comb, data=data, x=x, y=y, order=order)
annot.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
annot.apply_test()
ax, test_results = annot.annotate()
plt.xticks(rotation=90)

# %%

# Create a figure with two vertically stacked subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# First scatter plot on top subplot
sns.scatterplot(data=data, x='position', y='fitness', hue='cohort', ax=ax1)

# Second scatter plot on bottom subplot
sns.scatterplot(tet2_clinvar, x='new_locations', y='numerical_classification', 
                label='clinvar data', color='tab:grey', ax=ax2)

ax1.set_ylabel('Fitness')
ax2.set_xlabel('Chr position')
ax2.set_ylabel('')

# First, let's create an inverse dictionary, excluding NaN values
inverse_dict = {v: k for k, v in label_dict.items() if not pd.isna(v)}

# Now we can modify the yticks for ax2
# Assuming your current numeric yticks are -3 through 3
yticks = list(range(-3, 4))
yticklabels = [inverse_dict.get(val, '') for val in yticks]

# Apply to the axis
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticklabels)
# Adjust the layout to prevent overlap
plt.tight_layout()
sns.despine()
# Show the plot
plt.show()

# %%
