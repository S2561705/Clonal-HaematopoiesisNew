# %%
import sys
import os
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.network import *

# Export results
with open('../exports/Uddin/Uddin_processed_2024_06_26.pk', 'rb') as f:
    processed_uddin = pk.load(f)

# Export results
with open('../exports/LBC/merged_cohort_fitted.pk', 'rb') as f:
    processed_lbc = pk.load(f)

rename_col_dict = dict({'Gene.refGene':'PreferredSymbol',
                        'transcriptOI': 'HGVSc',
                        'pos': 'position',
                        'chrom': 'chromosome',
                        'ref': 'reference',
                        'alt': 'mutation',
                        'Gene_protein': 'p_key'})

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

# extract clonal structures
for i, part in enumerate(cohort):
    part.obs['clonal_structure_str'] = part.obs.clonal_structure.astype(str)
    part.obs['log_fitness'] = np.log(part.obs.fitness)

# %%

gene_list = []
for part in cohort:
    gene_list.extend(list(part.obs.PreferredSymbol))

gene_list = list(set(gene_list))
directory = '../results/network/gene/'

for gene in gene_list:
    create_gene_part_network(gene, cohort, directory)

# %%