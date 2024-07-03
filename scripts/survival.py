# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

import pandas as pd
import matplotlib as mpl


# Export results
with open('../exports/Uddin/Uddin_processed_2024_06_26.pk', 'rb') as f:
    processed_uddin = pk.load(f)

rename_col_dict = dict({'Gene.refGene':'PreferredSymbol',
                        'transcriptOI': 'HGVSc',
                        'pos': 'position',
                        'chrom': 'chromosome',
                        'ref': 'reference',
                        'alt': 'mutation',
                        'Gene_protein': 'p_key'})

for part in processed_uddin:
    part.obs = part.obs.rename(columns=rename_col_dict)

# %%



processed_uddin[0]
surv_df = pd.read_csv('../data/Uddin/outc_aging_ctos_inv.dat', sep='\t')
death_df = pd.read_csv('../data/Uddin/outc_death_all_discovered_inv.dat', sep='\t')
surv_df.iloc[0][[7,8]]
import anndata as ad
adata = ad.AnnData(surv_df, obs=death_df)

part = processed_uddin[0]



# %%
