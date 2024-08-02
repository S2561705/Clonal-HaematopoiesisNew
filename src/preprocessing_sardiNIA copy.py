import anndata as ad
import pandas as pd
import numpy as np
import pickle as pk
import json

data = pd.read_csv('../data/sardiNIA/CHvariantCalls_withReadCounts.csv')

sardiNIA_fabre_id_match = pd.read_csv('../data/sardiNIA/sardiNIA_id_match.csv')

#create ID dictionary
ID_PD_map = dict(zip(sardiNIA_fabre_id_match.PD_ID, sardiNIA_fabre_id_match.Sard_ID))

# rename participant_ids
data = data.rename(columns={'Sample ID':'participant_id_PD'})

data['participant_id'] = data['participant_id_PD'].map(ID_PD_map)

key_list = []
for i, row in data.iterrows():
    key_list.append(
        row['Gene'] + ' ' + 'c.' + str(row['Start']) + row['WT'] + '>' + row['MT'])

data['key'] = key_list
data['p_key'] = data['Gene'] + ' ' + data['Protein']

data = data.rename(columns={'Age': 'age',
                             'Chromosome': 'chromosome',
                             'Start':'position',
                             'Gene':'PreferredSymbol',
                             'WT':'reference',
                             'MT':'mutation',
                             'Effect':'Variant_Classification',
                             'Total Count':'DP',
                             'MT count': 'AO',
                             'VAF':'AF'})



obs_columns = ['PreferredSymbol', 'chromosome', 'position', 'reference', 'mutation', 'Variant_Classification', 'key', 'p_key']
uns_info = ['Sex']

# add information about cohort, required for mean age
data['cohort'] = 'sardiNIA'
mut_type = 'non_synonymous'

participant_list = []

unique_ids = data['participant_id'].unique()

# # create sex column
# meta[['ID', 'sex']].groupby('ID').agg(['unique'])

# Slice data assocaited with participants
for id in unique_ids:

    slice_data = data[data['participant_id'] == id].copy()

    unique_time_points = slice_data.age.unique()

    if unique_time_points.shape[0] == 1:
        continue

    # find mutations with observations at all time points
    valid_keys = [key for key in slice_data.key.unique() 
                if len(slice_data[slice_data.key == key])==len(unique_time_points)]

    slice_data = slice_data[slice_data.key.isin(valid_keys)].copy()

    # Sort data by mutation key and age
    slice_data = slice_data.sort_values(by=['key', 'age'])
    unique_time_points = slice_data.age.unique()

    delta_t = np.diff(unique_time_points, prepend=unique_time_points[0])
    var_df = pd.DataFrame(data=np.array([unique_time_points, delta_t]).T, columns=['time_points', 'delta_t'])
    obs_df = slice_data[slice_data.age == unique_time_points[0]][obs_columns]
    obs_df = obs_df.set_index('key')

    new_ad = ad.AnnData(np.reshape(np.array(slice_data['AF'])[:, None],#slice_data['AO']/slice_data['DP'],
                                        (len(obs_df),
                                        len(unique_time_points))),
                        obs=obs_df,
                        var=var_df
                        )
    
    new_ad.layers['AO'] = np.reshape(slice_data['AO'],
                                        (len(obs_df),
                                        len(unique_time_points))
    )

    new_ad.layers['DP'] = np.reshape(slice_data['DP'],
                                        (len(obs_df),
                                        len(unique_time_points))
    )
    
    new_ad.uns['participant_id'] = id
    new_ad.uns['mutation_type'] = mut_type

    # IDs were scrambled and can't recover sex
    # gender = meta.loc[meta.ID == id].sex.unique()[0]
    # new_ad.uns['gender'] = gender[0]

    # participant_list.append(new_ad)
    # QC on the minimum number of reads 
    # in any time-point of a clonal trajectory
    keep_idx = np.argwhere(new_ad.layers['DP'].min(axis=1)>200)
    keep_idx = keep_idx.flatten()
    if keep_idx.shape[0] > 0:
        new_ad_qc = new_ad[keep_idx.flatten()]
        participant_list.append(new_ad_qc)


with open('../exports/sardiNIA/sardiNIA.pk', 'wb') as f:
    pk.dump(participant_list, f)


filtered_1_list = []
for part in participant_list:
    keep_idx = np.argwhere(part.X.max(axis=1)>0.01).flatten()
    if keep_idx.shape[0]>0:
        filtered_1_list.append(part[keep_idx])

with open('../exports/sardiNIA/sardiNIA_1percent.pk', 'wb') as f:
    pk.dump(filtered_1_list, f)
