import anndata as ad
import pandas as pd
import numpy as np
import pickle as pk

data = pd.read_csv('../data/Uddin/Uddin_data.csv', sep='\t')
# keep only 
data = data[data['Variant Type'] == 'exonic']
data = data.dropna(axis=1)
# drop empty columns
data.to_csv('../data/Uddin/Uddin_data_exonic.csv')

data = pd.read_csv('../data/Uddin/Uddin_data_exonic.csv')

# Manipulations
data['VAF'] = data.ALT_Count/ data['Total Depth']
data = data.rename(columns={'Age (Blood Draw)': 'age'})

row = data.iloc[10]
key_list = []
for i, row in data.iterrows():
    key = row['Gene'] + ' '
    for ann in row['Annotation'].split(':'):
        if 'c.' in ann:
            ann_split = ann.split('.')
            if ann_split[1][0].isdigit():
                key += ann
            else:
                key += (
                    'c.' + ann_split[1][1:-1]
                    +  ann_split[1][0] + '>'
                    + ann_split[1][-1])
    key_list.append(key)
data['key'] = key_list

data['Gene_protein'] = data['Gene'] +' p.' + data['Protein Change']
obs_columns = ['key', 'Gene', 'Annotation', 'CHR', 'Position (hg19)', 'REF', 'ALT', 'Protein Change', 'Gene_protein']
unique_ids = data['Sample ID'].unique()

participant_list = []
nan_counter = 0
for id in unique_ids:
    slice_data = data[data['Sample ID'] == id].copy()
    unique_time_points = slice_data.age.unique()

    # find mutations with observations at all time points
    valid_keys = [key for key in slice_data.key.unique() 
                if len(slice_data[slice_data.key == key])==len(unique_time_points)]

    slice_data = slice_data[slice_data.key.isin(valid_keys)].copy()

    slice_data = slice_data.sort_values(by=['key', 'age'])
    unique_time_points = slice_data['age'].unique()
    unique_time_points.sort()

    if (len(slice_data)/unique_time_points.shape[0]).is_integer():
        delta_t = np.diff(unique_time_points, prepend=unique_time_points[0])
        var_df = pd.DataFrame(data=np.array([unique_time_points, delta_t]).T, columns=['time_points', 'delta_t'])
        obs_df = slice_data[slice_data['age'] == unique_time_points[0]][obs_columns]
        obs_df = obs_df.set_index('key')

        new_ad = ad.AnnData(np.reshape(slice_data['VAF'],
                                            (len(obs_df),
                                            len(unique_time_points))).astype(float),
                            obs=obs_df,
                            var=var_df
                            )
        new_ad.layers['AO'] = np.reshape(slice_data['ALT_Count'],
                                            (len(obs_df),
                                            len(unique_time_points))
        )

        new_ad.layers['DP'] = np.reshape(slice_data['Total Depth'],
                                            (len(obs_df),
                                            len(unique_time_points))
        )
        new_ad.uns['participant_id'] = id

        # deal with nan index
        new_index = []
        for idx in list(new_ad.obs.index):
            if idx != 'nan':
                new_index.append(idx)
            else:
                new_index.append(f'nan_{nan_counter}')
                nan_counter += 1

        new_ad.obs.index = new_index

        if new_ad.shape[1] > 1:
            participant_list.append(new_ad)

with open('../exports/Uddin.pk', 'wb') as f:
    pk.dump(participant_list, f)