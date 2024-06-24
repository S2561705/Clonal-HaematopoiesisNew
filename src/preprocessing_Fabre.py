import anndata as ad
import pandas as pd
import numpy as np
import pickle as pk

data = pd.read_csv('../data/Fabre.Restructured.AllVariants.tsv', sep='\t')

obs_columns = ['PreferredSymbol', 'HGVSc', 'chromosome', 'position', 'reference', 'mutation', 'consequence', 'Variant_Classification']
uns_info = ['Sex']
unique_ids = data['participant_id'].unique()

participant_list = []

for id in unique_ids:
    slice_data = data[data['participant_id'] == id].copy()
    slice_data = slice_data.sort_values(by=['PreferredSymbol', 'age'])
    unique_time_points = slice_data.age.unique()

    if (len(slice_data)/unique_time_points.shape[0]).is_integer():
        delta_t = np.diff(unique_time_points, prepend=unique_time_points[0])
        var_df = pd.DataFrame(data=np.array([unique_time_points, delta_t]).T, columns=['time_points', 'delta_t'])
        obs_df = slice_data[slice_data.age == unique_time_points[0]][obs_columns]
        obs_df = obs_df.set_index('HGVSc')

        gender = slice_data.gender.unique()

        new_ad = ad.AnnData(np.reshape(slice_data['AF'],
                                            (len(obs_df),
                                            len(unique_time_points))),
                            obs=obs_df,
                            var=var_df
                            )
        new_ad.layers['AO'] = np.reshape(slice_data['UAO'],
                                            (len(obs_df),
                                            len(unique_time_points))
        )

        new_ad.layers['DP'] = np.reshape(slice_data['DP'],
                                            (len(obs_df),
                                            len(unique_time_points))
        )
        new_ad.uns['participant_id'] = id
        new_ad.uns['gender'] = gender[0]
        
        participant_list.append(new_ad)


with open('../exports/Fabre.pk', 'wb') as f:
    pk.dump(participant_list, f)