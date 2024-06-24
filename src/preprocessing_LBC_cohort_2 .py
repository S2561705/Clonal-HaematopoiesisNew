import anndata as ad
import pandas as pd
import numpy as np
import pickle as pk
import plotly.express as px


data_ns = pd.read_csv('../data/LBC/mutation_data/LBC_ARCHER.2024_Cohort.27Mar24.1PCT_VAF_NON-SYNONYMOUS.tsv', sep='\t')
data_s = pd.read_csv('../data/LBC/mutation_data/LBC_ARCHER.2024_Cohort.27Mar24.1PCT_VAF_SYNONYMOUS.tsv', sep='\t')

meta = pd.read_csv('../data/LBC/lbc_meta.csv')

# LBC36 metadata subset
meta = meta[meta.cohort == 'LBC36'].copy()

obs_columns = ['PreferredSymbol', 'HGVSc', 'key', 'chromosome', 'position', 'reference', 'mutation', 'consequence', 'Variant_Classification', 'AF_Outlier_Pvalue', '95MDAF', 'is_error', 'is_germline', 'TYPE', 'p_key']
uns_info = ['Sex']

for i, data in enumerate([data_ns, data_s]):
    
    participant_list = []

    if i == 0:
        mut_type = 'non_synonymous'
    else:
        mut_type = 'synonymous'
    
    unique_ids = data['participant_id'].unique()

    # Cannot retrieve direct age information from meta as it's incomplete
    # Compute age approximation by using mean of participants in WAVE
    wave_to_age_dict = meta.groupby('WAVE').mean(numeric_only=True).age.to_dict()
    # no information about wave 5
    wave_to_age_dict[5] = wave_to_age_dict[4]+3

    # Create new column with age information
    data['age'] = data['wave'].map(wave_to_age_dict)

    # create sex column
    meta[['ID', 'sex']].groupby('ID').agg(['unique'])
    
    # Slice data assocaited with participants
    for id in unique_ids:

        slice_data = data[data['participant_id'] == id].copy()
        unique_time_points = slice_data.age.unique()

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

        gender = meta.loc[meta.ID == id].sex.unique()[0]

        new_ad = ad.AnnData(np.reshape(slice_data['AO']/slice_data['DP'],
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
        new_ad.uns['mutation_type'] = mut_type
        
        # participant_list.append(new_ad)
        # QC on the minimum number of reads 
        # in any time-point of a clonal trajectory
        keep_idx = np.argwhere(new_ad.layers['DP'].min(axis=1)>200)
        keep_idx = keep_idx.flatten()
        if keep_idx.shape[0] > 0:
            new_ad_qc = new_ad[keep_idx.flatten()]
            participant_list.append(new_ad_qc)

    if i == 0:
        with open('../exports/LBC/LBC_non_syn_cohort_2.pk', 'wb') as f:
            pk.dump(participant_list, f)

    if i == 1:
        with open('../exports/LBC/LBC_syn_cohort_2.pk', 'wb') as f:
            pk.dump(participant_list, f)