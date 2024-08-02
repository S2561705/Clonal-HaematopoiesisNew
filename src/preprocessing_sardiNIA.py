import anndata as ad
import pandas as pd
import numpy as np
import pickle as pk
import json

data_ns = pd.read_csv('../data/sardiNIA/Fabre_CHIP.05Jul24.2PCT_VAF_NON-SYNONYMOUS.tsv', sep='\t')
data_s = pd.read_csv('../data/sardiNIA/Fabre_CHIP.05Jul24.1PCT_VAF_SYNONYMOUS.tsv', sep='\t')


sardinia_df = pd.read_csv('../data/sardiNIA/fabre_deaths upd 2024.csv')
sardiNIA_fabre_id_match = pd.read_csv('../data/sardiNIA/sardiNIA_id_match.csv')


#create ID dictionary
ID_PD_map = dict(zip(sardiNIA_fabre_id_match.PD_ID, sardiNIA_fabre_id_match.Sard_ID))

# rename participant_ids
data_ns = data_ns.rename(columns={'participant_id':'participant_id_PD'})

data_ns['participant_id'] = data_ns['participant_id_PD'].map(ID_PD_map)
n_phase_dict =dict(zip(sardinia_df.ID, sardinia_df.n_phase))

id = data_ns.participant_id.unique()[0]

points_discrepancy = []
for id in data_ns.participant_id.unique():
    n_points = np.array(data_ns[data_ns.participant_id==id]['HGVSc'].value_counts())

    # expected number of points
    exp_points = n_phase_dict[id]

    points_discrepancy.extend(list(exp_points - n_points))

# percentage of complete trajectories
proportion = np.sum([p== 0 for p in points_discrepancy])/len(points_discrepancy)
proportion*100

import seaborn as sns
sns.histplot(x=points_discrepancy, bins=10)



data_ns[data_ns['participant_id']==data_ns['participant_id'].unique()[0]]
data_ns.wave.unique()
data_ns.groupby('participant_id').min(numeric_only=True).
len(data_ns.participant_id.unique())
len(data_ns)

import seaborn as sns
import plotly.express as px
px.histogram(x=data_ns.HGVSp.value_counts(),nbins=350)

# check single participant
data_ns[data_ns.participant_id == data_ns.participant_id.unique()[0]]

data_ns.wave.unique()
import seaborn as sns
# check single mutation:
sns.histplot(x=[len(data_ns[data_ns.HGVSc == mut]) for mut in  data_ns.HGVSc.unique()])

sardiNIA_fabre_id_match = pd.read_csv('../data/sardiNIA/sardiNIA_id_match.csv')
age_meta_df = pd.read_csv('../data/sardiNIA/fabre_deaths upd 2024.csv')

#create ID dictionary
ID_PD_map = dict(zip(sardiNIA_fabre_id_match.PD_ID, sardiNIA_fabre_id_match.Sard_ID))
# rename participant_ids
data_ns = data_ns.rename(columns={'participant_id':'participant_id_PD'})
data_s = data_s.rename(columns={'participant_id':'participant_id_PD'})

data_ns['participant_id'] = data_ns['participant_id_PD'].map(ID_PD_map)
data_s['participant_id'] = data_ns['participant_id_PD'].map(ID_PD_map)

# replace waves with age
id_wave_age_dict = dict()
for i, row in age_meta_df.iterrows():
    id_wave_age_dict[row.ID] = np.array(row.iloc[1:6])

data_ns.iloc[0].participant_id
data_ns.iloc[0].wave

data_ns.iloc[0].participant_id_PD
id_wave_age_dict[data_ns.iloc[0].participant_id]

data_ns.wave.unique()

data_ns[data_ns.participant_id_PD == data_ns.participant_id_PD.unique()[1]]

obs_columns = ['PreferredSymbol', 'HGVSc', 'chromosome', 'position', 'reference', 'mutation', 'consequence', 'Variant_Classification', 'AF_Outlier_Pvalue', 'X95MDAF', 'type', 'key', 'p_key', 'base_substitution']
uns_info = ['Sex']

for i, data in enumerate([data_ns, data_s]):
    
    data['old_id'] = data['participant_id'].map(inv_id_map)
    data = data[~data.old_id.isin(excluded_samples)]

    # Add mutation key and p_key
    data['key'] = data['PreferredSymbol'] + ' ' + data['base_substitution']
    data['p_key'] = data['PreferredSymbol'] + ' ' + data['protein_substitution']


    # add information about cohort, required for mean age
    cohort = []
    for j, row in data.iterrows():
        if row.old_id[3:5] == '36':
            cohort.append('LBC36')
        else:
            cohort.append('LBC21')
    data['cohort'] = cohort

    participant_list = []

    if i == 0:
        mut_type = 'non_synonymous'
    else:
        mut_type = 'synonymous'
    
    unique_ids = data['old_id'].unique()

    # Cannot retrieve direct age information from meta as it's incomplete
    # Compute age approximation by using mean of participants in WAVE
    cohort_age_dict = dict({'LBC21': 79, 'LBC36':70})
    
    # Create new column with age information
    data['age'] = data['cohort'].map(cohort_age_dict)
    data['age'] = data['age'] + 3*(data['wave']-1)

    # # create sex column
    # meta[['ID', 'sex']].groupby('ID').agg(['unique'])
    
    # Slice data assocaited with participants
    for id in unique_ids:

        slice_data = data[data['old_id'] == id].copy()

        unique_time_points = slice_data.age.unique()

        if unique_time_points.shape[0] == 1 :
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
        
        new_ad.layers['AO'] = np.reshape(slice_data['UAO'],
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

    if i == 0:
        with open('../exports/LBC/LBC_non_syn_cohort_1.pk', 'wb') as f:
            pk.dump(participant_list, f)

    if i == 1:
        with open('../exports/LBC/LBC_syn_cohort_1.pk', 'wb') as f:
            pk.dump(participant_list, f)
