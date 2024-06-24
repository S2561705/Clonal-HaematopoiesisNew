# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
import pandas as pd

with open('../exports/Fabre.pk', 'rb') as f:
    participant_list = pk.load(f)

participant_list[0].obs
sns.histplot(x=[part.shape[0] for part in participant_list])
plt.clf()

participant_list[0].uns
for part in participant_list:
    if part.shape[0]>10:
        break

plot_part(part, cohort='Fabre')

# %%

data = pd.read_csv('../data/Fabre_final_data.csv', sep=';', decimal=',')

data['Gene_protein'] = data['Gene'] + data['Protein']
obs_columns = ['Gene', 'Chromosome', 'Start', 'End', 'WT', 'MT', 'Protein', 'Effect', 'Gene_protein']
uns_info = ['Sex']
unique_ids = data['Sample ID'].unique()

fabre_2 = []
unique_ids
for id in unique_ids[:-1]:
    slice_data = data[data['Sample ID'] == id].copy()
    slice_data = slice_data.sort_values(by=['Gene_protein', 'Age'])
    unique_time_points = slice_data.Age.unique()

    if (len(slice_data)/unique_time_points.shape[0]).is_integer():
        delta_t = np.diff(unique_time_points, prepend=unique_time_points[0])
        var_df = pd.DataFrame(data=np.array([unique_time_points, delta_t]).T, columns=['time_points', 'delta_t'])
        obs_df = slice_data[slice_data.Age == unique_time_points[0]][obs_columns]
        obs_df = obs_df.set_index('Gene_protein')

        gender = slice_data.Sex.unique()

        new_ad = ad.AnnData(np.reshape(slice_data['VAF'],
                                            (len(obs_df),
                                            len(unique_time_points))),
                            obs=obs_df,
                            var=var_df
                            )
        
        new_ad.uns['participant_id'] = id
        new_ad.uns['gender'] = gender[0]
        
        fabre_2.append(new_ad)

# %%

sns.histplot(x=[part.shape[0] for part in fabre_2], label='no-reads')
sns.histplot(x=[part.shape[0] for part in participant_list], label='reads')
plt.legend()
plt.show()
plt.clf()

for part in participant_list:
    if part.shape[0]>10:
        plot_part(part, cohort='Fabre')
plt.show()
plt.clf()

for part in fabre_2:
    if part.shape[0]>10:
        plot_part(part)
plt.show()
plt.clf()


# %%
