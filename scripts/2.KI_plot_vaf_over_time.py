import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load processed MDS cohort AnnData objects
with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    mds_participant_list = pickle.load(f)

output_dir = '../exports/MDS_VAF_over_time'
os.makedirs(output_dir, exist_ok=True)

for adata in mds_participant_list:
    participant_id = adata.uns.get('participant_id', 'Unknown')
    AO = adata.layers['AO']
    DP = adata.layers['DP']
    mutations = adata.obs.index.values
    timepoints = adata.var['time_points'].values

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    for mut_idx, mut in enumerate(mutations):
        vaf = AO[mut_idx, :] / DP[mut_idx, :]
        se = np.sqrt(vaf * (1 - vaf) / DP[mut_idx, :])
        plt.plot(timepoints, vaf, marker='o', label=mut)
        plt.fill_between(timepoints, vaf - se, vaf + se, alpha=0.3)

    plt.xlabel('Timepoint')
    plt.ylabel('VAF')
    plt.title(f'Participant {participant_id}: VAF of Mutations Over Time')
    plt.ylim(0, 1)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{participant_id}_vaf_over_time.png'))
    plt.close()

print(f'Plots saved to {output_dir}')


