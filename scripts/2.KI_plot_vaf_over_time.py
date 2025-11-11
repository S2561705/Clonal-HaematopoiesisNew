import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load processed MDS cohort AnnData objects
with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    mds_participant_list = pickle.load(f)

# Output directory for plots
output_dir = '../exports/MDS_VAF_over_time'
os.makedirs(output_dir, exist_ok=True)

for adata in mds_participant_list:
    participant_id = adata.obs['participant_id'].iloc[0]
    df = adata.obs.copy()
    
    # Determine time column
    if 'VISIT_NUMBER' in df.columns:
        time_col = 'VISIT_NUMBER'
    elif 'age' in df.columns:
        time_col = 'age'
    else:
        print(f"No time column for participant {participant_id}, skipping.")
        continue

    # Plot setup
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Plot each mutation
    for mut in df['key'].unique():
        mut_df = df[df['key'] == mut].sort_values(time_col)
        x = mut_df[time_col]
        y = mut_df['AF']
        
        # Calculate binomial standard error: sqrt(p*(1-p)/DP)
        se = np.sqrt(y * (1 - y) / mut_df['DP'])
        
        # Plot VAF line
        plt.plot(x, y, marker='o', label=mut)
        # Add shaded error area
        plt.fill_between(x, y - se, y + se, alpha=0.3)
    
    plt.xlabel(time_col)
    plt.ylabel('VAF')
    plt.title(f'Participant {participant_id}: VAF of Mutations Over Time')
    plt.ylim(0, 1)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{participant_id}_vaf_over_time.png'))
    plt.close()

print(f'Plots saved to {output_dir}')
