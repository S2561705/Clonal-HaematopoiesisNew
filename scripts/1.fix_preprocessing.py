import pandas as pd
import numpy as np
import os
from anndata import AnnData
import pickle

# Load and clean your data
rename_map = {
    'SAMPLE_ID': 'participant_id',
    'GENE': 'PreferredSymbol',
    'cDNA_CHANGE': 'HGVSc',
    'PROTEIN_CHANGE': 'protein_substitution',
    'VAF': 'AF',
    'READ_DEPTH': 'DP',
    'AGE': 'age',
    'SEX': 'sex',
}

print("Loading MDS cohort data...")
mds_df = pd.read_csv('../data/MDS_COHORT.csv', delimiter=';', dtype=str)
mds_df = mds_df.rename(columns=rename_map)

def clean_column(series):
    series = series.replace('NULL', np.nan)
    series = series.apply(lambda x: str(x).replace(',', '.') if pd.notnull(x) else x)
    series = series.apply(lambda x: 0.5 if pd.notnull(x) and str(x).startswith('<') else x)
    return pd.to_numeric(series, errors='coerce')

mds_df['AF'] = clean_column(mds_df['AF']) / 100
mds_df['age'] = clean_column(mds_df['age'])
mds_df['DP'] = clean_column(mds_df['DP'])
mds_df['VISIT_NUMBER'] = mds_df['VISIT_NUMBER'].astype(int)

mds_df['wave'] = mds_df['VISIT_NUMBER']
mds_df['cohort'] = 'MDS'
mds_df['key'] = mds_df['PreferredSymbol'] + ' ' + mds_df['HGVSc']
mds_df['p_key'] = mds_df['PreferredSymbol'] + ' ' + mds_df['protein_substitution']

mds_df = mds_df.sort_values(['participant_id', 'VISIT_NUMBER'])
mds_df['DELTA_YEARS'] = mds_df.groupby('participant_id')['age'].diff().fillna(0)

mds_df = mds_df.dropna(subset=['AF', 'DP', 'age'])
mds_df = mds_df[mds_df['DP'] > 0]

print(f"Found {len(mds_df['participant_id'].unique())} participants")

# Process each participant
mds_participant_list = []
obs_columns = ['PreferredSymbol', 'HGVSc', 'protein_substitution', 'AF', 'DP', 'age', 'sex', 'key', 'p_key', 'VISIT_NUMBER']

for part_id in mds_df['participant_id'].unique():
    part_data = mds_df[mds_df['participant_id'] == part_id].copy()
    if part_data.empty:
        continue

    print(f"\nProcessing participant: {part_id}")
    print(f"part_data[obs_columns] for {part_id}:")
    print(part_data[obs_columns].head())
    
    # Get unique timepoints and mutations
    timepoints = sorted(part_data['VISIT_NUMBER'].unique())
    mutations = part_data['key'].unique()
    
    print(f"  Timepoints: {timepoints}")
    print(f"  Mutations: {len(mutations)}")
    
    # Create AO and DP matrices with shape (n_mutations, n_timepoints)
    AO_matrix = np.zeros((len(mutations), len(timepoints)))
    DP_matrix = np.zeros((len(mutations), len(timepoints)))
    
    # Fill matrices
    mutation_to_idx = {mut: i for i, mut in enumerate(mutations)}
    timepoint_to_idx = {tp: i for i, tp in enumerate(timepoints)}
    
    for _, row in part_data.iterrows():
        mut_idx = mutation_to_idx[row['key']]
        time_idx = timepoint_to_idx[row['VISIT_NUMBER']]
        AO_matrix[mut_idx, time_idx] = row['AF'] * row['DP']  # Calculate AO
        DP_matrix[mut_idx, time_idx] = row['DP']
    
    # Create observation dataframe (mutations)
    obs_data = {}
    for col in ['PreferredSymbol', 'HGVSc', 'protein_substitution']:
        obs_data[col] = part_data.groupby('key')[col].first().reindex(mutations).values
    
    obs_df = pd.DataFrame(obs_data, index=mutations)
    
    # Create variable dataframe (timepoints)
    var_df = pd.DataFrame({'time_points': timepoints}, index=[f'tp_{i}' for i in range(len(timepoints))])
    
    # Create a simple X matrix (required by AnnData)
    X_matrix = np.zeros((len(mutations), len(timepoints)))
    
    # Create AnnData object
    adata = AnnData(
        X=X_matrix,  # Required - use zeros as placeholder
        obs=obs_df,
        var=var_df
    )
    
    # Store matrices in layers
    adata.layers['AO'] = AO_matrix.astype(float)
    adata.layers['DP'] = DP_matrix.astype(float)
    
    # Add metadata
    adata.uns['participant_id'] = part_id
    adata.uns['cohort'] = 'MDS'
    adata.uns['mutation_type'] = 'non_synonymous'
    
    # Verify everything is set
    print(f"  Created AnnData with shape: {adata.shape}")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  AO layer shape: {adata.layers['AO'].shape}")
    print(f"  uns keys: {list(adata.uns.keys())}")
    print(f"  var columns: {list(adata.var.columns)}")
    
    mds_participant_list.append(adata)

print(f"\nProcessed {len(mds_participant_list)} participants")

# Save the data
os.makedirs('../exports/MDS', exist_ok=True)
output_path = '../exports/MDS/MDS_cohort_processed.pk'

print(f"\nSaving to {output_path}...")
with open(output_path, 'wb') as f:
    pickle.dump(mds_participant_list, f, protocol=4)

# Verify the save worked
print("Verifying save...")
with open(output_path, 'rb') as f:
    verified_data = pickle.load(f)

print(f"Verified: {len(verified_data)} participants")
for i, part in enumerate(verified_data):
    print(f"Part {i}: Shape: {part.shape}, Layers: {list(part.layers.keys())}, Participant: {part.uns.get('participant_id', 'Unknown')}")

print("MDS cohort preprocessing complete!")