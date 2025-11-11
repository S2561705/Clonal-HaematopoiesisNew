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

print(f"Processing {len(mds_df['participant_id'].unique())} participants")

# Process each participant
mds_participant_list = []

for part_id in mds_df['participant_id'].unique():
    part_data = mds_df[mds_df['participant_id'] == part_id].copy()
    if part_data.empty:
        continue

    print(f"\nProcessing participant: {part_id}")
    
    # Get unique timepoints and mutations
    timepoints = sorted(part_data['VISIT_NUMBER'].unique())
    mutations = part_data['key'].unique()
    
    print(f"  Timepoints: {timepoints}")
    print(f"  Mutations: {len(mutations)}")
    
    # Create AO and DP matrices
    AO_matrix = np.zeros((len(mutations), len(timepoints)))  # Shape: (mutations, timepoints)
    DP_matrix = np.zeros((len(mutations), len(timepoints)))  # Shape: (mutations, timepoints)
    
    # Fill matrices
    mutation_to_idx = {mut: i for i, mut in enumerate(mutations)}
    timepoint_to_idx = {tp: i for i, tp in enumerate(timepoints)}
    
    for _, row in part_data.iterrows():
        mut_idx = mutation_to_idx[row['key']]
        time_idx = timepoint_to_idx[row['VISIT_NUMBER']]
        AO_matrix[mut_idx, time_idx] = row['AF'] * row['DP']  # Calculate AO
        DP_matrix[mut_idx, time_idx] = row['DP']
    
    # Create observation dataframe (mutations)
    obs_dict = {}
    for col in ['PreferredSymbol', 'HGVSc', 'protein_substitution']:
        obs_dict[col] = part_data.groupby('key')[col].first().reindex(mutations).values
    
    obs_df = pd.DataFrame(obs_dict, index=mutations)
    
    # Create variable dataframe (timepoints)
    var_df = pd.DataFrame({'time_points': timepoints}, index=[f'tp_{tp}' for tp in timepoints])
    
    # Create AnnData object
    adata = AnnData(
        X=np.zeros((len(mutations), len(timepoints))),  # Dummy X matrix
        obs=obs_df,
        var=var_df
    )
    
    # Store matrices in layers - Shape should be (n_mutations, n_timepoints)
    adata.layers['AO'] = AO_matrix.astype(float)
    adata.layers['DP'] = DP_matrix.astype(float)
    
    print(f"  AO layer shape: {adata.layers['AO'].shape}")
    print(f"  DP layer shape: {adata.layers['DP'].shape}")
    
    # Add metadata to uns
    adata.uns['participant_id'] = part_id
    adata.uns['cohort'] = 'MDS'
    adata.uns['mutation_type'] = 'non_synonymous'
    
    # Verify everything is set
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  uns keys: {list(adata.uns.keys())}")
    print(f"  var columns: {list(adata.var.columns)}")
    
    mds_participant_list.append(adata)

print(f"\nProcessed {len(mds_participant_list)} participants")

# Save with protocol 4 for better compatibility
os.makedirs('../exports/MDS', exist_ok=True)
output_path = '../exports/MDS/MDS_cohort_processed.pk'

# Test saving and loading one object first
print("\nTesting save/load for first participant...")
test_adata = mds_participant_list[0]
with open('../exports/MDS/test_save.pk', 'wb') as f:
    pickle.dump([test_adata], f, protocol=4)

with open('../exports/MDS/test_save.pk', 'rb') as f:
    test_loaded = pickle.load(f)[0]
    print(f"Test load - Layers: {list(test_loaded.layers.keys())}")
    print(f"Test load - Shape: {test_loaded.shape}")

# Now save all
with open(output_path, 'wb') as f:
    pickle.dump(mds_participant_list, f, protocol=4)

print(f"Saved to {output_path}")

# Verify the save worked
print("\nVerifying saved data...")
with open(output_path, 'rb') as f:
    verified_data = pickle.load(f)

print(f"Verified: {len(verified_data)} participants")
for i, part in enumerate(verified_data):
    print(f"Part {i}: Layers: {list(part.layers.keys())}, Shape: {part.shape}, Participant: {part.uns.get('participant_id', 'Unknown')}")
    if 'AO' in part.layers:
        print(f"  AO shape: {part.layers['AO'].shape}")