import pandas as pd
import numpy as np

# Read the KI cohort CSV with semicolon delimiter
input_path = '../data/MDS_COHORT.csv'
df = pd.read_csv(input_path, delimiter=';', dtype=str)

# Rename columns to match repo expectations
rename_map = {
    'SAMPLE_ID': 'participant_id',
    'GENE': 'PreferredSymbol',
    'VAF': 'AF',
    'READ_DEPTH': 'DP',
    'AGE': 'age',
    # Add more mappings if needed
}
df = df.rename(columns=rename_map)

# Clean VAF column: handle '<1', replace comma with dot, convert to float
def clean_vaf(val):
    if pd.isnull(val) or val == 'NULL':
        return np.nan
    val = val.replace(',', '.')
    if val.startswith('<'):
        return 0.5  # treat <1 as 0.5
    try:
        return float(val)
    except ValueError:
        return np.nan

df['AF'] = df['AF'].apply(clean_vaf)

# Clean AGE column: replace comma with dot, convert to float
def clean_age(val):
    if pd.isnull(val) or val == 'NULL':
        return np.nan
    val = val.replace(',', '.')
    try:
        return float(val)
    except ValueError:
        return np.nan

df['age'] = df['age'].apply(clean_age)

# Clean READ_DEPTH: treat 'NULL' as NaN, convert to int
def clean_depth(val):
    if pd.isnull(val) or val == 'NULL':
        return np.nan
    try:
        return int(val)
    except ValueError:
        return np.nan

df['DP'] = df['DP'].apply(clean_depth)

# Compute AO
if {'AF', 'DP'}.issubset(df.columns):
    df['AO'] = (df['AF'] * df['DP']).round().astype(int)
    df = df[(df['AO'] >= 0) & (df['AO'] <= df['DP'])]

# Ensure age column
if 'age' not in df.columns:
    # Add logic to compute age if needed
    pass

# Create mutation identifier
if {'PreferredSymbol', 'PROTEIN_CHANGE'}.issubset(df.columns):
    df['Gene_protein'] = df['PreferredSymbol'] + '_' + df['PROTEIN_CHANGE']

# Drop missing essential data
required_cols = ['participant_id', 'age', 'PreferredSymbol', 'PROTEIN_CHANGE', 'AF', 'DP', 'AO']
df = df.dropna(subset=required_cols)

# Save cleaned output for inference
output_path = '../data/MDS_COHORT_cleaned.csv'
df.to_csv(output_path, index=False)

print(f'Preprocessing complete. Cleaned file saved to {output_path}')
