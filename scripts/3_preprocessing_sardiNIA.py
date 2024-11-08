# Import necessary libraries and set up the environment
import sys
sys.path.append("..")   # Add parent directory to Python path for imports
from src.general_imports import *
from src.preprocessing_aux import *

warnings.filterwarnings(action='ignore', category=UserWarning)

# Load the variant call data
data = pd.read_csv('../data/sardiNIA/CHvariantCalls_withReadCounts.csv')
cohort = 'sardiNIA'
data['cohort'] = 'sardiNIA'
data['sub_cohort'] = 'sardiNIA'

# Load ID matching data
sardiNIA_fabre_id_match = pd.read_csv('../data/sardiNIA/sardiNIA_id_match.csv')

# Create a dictionary and inverse to map between different ID types
ID_PD_map = dict(zip(sardiNIA_fabre_id_match.PD_ID, sardiNIA_fabre_id_match.Sard_ID))
inv_id_map ={v: k for k, v in ID_PD_map.items()}

# Rename columns and map participant IDs
data = data.rename(columns={'Sample ID':'participant_id_PD'})
data['participant_id'] = data['participant_id_PD'].map(ID_PD_map)
data['participant_id'] = data['participant_id'].astype('str')
data['participant_id'] = 'sardiNIA' + data['participant_id']

# Create unique keys for each mutation
key_list = []
for i, row in data.iterrows():
    key_list.append(
        row['Gene'] + ' ' + 'c.' + str(row['Start']) + row['WT'] + '>' + row['MT'])

data['key'] = key_list
data['p_key'] = data['Gene'] + ' ' + data['Protein']

# Rename columns to standardized names
data = data.rename(columns={'Age': 'age',
                             'Chromosome': 'chromosome',
                             'Start':'position',
                             'Gene':'PreferredSymbol',
                             'WT':'reference',
                             'MT':'mutation',
                             'Effect':'Variant_Classification',
                             'Total Count':'DP',
                             'MT count': 'AO',
                             'VAF':'AF',
                             'Sex':'sex'})

# Define columns to be used for observations
obs_columns = ['PreferredSymbol', 'chromosome', 'position', 'reference', 'mutation', 'Variant_Classification', 'key', 'p_key']

participant_list = []

unique_part_ids = data['participant_id'].unique()

# Loop trhough data of each participant
for part_id in unique_part_ids:
    # process participant data
    new_ad = preprocess_participant(data, part_id, obs_columns)
    if new_ad is None:
        continue
        new_ad.uns['participant_id_PD'] = inv_id_map[new_ad.uns['participant_id']]
    
    # append new participant
    participant_list.append(new_ad)

# Filter cohort based on qc 
filtered_participant_list = cohort_qc(participant_list, output_file='sardiNIA_qc.txt', LOH=True)

sardinia_survival_information(filtered_participant_list)

# Save the filtered data
with open('../exports/sardiNIA/sardiNIA_qc.pk', 'wb') as f:
    pk.dump(filtered_participant_list, f)