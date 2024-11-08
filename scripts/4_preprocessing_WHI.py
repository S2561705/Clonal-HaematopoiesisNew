# Import necessary libraries and set up the environment
import sys
sys.path.append("..")   # Add parent directory to Python path for imports
from src.general_imports import *
from src.preprocessing_aux import *

warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

cohort = 'WHI'

# Load the data
data = pd.read_csv('../data/WHI/randname.lowfreq.sam.allobs.freeze2021Sep27_commonid_COSMIC_CORRECTED.csv')

# set cohort in data
data['cohort'] = 'WHI'
data['sub_cohort'] = 'WHI'

# set sex as always female as this is a women's cohort
data['sex'] = 'F'

# Keep only exonic mutations
data = data[data['Func.refGene'] == 'exonic']

# Rename columns for consistency
data = data.rename(columns={
                        'commonid':'participant_id',
                        'DrawAge': 'age',
                        'Gene.refGene':'PreferredSymbol',
                        'transcriptOI': 'HGVSc',
                        'pos': 'position',
                        'chrom': 'chromosome',
                        'ref': 'reference',
                        'alt': 'mutation',
                        'vaf': 'AF',
                        'count_alt': 'AO',
                        'depth': 'DP'})

# Create unique keys for each mutation
key_list = []
for i, row in data.iterrows():
    key = row['PreferredSymbol'] + ' '
    for ann in row['HGVSc'].split(':'):
        if 'c.' in ann:
            ann_split = ann.split('.')
            if ann_split[1][0].isdigit():
                key += ann
            else:
                key += (
                    'c.' + ann_split[1][1:-1]
                    +  ann_split[1][0] + '>'
                    + ann_split[1][-1])
    key_list.append(key)

data['key'] = key_list

# Create protein change key
data['p_key'] = data['PreferredSymbol'] +' p.' + data['NonsynOI']

# Define columns to be used for observations
obs_columns = ['key', 'PreferredSymbol', 'HGVSc', 'chromosome', 'position', 'reference', 'mutation', 'NonsynOI', 'p_key']

# create list of participants
participant_list = []
unique_part_ids = data['participant_id'].unique()

# Loop trhough data of each participant
for part_id in unique_part_ids:
    # process participant data
    new_ad = preprocess_participant(data, part_id, obs_columns)
    if new_ad is None:
        continue

    # append new participant
    participant_list.append(new_ad)


# Filter cohort based on qc 
filtered_participant_list = cohort_qc(participant_list, output_file='WHI_qc.txt')

WHI_survival_information(filtered_participant_list)

# Save the filtered data
with open('../exports/WHI/WHI_qc.pk', 'wb') as f:
    pk.dump(filtered_participant_list, f)
