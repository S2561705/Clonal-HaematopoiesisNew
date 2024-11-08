# Import necessary libraries and set up the environment
import sys
sys.path.append("..")   # Add parent directory to Python path for imports
from src.general_imports import *
from src.preprocessing_aux import *

warnings.filterwarnings(action='ignore', category=UserWarning)

# Load non-synonymous and synonymous mutation data
data_ns = pd.read_csv('../data/LBC/mutation_data/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv', sep='\t')
data_s = pd.read_csv('../data/LBC/mutation_data/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.synonymous.tsv', sep='\t')

# Load list of excluded participants
excluded_samples = load_json('../resources/excluded_samples.json')

# Load participant ID mapping and create inverse map
id_map = load_pickle('../resources/LBC_cohort_1_id_map.pkl')
inv_id_map ={v: k for k, v in id_map.items()}

# Load metadata
meta = pd.read_csv('../data/LBC/lbc_meta.csv')

# create sex dictionary for mapping
sex_dict = meta.groupby('ID').max()[['sex']].to_dict()['sex']

# Create a mapping of participant ID and wave to age
age_mapping = meta.set_index(['ID', 'WAVE'])['age'].to_dict()

# Create a function to determine the cohort based on participant_id
def get_cohort(participant_id):
    if participant_id.startswith('LBC36'):
        return 'LBC36'
    else:
        return 'LBC21'

# Function to get age for a participant at a specific wave
def get_age(row):
    if row['wave'] == 5:
        if row['sub_cohort'] == 'LBC36':
            return 83  # 71 + 3 * 4

        elif row['sub_cohort'] =='LBC21':
            return 91  # 79 + 3 * 4

    return age_mapping.get((row['participant_id'], row['wave']), None)
        
# Define columns to be used for observations
obs_columns = ['PreferredSymbol', 'HGVSc', 'chromosome', 'position', 'reference', 'mutation', 'consequence', 'Variant_Classification', 'AF_Outlier_Pvalue', 'X95MDAF', 'type', 'key', 'p_key', 'base_substitution']

# Process both non-synonymous and synonymous data
for i, data in enumerate([data_ns, data_s]):
    
    # Map participant IDs and exclude specified samples
    data['scrambled_participant_id'] = data['participant_id']
    data['participant_id'] = data['participant_id'].map(inv_id_map)
    data = data[~data.participant_id.isin(excluded_samples)].copy()
    
    # Add cohort and subcohort column to data
    data['cohort'] = 'LBC'
    data['sub_cohort'] = data['participant_id'].apply(get_cohort)

    # Add the age column to data_ns
    data['age'] = data.apply(get_age, axis=1)

    # List of cohorts
    sub_cohorts = ['LBC21', 'LBC36']

    # List of waves (assuming waves are 1, 2, 3, 4 - adjust if necessary)
    waves = data['wave'].unique()

    # Fill NaN values for each sub_cohort and wave
    for sub_cohort in sub_cohorts:
        for wave in waves:
            mask = (data['sub_cohort'] == sub_cohort) & (data['wave'] == wave)
            avg_age = data.loc[mask, 'age'].mean()
            data.loc[mask, 'age'] = data.loc[mask, 'age'].fillna(avg_age)

    # append sex of participant to mutation
    data['sex'] = data['participant_id'].map(sex_dict)
    
    # Set mutation type based on current dataset
    if i == 0:
        mut_type = 'non_synonymous'
    else:
        mut_type = 'synonymous'

    # Add mutation key and protein key
    data['key'] = data['PreferredSymbol'] + ' ' + data['base_substitution']
    data['p_key'] = data['PreferredSymbol'] + ' ' + data['protein_substitution']
        
    # create list of participants
    participant_list = []
    unique_part_ids = data['participant_id'].unique()
    
    # Loop trhough data of each participant
    for part_id in unique_part_ids:
        # process participant data
        new_ad = preprocess_participant(data, part_id, obs_columns)
        if new_ad is None:
            continue

        # Add LBC specific metadata to the AnnData object
        new_ad.uns['mutation_type'] = mut_type

        # append new participant
        participant_list.append(new_ad)


    # Filter cohort based on qc
    if i == 0:
        output_file = 'LBC_1_non_qc.txt'
    else:
        output_file = 'LBC_1_syn_qc.txt'
       
    filtered_participant_list = cohort_qc(participant_list, output_file=output_file)

    LBC_survival_information(filtered_participant_list)

    # Save processed data
    if i == 0:
        dump_pickle(filtered_participant_list,
                    '../exports/LBC/LBC_non_syn_cohort_1_qc.pk')

    if i == 1:
        dump_pickle(filtered_participant_list,
                    '../exports/LBC/LBC_syn_cohort_1_qc.pk')