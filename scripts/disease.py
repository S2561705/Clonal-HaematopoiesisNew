# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

from scipy import stats


# Import results
participant_df = pd.read_csv('../results/participant_df.csv', index_col=0)
# create cohort_specific_id
participant_df['participant_id'] = participant_df.apply(lambda row: row['cohort'] + row['participant_id'] if not row['cohort'].startswith('LBC') else row['participant_id'], axis=1)
duplicated_ids = participant_df[participant_df.duplicated(subset='participant_id', keep=False)]['participant_id'].unique()
participant_df = participant_df.drop_duplicates(subset='participant_id', keep='first')

# Import cohort
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)


new_cohort = []

for part in cohort:
    # modify cohort specific id
    if part.uns['cohort'] != 'LBC':
        part.uns['participant_id'] = part.uns['cohort'] + str(part.uns['participant_id'])

    participant_id = part.uns['participant_id']
    
    if participant_id in duplicated_ids:
        # Check if this is the first instance of this participant_id
        if not any(p.uns['participant_id'] == participant_id for p in new_cohort):
            new_cohort.append(part)
    else:
        # If not a duplicated ID, always keep it
        new_cohort.append(part)

# Replace the original cohort with the new one
cohort = new_cohort


lbc21 = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC21/LBC1921_Survival_and_SingleTimepoint.obs_matrix.tsv', sep='\t')
lbc36 = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC36/LBC1936_Survival_and_SingleTimepoint.obs_matrix.tsv', sep='\t')
WHI = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/WHI/WHI_SingleTimepoint_Disease.obs_matrix.tsv', sep='\t')
WHI = WHI.rename(columns={'ID':'participant_id'})
WHI.participant_id = WHI.participant_id.astype(str)
WHI.participant_id = 'WHI' + WHI.participant_id
marker_map = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/cohort_param_values_map.txt', sep='\t')

for col in lbc21.columns:
    if col in marker_map['LBC21']:
        lbc21.rename(columns={col: ['LBC21', 'global_name']})

for col in lbc36.columns:
    if col in marker_map['LBC36']:
        lbc36.rename(columns={col: ['LBC36', 'global_name']})

for col in WHI.columns:
    if col in list(marker_map['WHI']):
        WHI = WHI.rename(columns={col: marker_map.loc[marker_map.WHI==col].global_name.values[0]})

df = pd.concat([lbc21,lbc36,WHI])
df = pd.merge(participant_df, df, on='participant_id')

# %%


marker = 'HRT'
print(df[marker].value_counts())
formula = ('max_size_prediction_120_z_score',
           'max_fitness_z_score',
           'age_wave_1')
formatted_formula = ' +'.join(formula)
res = ols(data=df, formula=marker + '~' + formatted_formula).fit()
res.summary()


# %%
