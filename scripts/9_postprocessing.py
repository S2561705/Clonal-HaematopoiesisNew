import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.deterministic_aux import *
from src.preprocessing_aux import *
from src.blood_markers_aux import *
from src.aux import *

# import participant_data
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

# drop participants with warnings
cohort = [part for part in cohort if part.uns['warning'] is None]

# drop MYC participant
cohort = [part for part in cohort if part.uns['participant_id']!='LBC360020']

# Compute overall and within clone order of mutations
cohort = [order_of_mutations(part) 
    for part in tqdm(cohort)]


# update LBC survival dataframe
LBC_survival_information(cohort)

# add extra information for participants
for part in cohort:
    # mutation information
    part.obs['mut_max_VAF'] = part.X.max(axis=1)
    part.obs['mut_VAF_tp_0'] = part.X[:, 0]
    part.obs['part_age_tp_0'] = part.var.time_points.min()
    part.obs['clonal_structure_size'] = part.obs.clonal_structure.apply(len)
    part.obs['size_prediction_120'] = np.exp(part.obs.fitness*(120-part.obs.init_age))
    part.obs['size_prediction_next_30'] = np.exp(part.obs.fitness*(30+part.obs.part_age_tp_0-part.obs.init_age))
    # part.obs['VAF_prediction_120'] = (
    #     part.obs.size_prediction_120/(2*(
    #         100_000 + part.obs.size_prediction_120)))

    # part.obs['VAF_prediction_next_30'] = (
    #     part.obs.size_prediction_next_30/(2*(
    #         100_000 + part.obs.size_prediction_next_30)))
    
    # participant_information
    part.uns['survival_df']['sex'] = part.obs.sex.unique()[0]
    part.uns['survival_df']['max_VAF_tp_0'] = part.X[:, 0].max()
    part.uns['survival_df']['max_VAF'] = part.X.max()
    part.uns['survival_df']['max_fitness'] = part.obs.fitness.max()
    part.uns['survival_df']['n_mutations'] = part.shape[0]
    part.uns['survival_df']['n_clones'] = part.obs.clonal_index.max()
    part.uns['survival_df']['max_size_prediction_120'] = part.obs.size_prediction_120.max()
    part.uns['survival_df']['max_size_prediction_next_30'] = part.obs.size_prediction_next_30.max()
    # part.uns['survival_df']['max_VAF_prediction_120'] = part.obs.VAF_prediction_120.max()
    # part.uns['survival_df']['max_VAF_prediction_next_30'] = part.obs.VAF_prediction_next_30.max()

    
# Export participant-level dataframe
all_columns = [set(part.uns['survival_df'].columns) for part in cohort]
overlap_columns = list(set.intersection(*all_columns))

participant_df = pd.concat([part.uns['survival_df'][overlap_columns] for part in cohort])

# Create binary Sex classification
participant_df['Female'] = 1*(participant_df.sex == 'F')

normalise_parameter(participant_df, 'max_fitness')
normalise_parameter(participant_df, 'max_VAF_tp_0')
normalise_parameter(participant_df, 'max_VAF')
normalise_parameter(participant_df, 'max_size_prediction_120')
normalise_parameter(participant_df, 'max_size_prediction_next_30')

# # Normalise maximum VAF and fitness across participants
# participant_df['norm_max_VAF'] = normalise_column(participant_df.max_VAF)
# participant_df['norm_max_fitness'] = normalise_column(participant_df['max_fitness'])

# # Log transform and normalise maximum size prediction at 120 and in the next 30 years
# participant_df['log_max_size_prediction_120'] = np.log(participant_df.max_size_prediction_120)
# participant_df['norm_log_max_size_prediction_120'] =  normalise_column(
#     participant_df['log_max_size_prediction_120'])

# participant_df['log_max_size_prediction_next_30'] = np.log(participant_df.max_size_prediction_next_30)
# participant_df['norm_log_max_size_prediction_next_30'] =  normalise_column(
#     participant_df['log_max_size_prediction_next_30'])
 
participant_df = participant_df.reset_index(drop=True)
participant_df.to_csv('../results/participant_df.csv')

# Export mutation-level dataframe
overlaping_columns = [set(part.obs.columns) for part in cohort]
overlaping_columns = list(set.intersection(*overlaping_columns))

summary = pd.concat([part.obs[overlaping_columns] for part in cohort])
summary = summary.sort_values(by='fitness', ascending=False)

normalise_parameter(summary, 'fitness')
normalise_parameter(summary, 'size_prediction_120')
normalise_parameter(summary, 'mut_VAF_tp_0')

summary.to_csv('../results/mutation_df.csv')

# export participant_data
with open('../exports/final_participant_list.pk', 'wb') as f:
    pk.dump(cohort, f)
