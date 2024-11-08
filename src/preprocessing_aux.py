# Import necessary libraries and set up the environment
import sys
sys.path.append("..")   # Add parent directory to Python path for imports
from src.general_imports import *
import os

from scipy.stats import norm

def preprocess_participant(data, part_id, obs_columns):
    # create slice of data associated to participant
    slice_data = data[data['participant_id'] == part_id].copy()

    # drop observations with 0 read depth
    slice_data = slice_data[slice_data.DP!=0].copy()
    
    # compute unique time points
    unique_time_points = slice_data.age.unique()

    # Skip if only one time point
    if unique_time_points.shape[0] == 1 :
        return None

    # Find mutations with observations at all time points
    valid_keys = [key for key in slice_data.key.unique() 
                if len(slice_data[slice_data.key == key])==len(unique_time_points)]
    slice_data = slice_data[slice_data.key.isin(valid_keys)].copy()

    # Sort data by mutation key and age
    slice_data = slice_data.sort_values(by=['key', 'age'])
    unique_time_points = slice_data.age.unique()

    # check if unique_time_points is sorted        
    if np.all(np.diff(unique_time_points) >= 0) is False:
        print('TIME POINTS NOT SORTED')
        return None

    # Calculate time differences
    delta_t = np.diff(unique_time_points, prepend=unique_time_points[0])
    var_df = pd.DataFrame(data=np.array([unique_time_points, delta_t]).T, columns=['time_points', 'delta_t'])
    obs_df = slice_data[slice_data.age == unique_time_points[0]][obs_columns]
    obs_df = obs_df.set_index('key')

    # Create AnnData object for the participant
    new_ad = ad.AnnData(np.reshape(np.array(slice_data['AF'])[:, None],
                                        (len(obs_df),
                                        len(unique_time_points))),
                        obs=obs_df,
                        var=var_df
                        )
    
    # Add layers for Alternate allele Observations (AO) and Depth (DP)
    new_ad.layers['AO'] = np.reshape(slice_data['AO'],
                                        (len(obs_df),
                                        len(unique_time_points))
    )

    new_ad.layers['DP'] = np.reshape(slice_data['DP'],
                                        (len(obs_df),
                                        len(unique_time_points))
    )


    # Deal with nan index
    new_index = []
    nan_counter = 0
    for idx in list(new_ad.obs.index):
        if idx != 'nan':
            new_index.append(idx)
        else:
            new_index.append(f'{participant_ID}_{nan_counter}')
            nan_counter += 1
    new_ad.obs.index = new_index

    new_ad.uns['participant_id'] = part_id
    new_ad.uns['cohort'] = slice_data.cohort.unique()[0]
    new_ad.uns['sub_cohort'] = slice_data.sub_cohort.unique()[0]
    new_ad.uns['sex'] = slice_data.sex.unique()[0]

    new_ad.obs['participant_id'] = new_ad.uns['participant_id'] 
    new_ad.obs['sex'] = new_ad.uns['sex']
    new_ad.obs['cohort'] = new_ad.uns['cohort'] 
    new_ad.obs['sub_cohort'] = new_ad.uns['sub_cohort'] 

    # Exclude time points with very low sequencing depth
    # Compute average sequencing depth excluding current time point for each mutation
    time_point_excluded_means = np.array(
        [np.delete(new_ad.layers['DP'], i, axis=1).mean(axis=1)
        for i in range(new_ad.shape[1])]).T

    # Compute proportion between read depth in current time point and average time_point excluded read depth
    proportional_depth_compared_to_excluded_means = (
        new_ad.layers['DP']/time_point_excluded_means)

    # Compute average proportional depth across all mutations
    avg_proportional_depth = proportional_depth_compared_to_excluded_means.mean(axis=0)
    
    # Keep time_points whose proportional read depth is >0.5
    prop_time_points_idx = set(np.argwhere(avg_proportional_depth>0.5).flatten())
    # Keep time points with average read depth > 1_000
    high_depth_tp_idx = set(np.argwhere(new_ad.layers['DP'].mean(axis=0) > 1_000).flatten())
    
    # Compute union of time points qc
    keep_time_points_idx = list(prop_time_points_idx.union(high_depth_tp_idx))

    # Append information on time_points qc 
    new_ad.uns['keep_time_points_idx_DP'] = keep_time_points_idx

    # Drop mutations that never reach 1% VAF in non-excluded
    keep_mut_idx = np.argwhere(new_ad[:, keep_time_points_idx].X.max(axis=1)>0.01).flatten()
    new_ad.uns['keep_mut_idx_1%'] = keep_mut_idx

    # Compute LOH participants:
    # 1. Compute difference between reported VAF and AO/DP
    new_ad.layers['VAF_diff'] = new_ad.X - (new_ad.layers['AO']/new_ad.layers['DP'])
    
    # 2. Compute maximum VAF difference
    new_ad.uns['max_VAF_diff'] = abs(new_ad.layers['VAF_diff']).max()

    return new_ad

def cohort_qc(participant_list, output_file, LOH=False):
    folder_path = "../results/qc_text/"
    os.makedirs(folder_path, exist_ok=True)

    output_file = os.path.join(folder_path, output_file)
    with open(output_file, 'w') as f:

        # Check distribution of corrected VAFs to infer LOH events
        total_vaf_diff =[]
        for part in participant_list:
            total_vaf_diff.extend(list(part.layers['VAF_diff'].flatten()))
        
        sns.histplot(total_vaf_diff)
        plt.show()
        plt.clf()

        if LOH is False:
            print('No LOH events', file=f)

            # Set LOH_event to False for all participants
            for part in participant_list:
                part.uns['LOH_event'] = False
        
        else:           
            # Fit normal distribution to max_VAF_diff
            max_VAF_diff_mean, max_VAF_diff_std = norm.fit(total_vaf_diff)

            max_VAF_2std = max_VAF_diff_mean - 2*max_VAF_diff_std

            # Determine LOH events
            for part in participant_list:
                part.uns['LOH_event'] = part.layers['VAF_diff'].min() < max_VAF_2std

        # Filter only mutations and time_points passing above qcs
        filtered_participant_list = []
        LOH_events = 0
        for part in participant_list:
            if part.uns['LOH_event'] == False:
                new_part = part[part.uns['keep_mut_idx_1%'],
                                part.uns['keep_time_points_idx_DP']].copy()
            else:
                LOH_events += 1

            # Filter mutations with no observation 
            # (only observed time point might have been dropped)
            nonzero_idx = np.argwhere(new_part.layers['AO'].sum(axis=1)!= 0).flatten()
            filtered_participant_list.append(new_part[nonzero_idx])

        # Filter all empty rows and columns
        filtered_participant_list = [part for part in filtered_participant_list if 
                                    (part.shape[0]>0 and part.shape[1]>1)]

        print(f'We detected {LOH_events} participants with LOH events', file=f)

        # Create filtered lists based on mutation and time point criteria
        filtered_participant_list_1 = []
        for part in participant_list:
                filtered_participant_list_1.append(
                    part[part.uns['keep_mut_idx_1%']])

        filtered_participant_list_tp = []
        for part in participant_list:
                filtered_participant_list_tp.append(
                    part[:,part.uns['keep_time_points_idx_DP']])

        # Calculate and print statistics on filtered mutations
        filt_mut = np.array([part.shape[0] for part in filtered_participant_list_1]).sum()
        total_mut = np.array([part.shape[0] for part in participant_list]).sum()

        print(f'{total_mut - filt_mut} mutations out of ' + 
            f'a total of {total_mut} did not pass the QC' +
            f'based on 1% filtering', file=f)

        # Calculate and print statistics on filtered time points
        filt_tp = np.array([part.shape[1] for part in filtered_participant_list_tp]).sum()
        total_tp = np.array([part.shape[1] for part in participant_list]).sum()

        print(f'{total_tp - filt_tp} time_points out of ' + 
            f'a total of {total_tp} did not pass the QC' +
            f'based on stability of read depth', file=f)

        # Print overall statistics
        total_mut = np.array([part.shape[0] for part in filtered_participant_list]).sum()
        total_tp = np.array([part.shape[1] for part in filtered_participant_list]).sum()

        print(f'Overall {len(filtered_participant_list)} participants '+
            f'(out of {len(participant_list)}) '
            f'with a total of {total_mut} mutations '+ 
            f'and {total_tp} time points were selected for clonal fitting', file=f)

    return filtered_participant_list



# LBC survival information
def LBC_survival_information(participant_list):
    lbc21_meta = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC21/LBC1921_Survival_and_SingleTimepoint.obs_matrix.tsv', sep='\t')
    lbc36_meta = pd.read_csv('../data/ERIC_COHORT_EPI_VALUES/LBC36/LBC1936_Survival_and_SingleTimepoint.obs_matrix.tsv', sep='\t')

    lbc21_meta = lbc21_meta.rename(columns={'studyno': 'participant_id',
                        'agedaysApx_LastCensor':'agedays_lastcensor'})

    lbc36_meta = lbc36_meta.rename(columns={'lbc36no': 'participant_id',
                        'AgedaysApx_LastCensor':'agedays_lastcensor'})

    survival_df = pd.merge(lbc21_meta, lbc36_meta, how='outer')

    # filter merged dataset to only fitted participants
    survival_df = survival_df[survival_df.participant_id.isin([part.uns['participant_id'] for part in participant_list])].copy()

    age_wave_1_dict = {part.uns['participant_id']:part.var.time_points.min() for part in participant_list}
    survival_df['age_wave_1'] = survival_df['participant_id'].map(age_wave_1_dict)

    # find maximum between death and censorhip
    survival_df['ageyrs_lastcensor'] = survival_df['agedays_lastcensor']/365.2422
    survival_df['age_death'] = survival_df[['ageyrs_death', 'ageyrs_lastcensor']].max(axis=1)
    survival_df['from_wave_1'] = survival_df['age_death'] - survival_df['age_wave_1']
    survival_df['death_cause'] = 'NaN'
    survival_df['death_cause_num'] = np.nan
    survival_df['cohort'] = 'LBC'   

    for part in participant_list:
        if part.uns['cohort'] == 'LBC':
            part.uns['survival_df'] = survival_df[
                survival_df.participant_id == 
                part.uns['participant_id']]



def WHI_survival_information(participant_list):
    # load survival data
    # aging_df = pd.read_csv('../data/Uddin/outc_aging_ctos_inv.dat', sep='\t')
    survival_df = pd.read_csv('../data/WHI/outc_death_all_discovered_inv.dat', sep='\t')

    survival_df = survival_df.rename(columns={'ID': 'participant_id'})
    with open('../resources/WHI_death_cause.json', 'r') as f:
        death_cause_dict = json.load(f)

    death_cause_dict = {int(k):v for k,v in death_cause_dict.items()}
    # filter survival data to sequencing ids
    ids = [part.uns['participant_id'] for part in participant_list]
    # aging_df = aging_df[aging_df.ID.isin(ids)].copy()
    survival_df = survival_df[survival_df.participant_id.isin([part.uns['participant_id'] for part in participant_list])].copy()

    survival_df = survival_df.rename(columns={'DEATHALL': 'dead',
                                'DEATHALLCAUSE': 'death_cause_num',
                                'ENDFOLLOWALLDY': 'from_wave_1_days'})

    # create age_wave_1 dict
    age_wave_1_dict = {part.uns['participant_id']:part.var.time_points.min() for part in participant_list}
    survival_df['age_wave_1'] = survival_df['participant_id'].map(age_wave_1_dict)

    survival_df['from_wave_1'] = survival_df['from_wave_1_days']/365.2422
    survival_df['age_death'] = survival_df['age_wave_1'] + survival_df['from_wave_1']
    # Filter survival_df and add cause of death
    survival_df = survival_df[['participant_id', 'dead', 'age_death', 'death_cause_num', 'from_wave_1', 'age_wave_1']].copy()
    survival_df['death_cause'] = survival_df['death_cause_num'].map(death_cause_dict)
    survival_df['cohort'] = 'WHI'


    for part in participant_list:
        part.uns['survival_df'] = survival_df[
            survival_df.participant_id == 
            part.uns['participant_id']]



def sardinia_survival_information(participant_list):

    # load survival data
    survival_df = pd.read_csv('../data/sardiNIA/fabre_deaths upd 2024.csv')

    survival_df = survival_df.rename(columns={'ID':'participant_id'})
    survival_df = survival_df[survival_df.participant_id.isin([
                                part.uns['participant_id']
                                for part in participant_list])].copy()

    survival_df['dead'] = ~np.isnan(survival_df.Death_AGE)*1
    survival_df['age_death'] = np.nanmax(
        survival_df[['AGE1', 'AGE2', 'AGE3',
        'AGE4', 'AGE5', 'Death_AGE']],
        axis=1)

    # create age_wave_1 dict
    age_wave_1_dict = {part.uns['participant_id']:part.var.time_points.min() for part in participant_list}
    survival_df['age_wave_1'] = survival_df['participant_id'].map(age_wave_1_dict)

    survival_df['dead'] = survival_df['dead'].astype('int')
    survival_df['from_wave_1'] = (survival_df['age_death'] 
                - survival_df['age_wave_1'])

    survival_df['cohort'] = 'sardiNIA'
    survival_df['death_cause'] = 'NaN'
    survival_df['death_cause_num'] = np.nan

    for part in participant_list:
        part.uns['survival_df'] = survival_df[
            survival_df.participant_id == 
            part.uns['participant_id']]
