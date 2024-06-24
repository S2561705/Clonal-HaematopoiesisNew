import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.aux import *
import pymc as pm
import os


def extract_priors(participant_list, cohort=None):

    # create dictionary of mutations
    mut_dict = dict()
    for part in participant_list:
        for mut in part:
            key = mut.obs.index[0]
            if key not in mut_dict.keys():
                mut_dict[key] = [mut]
            else:
                mut_dict[key].append(mut)

    # Compute Binomial priors
    # for singly occuring mutations
    single_occurence = [data_list for data_list in mut_dict.values() 
                        if len(data_list)==1]
    p_list = []
    for data_list in tqdm(single_occurence):
        with pm.Model() as binom:
            p = pm.Uniform('p', lower=0, upper=1)
            AO_data = np.hstack([data.layers['AO'].flatten() for data in data_list])
            DP_data = np.hstack([data.layers['DP'].flatten() for data in data_list])

            obs = pm.Binomial('obs', p=p, n=DP_data, observed=AO_data)
            map = pm.find_MAP(progressbar=False)

        p_list.append(map['p'])

    prior_binom_p = extract_kde(np.array(p_list))


    # Beta Binomial priors
    # for recurrent mutations
    alpha_list = []
    beta_list = []
    multiple_occurence = [data_list for data_list in mut_dict.values() 
                        if len(data_list)>1]

    for data_list in tqdm(multiple_occurence):
        AO_data = np.hstack([data.layers['AO'].flatten() for data in data_list])
        DP_data = np.hstack([data.layers['DP'].flatten() for data in data_list])

        with pm.Model() as beta_binom:
            
            p_bb = pm.Uniform('p_bb', lower=0, upper=0.05)
            beta = pm.Uniform('beta', lower=0, upper=1e13)
            # alpha = pm.HalfFlat('alpha')
            alpha = pm.Deterministic('alpha', beta*p_bb/(1-p_bb))

            obs = pm.BetaBinomial('obs', alpha=alpha, beta=beta, n=DP_data, observed=AO_data)
            map = pm.find_MAP(progressbar=False)

        alpha_list.append(map['alpha'])
        beta_list.append(map['beta'])

    # Extract priors
    prior_betabinom_alpha = extract_kde(np.array(alpha_list))
    prior_betabinom_beta = extract_kde(np.array(beta_list))
    # prior_betabinom_p = extract_kde(np.array(p_bb_list))

    if cohort is not None:
        if not os.path.exists(f'../exports/{cohort}/'):
            os.makedirs(f'../exports/{cohort}/')
    
        with open(f'../exports/{cohort}/prior_binom_p.npy', 'wb') as f:
            np.save(f, prior_binom_p)
        with open(f'../exports/{cohort}/prior_betabinom_beta.npy', 'wb') as f:
            np.save(f, prior_betabinom_beta)

        # with open(f'../exports/{cohort}/prior_betabinom_p.npy', 'wb') as f:
        #     np.save(f, prior_betabinom_p)

        with open(f'../exports/{cohort}/prior_betabinom_alpha.npy', 'wb') as f:
            np.save(f, prior_betabinom_alpha)

    return prior_binom_p, prior_betabinom_alpha, prior_betabinom_beta
# %%



