import sys
sys.path.append("..") 
from src.prior_extraction import *
from src.general_imports import *
from src.clonal_inference import *

import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import betabinom, binom
from jax import vmap

import scipy.stats as stats

from scipy.optimize import minimize
# from scipy.stats import binom, betabinom, nbinom
from multiprocessing import Pool


# Define mutation_class
class mutation_class():
    """Mutation class object, used for LiFT filtering"""
    def __init__(self, key, data, artefact_prob=None, mut_prob=None, type=None):
        self.key = key
        self.data = data
        self.artefact_prob = artefact_prob
        self.mut_prob = mut_prob
        self.type = type

def create_mut_dict (cohort):
    """create dictionary of mutations in cohort
    - dictionary key is a mutation key.
    - dictionary value is a list of AnnData dataframes
      with a single row corresponding to an instance of
      a mutation in a participant
    """ 
    mut_dict = dict()
    for part in cohort:
        for mut in part:
            key = mut.obs.index[0]
            if key not in mut_dict.keys():
                mut_dict[key] = [mut]
            else:
                mut_dict[key].append(mut)

    return mut_dict

##################
#region: Binomial and Betabinomial prior fitting
##################

def binom_artifact_nll(params, mut_data_list):
    """Compute negative log likelihood of a list of time-series data
    assuming the data was produced by a binomial distribution conditional
    on a given binomial probability.

    Parameters
    -----------
    params: tuple (float, float). Tuple of floats determining the conditional
    binomial probability.

    mut_data_list: List of AnnData objects with time-series observations.
                   Each object must have layers 'AO' and 'DP' relating 
                   to allele observations and read depth.

    Returns
    -----------
    nll: float. negative log-likelihood of observing a given data.
     """
    # Extract parameters
    p = params[0]

    # Flatten all observations
    AO = jnp.hstack([data.layers['AO'].flatten()[0:]
                     for data in mut_data_list])
    DP = jnp.hstack([data.layers['DP'].flatten()[0:]
                     for data in mut_data_list])
    
    likelihood = -stats.binom.logpmf(k=AO, n=DP, p=p).sum()

    return likelihood


def binom_artifact_fit(mut_data_list):
    """Compute the minimium negative log-likelihood of observing a time-series
    data, assuming the data was produced by gaussian noise.

    Parameters
    -----------
    trajectories: list of Pandas DataFrames. Each dataset must includes columns
                  'AO' and 'DP'.
    params: tuple (float, float). Tuple of floats determining the mean and
            standard deviation of the gaussian distribution used to initiate
            fitting proccess.
            By default this is set to mean: 0.01 and std_dev: 0.01.

    Returns
    -----------
    model: tuple (scipy.OptimizeResult, float). Optimized fit of a gaussian
           distribution to describe the time-series data. Bayesian information
           criterion.
     """

    # initial parameters
    p = np.array(0.01)
    bounds = [(0,0.05)]
    model = minimize(binom_artifact_nll, x0=p, bounds=bounds,
                     args=mut_data_list, method='Nelder-Mead')

    return model

def betabinom_artifact_nll(params, mut_data_list):
    """Compute negative log likelihood of a list of time-series data
    assuming the data was produced by a betabinomial distribution conditional
    on a given mean and beta value.

    Parameters
    -----------
    params: tuple (float, float). Tuple of floats determining the conditional
    betabinomial mean and beta.

    mut_data_list: List of AnnData objects with time-series observations.
                   Each object must have layers 'AO' and 'DP' relating 
                   to allele observations and read depth.

    Returns
    -----------
    nll: float. negative log-likelihood of observing a given data.
     """
    # Extract parameters
    p = params[0]
    beta = params[1]
    alpha = beta*p/(1-p)


    # Flatten all observations
    AO = jnp.hstack([data.layers['AO'].flatten() for data in mut_data_list])
    DP = jnp.hstack([data.layers['DP'].flatten() for data in mut_data_list])
    
    likelihood = -stats.betabinom.logpmf(k=AO, n=DP, a=alpha, b=beta).sum()

    return likelihood

def betabinom_artifact_fit(mut_data_list, x0=False):
    """Find the optimal fit to a list of time-series data
     assuming a betabinomial model

    Parameters
    -----------
    mut_data_list: List of AnnData objects with time-series observations.
                   Each object must have layers 'AO' and 'DP' relating 
                   to allele observations and read depth.
    x0: array with initialization of x0 parameters p and beta

    Returns
    -----------
    model: scipy.optimize.minimize model object.
     """

    if x0 is False:
        # initialise parameters
        p = 0.01
        beta = 1e5
        x0 = np.array([p, beta])
    
    bounds = ((0, 0.05), (0,1e13))

    model = minimize(betabinom_artifact_nll, x0=x0, bounds=bounds,
                     args=mut_data_list, method='Nelder-Mead')

    return model

def repeat_betabinom(data_list, n_fits=10):

    # create different random initializations for p and beta
    p_init = np.random.uniform(0, 0.05, size=n_fits)
    beta_log_init = np.random.uniform(low=0, high=7, size=n_fits)
    beta_init = np.power(10, beta_log_init)

    x0 = np.vstack([p_init, beta_init]).T

    """Repeat model fitting """
    models = [betabinom_artifact_fit(data_list, x0[i]) for i in range(n_fits)]
    opt_model = min(models, key=lambda m: m.fun)
    return opt_model

def fit_artefact_priors(mut_dict):
    """Retrieve artefact parameter priors from mutation dictionaries"""
    # fit separately singly occuring mutations vs recurring mutations
    single_occurence = [data_list for data_list in mut_dict.values() 
                        if len(data_list)== 1]

    multiple_occurence = [data_list for data_list in mut_dict.values() 
                        if len(data_list)>1]

    # initialise parameter lists
    binom_p_list = []
    betabinom_p_list = []
    # betabinom_alpha_list = []
    betabinom_beta_list = []
    for data_list in tqdm(single_occurence):
        binom_p_list.append(binom_artifact_fit(data_list).x[0])

    with Pool(8) as p:
        multiple_occurence_models = list(tqdm(
            p.imap(repeat_betabinom, multiple_occurence),
            total=len(multiple_occurence)))

    for opt_model in multiple_occurence_models:
        if opt_model.message == 'Optimization terminated successfully.':
            betabinom_p_list.append(opt_model.x[0])
            # betabinom_alpha_list.append(opt_model.x[0])
            betabinom_beta_list.append(opt_model.x[1])

    prior_binom_p = extract_kde(np.array(binom_p_list))
    prior_betabinom_p = extract_kde(np.array(betabinom_p_list))
    prior_betabinom_beta = extract_kde(np.array(betabinom_beta_list))

    return prior_binom_p, prior_betabinom_p, prior_betabinom_beta

# artifact model probability
def artefact_probability_p (mut, prior_list, init_point=1):
    """Compute probability of data associated with a 
    mutation being the result of a statistical artefact."""
   
    prior_binom_p, prior_betabinom_p, prior_betabinom_beta= prior_list

    # Extract data for each occurrence of mutation
    mut_data_list = mut.data

    # Flatten all observations
    AO = jnp.hstack([data.layers['AO'].flatten()[init_point:] for data in mut_data_list])
    DP = jnp.hstack([data.layers['DP'].flatten()[init_point:] for data in mut_data_list])

    # If singly occurring assume binomial model
    if len(mut_data_list) == 1:

        prob = binom.pmf(AO[:, None], p=prior_binom_p[0], n=DP[:, None])
        model_prob = np.trapz(y=prob.prod(axis=0)*prior_binom_p[1],
                              x=prior_binom_p[0])

    # If multiple observations assume beta binomial model
    elif len(mut_data_list)>1:
        marginal_p = []
        for p in prior_betabinom_p[0]:
            alpha = p*prior_betabinom_beta[0]/(1-p)
            p_prob = betabinom.pmf(AO[:, None], a=alpha, b=prior_betabinom_beta[0], n=DP[:, None])
            marginal_p.append(
                np.trapz(y=p_prob.prod(axis=0)*prior_betabinom_beta[1],
                        x=prior_betabinom_beta[0])
            )

        model_prob = np.trapz(y=np.array(marginal_p)*prior_betabinom_p[1],
                              x=prior_betabinom_p[0])
        
    # Save artefact probability in class object
    mut.artefact_prob = model_prob



# @jit
# def betabinom_cond_alpha_beta(beta, alpha, AO, DP):
#     return betabinom.pmf(k=AO, n=DP, a=alpha, b=beta).prod()

# @jit
# def betabinom_alpha_cond (alpha, AO, DP):
#     alpha_beta_cond_prob = vmap(betabinom_cond_alpha_beta, (0, None, None, None))(prior_betabinom_beta[0], alpha, AO, DP)
    
#     # marginalise beta parameter 
#     alpha_cond_prob = trapezoid(
#         y=alpha_beta_cond_prob*prior_betabinom_beta[1],
#         x=prior_betabinom_beta[0])
    
#     return alpha_cond_prob

# @jit
# def binom_p_cond (p, AO, DP):
#     return binom.pmf(k=AO, n=DP, p=p).prod()

# def artefact_probability (mut):

#     mut_data_list = mut.data

#     AO = jnp.hstack([data.layers['AO'].flatten() for data in mut_data_list])
#     DP = jnp.hstack([data.layers['DP'].flatten() for data in mut_data_list])

#     if len(mut_data_list) ==1:
#         # compute conditional probability of bionmial model
#         binom_cond_prob = vmap(binom_p_cond, (0, None, None))(prior_binom_p[0], AO ,DP)
        
#         # marginalise p to obtain model prob
#         model_prob = trapezoid(
#             y=binom_cond_prob*prior_binom_p[1],
#             x=prior_binom_p[0])

#     else:
#         # compute alpha conditional probability
#         alpha_cond_prob = vmap(betabinom_alpha_cond, (0, None, None))(prior_betabinom_alpha[0], AO ,DP)
        
#         # marginalise alpha to obtain model probability
#         model_prob = trapezoid(
#             y=alpha_cond_prob*prior_betabinom_alpha[1],
#             x=prior_betabinom_alpha[0])
    
#     mut.artefact_prob = model_prob


def real_mut_probability(mut, s_resolution=51, min_s=0.05, max_s=0.5, disable_progressbar=False):
    """Compute probability of list of mutations associated with being real,
    each with their own fitness"""

    # Unpack data for each mutation occurrence
    mut_data_list = mut.data
    
    # Compute probability of single clonal structure
    updated_mut_data_list = [compute_clonal_models_prob_vec(m, s_resolution=s_resolution, min_s=min_s, max_s=max_s, disable_progressbar=disable_progressbar)
                            for m in mut_data_list]

    # save model probability
    m_prob = np.prod([m.uns['model_dict']['model_0'][1] for m in updated_mut_data_list])

    # save information in mutation class object
    mut.mut_prob = m_prob


def update_cohort_LiFT(cohort, mut_obj_list):
    artefact_prob_dict = dict([(mut.key, mut.artefact_prob) for mut in mut_obj_list])
    mut_prob_dict = dict([(mut.key, mut.mut_prob) for mut in mut_obj_list])

    for part in cohort:
        part.obs['artefact_prob'] = part.obs.index.map(artefact_prob_dict)
        part.obs['mutation_prob'] = part.obs.index.map(mut_prob_dict)
        part.obs['LiFT_value'] = part.obs['mutation_prob']/part.obs['artefact_prob']

    return cohort