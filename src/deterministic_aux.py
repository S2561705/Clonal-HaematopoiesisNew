import pymc as pm
import arviz as az
import pickle as pk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def deterministic_fit(part, return_trace=False, save_output=False):
    # Set up a PyMC model for Bayesian inference
    # extract clonal_structure of opitmal model
    cs = part.uns['optimal_model']['clonal_structure']

    N = 100_000  # Total population size

    # center prior for init age 
    # based on mean VAF for each mutation
    # and logistic growth with 
    # leading clonal term on the denominator 
    
    # Compute mean age and vaf for each mutation
    mean_age = part.var.time_points.mean()
    mean_vaf = part.X.mean(axis=1)

    # extract leading term for clonal size of denominator
    max_vaf = mean_vaf.max()

    # invert deterministic logistic growth to solve for init age
    mean_init = mean_age- np.log(2*mean_vaf*(N+max_vaf))/np.array(part.obs.fitness)
    part.obs['init_ages_guess'] = mean_init

    # Use bayesian inference to find age of mutation acquisition
    # note that age can be negative to reflect growth during developmental stages 
    with pm.Model() as model:

        # Establish prior for initial ages 
        # based on inverted logistic growth
        init = pm.TruncatedNormal('init_age', 
            mu = mean_init,
            sigma = 50*np.ones(part.obs.shape[0]), 
            lower= mean_init - 250,
            upper=mean_init + 250,
            shape=part.shape[0])

        # Compute time from mutation to observation
        t_from_init = np.array(part.var.time_points) - init[:, None]
        
        # Calculate deterministic evolution of clone sizes
        fitness = np.array(part.obs.fitness)[:, None]
        sizes = np.exp(t_from_init * fitness)

        # clip for negative sizes resulting 
        sizes = pm.math.where(sizes<0, 0,  sizes)

        # clone = cs[0]
        # max_within_clone = np.argmax(sizes[clone].sum(axis=1))
        # clone_size = []
        max_ids = []
        for i, clone in enumerate(cs):
            max_within_clone = np.argmax(sizes[clone].sum(axis=1))
            max_ids.append(clone[max_within_clone])
            # max_size = np.array(sizes[clone]).max(axis=0)
            # clone_size.append(max_size)

        leading_sizes = sizes[max_ids, :]

        # Calculate deterministic size of Variant Allele Frequency (VAF)
        vaf_sizes = sizes / (2*(N + leading_sizes.sum(axis=0)))

        # Define the observed data using a Binomial distribution
        obs = pm.Binomial('observations',
                        n=part.layers['DP'], p=vaf_sizes,
                        observed=part.layers['AO'])
        
        # Sample from the posterior distribution
        trace = pm.sample(progressbar=False, cores=1)

    part.obs[['init_age', 'init_age_hdi_3%', 'init_age_hdi97%']] = az.summary(trace)[['mean', 'hdi_3%', 'hdi_97%']].to_numpy()

    if save_output is True:
        id = part.uns['participant_id']
        with open(f'../results/deterministic_fits/{id}.pk', 'wb') as f:
            pk.dump(part, f)

    if return_trace is True:
        return trace

    return part    

def order_of_mutations(part):
    """
    Assign overall and within-clonal order to mutations in an AnnData object.

    This function adds two new columns to the obs DataFrame of the input AnnData object:
    'overall_order' and 'within_clonal_order'. The ordering is based on the 'init_age' of
    mutations and takes into account the highest density interval (HDI) confidence intervals.

    Parameters:
    -----------
    part : AnnData
        An AnnData object containing mutation data. The obs DataFrame should have the following columns:
        - 'init_age': The initial age of the mutation
        - 'hdi_3%': The lower bound of the HDI confidence interval
        - 'hdi97%': The upper bound of the HDI confidence interval
        - 'cohort': The cohort to which the mutation belongs
        - 'clonal_structure': The clonal structure to which the mutation belongs

    Returns:
    --------
    AnnData
        The input AnnData object with two new columns added to its obs DataFrame:
        - 'overall_order': The order of mutations across all cohorts
        - 'within_clonal_order': The order of mutations within each clonal structure

    Notes:
    ------
    - Mutations with overlapping HDI intervals are assigned the same order.
    - The overall order is calculated within each cohort.
    - The within-clonal order is calculated for each unique clonal structure.
    - The original AnnData object is modified in-place.
    """
  
    if part.shape[0] == 1:
        part.obs['overall_order'] = 1
        part.obs['within_clone_order'] = 1

        return part
        
    # sort Anndata by init_age
    part = part[part.obs.sort_values('init_age').index].copy()
    
    # create a copy of participant observations 
    # to avoid modifying original

    # Function to assign overall order considering HDI
    def assign_order(group):
        order = 0
        previous_hdi_97 = float('-inf')
        orders = []
        for _, row in group.iterrows():
            if row['init_age'] > previous_hdi_97:
                order += 1
            orders.append(order)
            previous_hdi_97 = max(previous_hdi_97, row['init_age_hdi97%'])
        return pd.Series(orders, index=group.index)

    # Apply the function to assign overall order
    overall_order = part.obs.groupby('cohort', group_keys=False).apply(assign_order)
    overall_order_dict = overall_order.iloc[0].to_dict()
    part.obs['overall_order'] = part.obs.index.map(overall_order_dict)

    within_clone_order_dict = {}
    for clonal_index in part.obs.clonal_index.unique():
        idx = part.obs[part.obs.clonal_index == clonal_index].index

        # Group by 'clonal_structure' and apply the function
        within_clone_order = part[idx].obs.groupby('clonal_index').apply(assign_order).reset_index(level=0, drop=True)
        within_clone_order_dict.update(within_clone_order.iloc[0].to_dict())

    part.obs['within_clone_order'] = part.obs.index.map(within_clone_order_dict)
    return part

def plot_deterministic(part, lines=False):

    # Create a deterministic plot of the model results
    N = 100_000
    cs = part.uns['optimal_model']['clonal_structure']

    init = part.obs.init_age.to_numpy()
    time_points = np.linspace(part.var.time_points.min()-5, part.var.time_points.max()+5)

    # Compute time from mutation to observation
    t_from_init = np.array(time_points) - init[:, None]

    # Calculate deterministic evolution of clone sizes
    fitness = np.array(part.obs.fitness)[:, None]
    sizes = np.exp(t_from_init * fitness)

    sizes = np.where(sizes==0, 0, sizes)
    max_ids = []
    for i, clone in enumerate(cs):
        max_within_clone = np.argmax(sizes[clone].sum(axis=1))
        max_ids.append(clone[max_within_clone])
        # max_size = np.array(sizes[clone]).max(axis=0)
        # clone_size.append(max_size)

    leading_sizes = sizes[max_ids, :]

    # Calculate deterministic size of Variant Allele Frequency (VAF)
    vaf_sizes = sizes / (2*(N + leading_sizes.sum(axis=0)))

    # Plot the results
    for i, size in enumerate(vaf_sizes):
        if lines is True:
            sns.lineplot(x=part[i].var.time_points, y=part[i].X.flatten(), color=sns.color_palette()[i],  label=part[i].obs.p_key.values[0])

        elif lines is False:
            sns.scatterplot(x=part[i].var.time_points, y=part[i].X.flatten(), color=sns.color_palette()[i],  label=part[i].obs.p_key.values[0])
        
        # add deterministic line
        sns.lineplot(x=time_points, y=size, color=sns.color_palette()[i], linestyle='--')
        

def combined_posterior_plot(part):

    # fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5, 7), gridspec_kw={'height_ratios': [2, 1], 'hspace':0.3})

    # Create a deterministic plot of the model results
    N = 100_000
    cs = part.uns['optimal_model']['clonal_structure']

    init = part.obs.init_age.to_numpy()
    time_points = np.linspace(part.var.time_points.min()-5, part.var.time_points.max()+5)

    # Compute time from mutation to observation
    t_from_init = np.array(time_points) - init[:, None]
    clipped_ages = np.where(part.obs.init_age.values<0, 0 , np.floor(part.obs.init_age))

    # Calculate deterministic evolution of clone sizes
    fitness = np.array(part.obs.fitness)[:, None]
    sizes = np.exp(t_from_init * fitness)

    sizes = np.where(sizes==0, 0, sizes)
    max_ids = []
    for i, clone in enumerate(cs):
        max_within_clone = np.argmax(sizes[clone].sum(axis=1))
        max_ids.append(clone[max_within_clone])
        # max_size = np.array(sizes[clone]).max(axis=0)
        # clone_size.append(max_size)

    leading_sizes = sizes[max_ids, :]

    # Calculate deterministic size of Variant Allele Frequency (VAF)
    vaf_sizes = sizes / (2*(N + leading_sizes.sum(axis=0)))

    # Plot the results
    for i, size in enumerate(vaf_sizes):
        label = part[i].obs.p_key.values[0]
        if type(label) != str:
            label = part[i].obs.PreferredSymbol.values[0] + ' Splicing'

        color = sns.color_palette()[i]
        # add data points
        sns.scatterplot(x=part[i].var.time_points, y=part[i].X.flatten(), color=color,  label=label, ax=ax1)
        
        # add deterministic line
        sns.lineplot(x=time_points, y=size, color=color, linestyle='--', ax=ax1)
        # Add clipped age text at the beginning of the line
        first_point_x = time_points[15]
        first_point_y = size[0]
        ax1.text(first_point_x, first_point_y, f'ATMA: {clipped_ages[i]:.0f}', color=color, 
                ha='right', va='bottom', fontweight='bold', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', pad=1))
    if part.uns['warning'] is not None:
        print('WARNING: ' + part.uns['warning'])
    
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    s_range = model['s_range']

    # normalisation constant
    norm_max = np.max(output, axis=0)

    # Plot
    for i in range(len(cs)):
        first_mut_in_cs_idx = cs[i][0]
        p_key_str = f''
        for k, j in enumerate(cs[i]):
            label = part[j].obs.p_key.values[0]
            if type(label) != str:
                label = part[j].obs.PreferredSymbol.values[0] + ' Splicing'
            if k == 0:
                p_key_str += f'{label}'
            if k > 0:
                p_key_str += f'\n{label}'

        color = sns.color_palette()[first_mut_in_cs_idx]
        sns.lineplot(x=s_range,
                    y=output[:, i]/ norm_max[i],
                    label=p_key_str,
                    color=color,
                    ax=ax2)

                # Fill the area under the line
        ax2.fill_between(s_range, 
                         output[:, i]/ norm_max[i], 
                         alpha=0.3,  # Adjust alpha for transparency
                         color=color)

    sns.despine()  
    ax1.title.set_text('Clonal dynamics inference')
    ax2.title.set_text('Clonal Fitness')
    ax1.set_xlabel('Age (yrs)')
    ax1.set_ylabel('VAF')
    ax2.set_xlabel('Fitness')
    ax2.set_ylabel('Normalised likelihood')
    # ax2.set_xscale('log')
    # ax2.legend(loc='upper right', bbox_to_anchor=(1.7,1))
    
