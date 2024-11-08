import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats

prior_resolution = 100
def extract_kde(data, quantiles=[0.01, 0.99], resolution=prior_resolution,
                flat_left_tail=False):
    """Create kde profile from data.
    Parameters:
    -----------
    data: List. List of data for which we want to extract its kde profile.
    quantiles: list. List of low and top quantiles range for kdee evaluation.

    Returns:
    kde_profil: Array. 2-d array containing kde profile.
    -----------

    """
    kernel_dist = gaussian_kde(data)

    # Extract support of distribution
    kernel_support = np.quantile(data, q=quantiles)

    # Sample uniformly from the range of binomial probabilities
    kernel_sample = np.linspace(kernel_support[0],
                                kernel_support[1],
                                resolution)

    # Compute associated prior probabilities using kde
    kernel_values = kernel_dist.evaluate(kernel_sample)

    # Flatten left tail of distribution
    if flat_left_tail is True:
        max_index = np.argmax(kernel_values)
        kernel_values[:max_index] = np.max(kernel_values)

    # normalise kernel to unit integral
    kernel_values = kernel_values / np.trapz(x=kernel_sample, y=kernel_values)

    return np.array([kernel_sample, kernel_values])

def normalise_column (data):
    return (data - np.mean(data))/np.std(data)


def find_best_transformation(data):
    transformations = {
        'log': np.log1p,
        'sqrt': np.sqrt,
        'boxcox': lambda x: stats.boxcox(x + 1)[0],
        'yeojohnson': stats.yeojohnson,
        'no_transform': np.array
    }

    flat_data = data.ravel()
    mask = ~np.isnan(flat_data)
    non_nan_data = flat_data[mask]
    
    # make data >1 by simply translating 
    # this doesn't affect the shape of values, so fine for converting into z-scores
    non_nan_data = non_nan_data + 1 + -min(np.min(non_nan_data), 0)

    best_score = -np.inf 
    best_transform = None

    for name, func in transformations.items():
        try:
            transformed = func(non_nan_data)
            _, p_value = stats.normaltest(transformed)
            if p_value > best_score:
                best_score = p_value
                best_transform = name
        except:
            continue

    # tranform data using optimal transformation
    best_transform_func = transformations[best_transform]
    flat_data[mask] = best_transform_func(non_nan_data)
    transformed_data = flat_data.reshape(data.shape)

    return transformed_data, best_transform, best_score


def z_score(data):
    """
    Calculate z-scores for input data.

    Args:
    data (np.array): Input data

    Returns:
    np.array: Z-scored data
    """
    mean, std = np.nanmean(data), np.nanstd(data)
    return (data - mean) / std

def normalise_parameter(df, parameter, return_transform=False):
    """
    Normalize and convert markers to z-scores.

    Args:
    data (pandas): Input Pandas object
    parameter (str): Name of the observation to normalize

    Returns:
    None: Updates the data object in-place
    """
    data = df[parameter].copy()
    
    transformed_data, _, _ = find_best_transformation(data)
    z_data = z_score(transformed_data)
    df[parameter+'_z_score'] = z_data

    if return_transform is True:
        return find_best_transformation
