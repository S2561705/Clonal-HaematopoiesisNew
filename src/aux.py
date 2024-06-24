import numpy as np
from scipy.stats import gaussian_kde

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
