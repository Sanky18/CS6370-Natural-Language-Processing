import numpy as np
from scipy.special import stdtr

def hypothesis_test(sample1, sample2):
    """
    Perform a two-sample t-test to compare the means of two samples.

    Parameters:
    - sample1: List or array-like, values of sample 1
    - sample2: List or array-like, values of sample 2

    Returns:
    - t_value: float, t-value of the hypothesis test
    - p_value: float, p-value of the hypothesis test
    """
    # Convert samples to numpy arrays for computation
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    # Compute mean and variance of sample 1
    mean_sample1 = sample1.mean()
    var_sample1 = sample1.var(ddof=1)  # ddof=1 for unbiased estimator of variance
    size_sample1 = sample1.size
    dof_sample1 = size_sample1 - 1

    # Compute mean and variance of sample 2
    mean_sample2 = sample2.mean()
    var_sample2 = sample2.var(ddof=1)  # ddof=1 for unbiased estimator of variance
    size_sample2 = sample2.size
    dof_sample2 = size_sample2 - 1

    # Calculate t-value using the formula for two-sample t-test
    t_value = (mean_sample1 - mean_sample2) / np.sqrt(var_sample1/size_sample1 + var_sample2/size_sample2)

    # Calculate degrees of freedom using Welch's approximation
    dof = (var_sample1/size_sample1 + var_sample2/size_sample2)**2 / \
          (var_sample1**2/(size_sample1**2*dof_sample1) + var_sample2**2/(size_sample2**2*dof_sample2))

    # Calculate p-value using Student's t distribution
    p_value = 2 * stdtr(dof, -np.abs(t_value))

    # Print the results
    print("t-value:", t_value)
    print("p-value:", p_value)
