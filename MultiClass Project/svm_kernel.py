import numpy as np

# Regularized Fourier (stronger mode regularization)
def regularized_fourier_kernel(x, y, gamma=0.9, degree=2):
    """
    Compute the regularized Fourier kernel between two samples x and y.

    Args:
        x: The first sample, shape (n_features,).
        y: The second sample, shape (n_features,).
        gamma: The gamma parameter of the kernel.
        degree: The degree of the kernel.

    Returns:
        The kernel between x and y.
    """
    
    return np.exp(-gamma * np.sum((x - y)**2)) * np.cos(gamma * np.sum((x - y)**2))**degree
