import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as P


def E_step(data, phi, mu, sigma):
    """
       Perform E-step on GMM model
       Each datapoint will find its corresponding best cluster center
       ---
       Inputs:
            data: (n, d) array, where n is # of data points and d is # of dimensions
            phi: (k, d) array, where k is # of clusters
            mu: (k, d) array, where k is # of clusters and d is # of dimensions
            sigma: (k, d, d) array, where k is # of clusters and d is # of dimensions

       Returns:
            'w': (k,n) array indicating the cluster of each data point
                        where k is # of clusters and n is # of data points
       """
    n = len(data)
    k = len(phi)
    w = np.zeros(k, n)
    for i in range(n):
        norm_i = 0
        for j in range(k):
            w[j, i] = P(mu[j], sigma[j]).pdf(data[i]) * phi[j]
            norm_i += w[j, i]
        w[:, i] /= norm_i
    return w


def M_step(data, w, phi, mu, sigma):
    """
    M-step: Update our estimate of μ, σ2 and using the new values of the latent variables z.
    Inputs:
        clusters: (n) array indicating the cluster of each data point
        data: (n, d) array, where n is # of data points and d is # of dimensions
        mu: (k, d) array, where k is # of clusters and d is # of dimensions
        sigma: (k, d, d) array, where k is # of clusters and d is # of dimensions

    Returns:
        mu: (k, d) array, where k is # of clusters and d is # of dimensions
        sigma: (k, d, d) array, where k is # of clusters and d is # of dimensions
    """
    sum_of_all_w = np.sum(w)
    # iterate over each gaussian, calculate μ, σ2:

    for j in range(len(mu)):
        sum_over_wj = np.sum(w[j])
        # μ <- (1/sum over wj) * weighted sum over all the data points
        mu[j] = np.sum(np.array([xi * w[j, i] for i, xi in enumerate(data)]), axis=0) / sum_over_wj
        # σ2 <- wheighted sum over all the data points in cluster((datapoint value - μ_new)**2)
        sigma[j] = np.sum(np.array([np.outer((xi - mu[j]).T, xi - mu[j]) * w[j, i] for i, xi in enumerate(data)]), axis=0)/sum_over_wj

        phi[j] = sum_over_wj/sum_of_all_w

    return phi, mu, sigma

# Input: data, model

def EM(data, initial_model):
    # 1. Initialize model parameters
    phi = initial_model[0]
    mu = initial_model[1]
    sigma = initial_model[2]

    # 2. while not converged:
    converged = False
    i = 0
    while not converged and i<5: #TODO: ADD CONVERGENCE CONDITIONS

        # 2.1     E-step: compute expected value of latent variable given current parameters
        w = E_step(data, phi, mu, sigma)

        # 4.     M-step: update parameters to maximize expected likelihood
        phi, mu, sigma = M_step(data, w, phi, mu, sigma)

        # Plot:
        plt.scatter(data[:, 0], data[:, 1], c=1 + w[0] - w[1], alpha=0.1, cmap='RdYlBu')
        plt.show()

        i += 1
    # 5. return model
    return phi, mu, sigma

mu = np.array([[0,0],[2,2]])
sigma = np.array([[[1, 0], [0, 1]], [[1, 0],[0, 1] ] ])
phi = [0.7,0.3]
initial_model = [phi,mu,sigma]

EM(data,initial_model)

