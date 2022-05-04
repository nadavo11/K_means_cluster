import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as P
import matplotlib.pyplot as plt

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
    w = np.zeros((k, n))
    log_likelyhood = 0
    for i in range(n):
        norm_i = 0
        for j in range(k):
            w[j, i] = P(mu[j], sigma[j]).pdf(data[i]) * phi[j]
            norm_i += w[j, i]

        w[:, i] /= norm_i                   # normalize w
        log_likelyhood -= np.log(norm_i)    # compute log-likelyhood
    return w, log_likelyhood


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
    k = len(phi)
    # 2. while not converged:
    converged = False
    i = 0
    iteration_log_likelihood = [0.0]
    while not converged and i<15: #TODO: ADD CONVERGENCE CONDITIONS

        # 2.1     E-step: compute expected value of latent variable given current parameters
        w,ll = E_step(data, phi, mu, sigma)
        iteration_log_likelihood.append(ll)

        # 4.     M-step: update parameters to maximize expected likelihood
        phi, mu, sigma = M_step(data, w, phi, mu, sigma)

        # Plot:
        plt.scatter(data[:, 0], data[:, 1], c=1 + w[0] - w[1], alpha=0.1, cmap='RdYlBu')
        plt.show()

        i += 1
        converged = (iteration_log_likelihood[i]-iteration_log_likelihood[i-1] < 0.001)
    # 5. return model
    return phi, mu, sigma, iteration_log_likelihood

def log_likelihood(phi, mu, sigma, data):
    ll = 0
    for i in data:
        ll += P(mu[j], sigma[j]).pdf(data[i]) * phi[j]

def initial_model(k,d):
    """

    :param k: # of Gaussians in model
    :param d: # of dimentions of datapoints
    :return: an arbitrary initial condition  model for EM. containing:
             phi: amplitude of Gaussians, np array (k)
             mu: np array(k,d)
             sigma: covariance matrix of gausians, np aray (k,d,d)


    """
    phi = np.random.rand(k)
    phi /= sum(phi)

    mu = np.random.multivariate_normal([0]*d, np.eye(d))

    sigma = np.random.rand(d, d)
    sigma = np.dot(sigma, sigma.T)
    return phi, mu, sigma
