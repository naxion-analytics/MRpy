from numba import jit
import numpy as np


def log_likelihood(theta,x,y):
    alpha, beta, sigma = theta
    model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma*2) + (y - model)**2/(sigma**2) )

def log_prior(theta):
    alpha, beta, sigma = theta
    return -np.log(theta) - 1.5*np.log(1+beta**2)

def log_posterior(theta,x,y):
    return log_prior(theta) + log_likelihood(theta,x,y)
