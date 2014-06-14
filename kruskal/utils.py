import numpy as np
from numba import jit

__all__ = ['SSE','SS_tot','NVars','ColIndices']


@jit('f8( f8[:,:],f8[:],f8[:,:])')
def SSE(X,Y,weights):
    ### compute the weighted sum of squared errors
    X = np.append(np.ones((X.shape[0],1)), X,axis=1)
    Beta = np.linalg.inv(X.transpose().dot(weights.dot(X))).dot(X.transpose().dot(weights.dot(Y)))
    return ( Y-X.dot(Beta) ).dot(weights.dot(Y-X.dot(Beta)).transpose())


@jit('f8( f8[:,:],f8[:],f8[:,:])')
def SS_tot(X,Y,weights):
    ### model with no covariates
    X = np.ones((X.shape[0],1))
    Beta = np.linalg.inv(X.transpose().dot(weights.dot(X))).dot(X.transpose().dot(weights.dot(Y)))
    return ( Y-X.dot(Beta) ).dot(weights.dot(Y-X.dot(Beta)).transpose())


@jit('i8(i8)')
def NVars(n):
    ### count number of cols are used in model n
    count=0
    while n>0:
        count+=1
        n = n&(n-1)
    return count


@jit('i8[:](i8,i8)')
def ColIndices(n, N):
    ### convert bit string to col indices
    return np.array([k for k in range(N) if n & (1<<k)!= 0],dtype=int)