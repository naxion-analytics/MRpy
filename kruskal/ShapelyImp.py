import numpy as np
from numba import jit
from math import factorial
from .utils import *

__all__ = ['Nvars','R2','ShapelyImp']

@jit('i8(i8)')
def NVars(n):
    ### count number of cols are used in model n
    count=0
    while n>0:
        count+=1
        n = n&(n-1)
    return count

@jit('f8(f8[:,:],i8,i8)')
def R2(S, i, n):
    covariates = np.array([k+1 for k in range(n) if i &(1<<k)!=0])
    Sxx_inv = np.linalg.inv( S[covariates,covariates] )
    Syx = S[0,covariates]; Sxy = S[covariates,0]
    return Syx.dot(Sxx_inv.dot(Sxy) )/S[0,0]

@jit('f8[:](f8[:],f8[:,:],f8[:,:])')
def ShapelyImp( Y, X, weights ):
    
    ### weighted covariance matrix - corrects scipy version
    w = np.diagflat( weights/np.sum(weights),k=0)
    X = np.sqrt(w).dot(X - np.sum( w.dot(X),axis=0))
    S = 1./(1. - np.sum(w**2) ) * X.T.dot(X)

    n = S.shape[1]
    model = np.zeros(2**n)
    kruskal = np.zeros(n)
    
    model[0] = S[0,0]  ### no covariates in model, base case
    for i in xrange(2**n):
        if i%(2**n/2**10) == 0: print '%d combinations computed'%i

        k = NVars(i)
        if model[i] == 0: model[i] = R2(S,i,n)
        
        for ij in xrange(n):
            j = (1<<ij)
            if i == i|j: continue
            if model[i|j] == 0:  model[i|j] = R2(S,i|j,n) 
            
            kruskal[ ij ] += factorial(k) * 1.*factorial(n - k -1)/factorial(n)* (model[i]-model[i|j])/model[0]
            
    return kruskal