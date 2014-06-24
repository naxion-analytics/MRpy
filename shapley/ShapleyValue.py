import numpy as np
from numba import jit
from math import factorial

__all__ = ['NVars','R2','ShapelyImp']

#@jit('i8(i8)')
def NVars(n):
    ### count number of cols are used in model n
    count=0
    while n>0:
        count+=1
        n = n&(n-1)
    return count

#@jit('f8(f8[:,:],i8,i8)')
def R2(S, i, n):
    
    cov = np.zeros(n+1,dtype=bool)
    for k in range(n):
        if i&(1<<k)!=0: cov[k+1] = True
    
    Sxx = S[:,cov]
    Sxx_inv = np.linalg.inv( Sxx[cov,:] )
    Syx = S[0,cov]; Sxy = S[cov,0]
    return Syx.dot(Sxx_inv.dot(Sxy) )/S[0,0]

#@jit('f8[:](f8[:,:])')
def ShapleyValue( S ):
    n_cov = S.shape[1]-1
    model = np.zeros(2**n_cov)
    shapley =np.zeros(n_cov)
    
    model[0] = S[0,0]  ### no covariates in model, base case
    for i in xrange(2**n_cov):
        
        k = NVars(i)
        if model[i] == 0: model[i] = R2(S,i,n_cov)
        
        for ij in xrange(n_cov):
            j = (1<<ij)
            if i == i|j: continue
                
            if model[i|j] == 0:  model[i|j] = R2(S,i|j,n_cov) 
            shapley[ ij ] += factorial(k) * 1.*factorial(n_cov - k -1)/factorial(n_cov)* (model[i]-model[i|j])/model[0]
            
    return shapley




    