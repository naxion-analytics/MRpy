import numpy as np
from numba import jit
from math import factorial
from .utils import *

__all__ = ['ShapelyImp']

@jit('f8[:](f8[:],f8[:,:],f8[:,:])')
def ShapelyImp( Y, X, weights):
    
    n = X.shape[1]
    model = np.zeros(2**n)
    Imps =  np.zeros(n)
    
    model[0] = SS_tot(X,Y,weights)  ### no covariates in model, base case
    for i in xrange(2**n):
    	k = NVars(i)
    
    	if model[i] == 0:    
    		indices_i = ColIndices(i,n)
    		model[i] = SSE( X[:, indices_i], Y, weights )

        for ij in xrange(n):
            j = (1<<ij)
            if i == i|j: continue
              
            if model[i|j] == 0: 
            	indices_ij = ColIndices(i|j,n)    
            	model[i|j] = SSE( X[:,indices_ij], Y, weights ) ## bin(i|j)[2:] covariates in model (add extra variable)

       		Imps[ ij ] += factorial(k) * 1.*factorial(n - k -1)/factorial(n)* (model[i]-model[i|j])/model[0]     
       
            #if ij==0:
            #    print bin(i)[2:].zfill(n), bin(i|j)[2:].zfill(n), model[i], model[i|j]
            #    print factorial(NVars(i)) * 1.*factorial(n - NVars(i)-1),' x ', (1.-model[i|j]/model[0])    
    return Imps