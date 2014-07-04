import numpy as np
from scipy import optimize
from numba import jit
from math import factorial

__all__ = ['NVars','Rsq', 'ShapleyValue', 'KruskalReg']

@jit('i8(i8)')
def NVars(n):
    """
    Count the number of non-zero bits in the integer n. These correspond to the columns/variables
    included in the model.
    """
    count=0
    while n>0:
        count += 1
        n = n & (n-1)
    return count

@jit('f8(f8[:,:],i8,i8)')
def Rsq(S, i, n):
    """
    Computs the coefficient of determinantion (R-squared) from the covariance matrix S for a model 
    containing variables i out of n total covariates. 
    """
    cov = np.zeros(n+1,dtype=bool)
    for k in range(n):
        if i & (1<<k)!=0: cov[k+1] = True

    Sxx = S[:,cov]
    Sxx_inv = np.linalg.inv( Sxx[cov,:] )
    Syx = S[0,cov]; Sxy = S[cov,0]
    return Syx.dot(Sxx_inv.dot(Sxy) )

@jit('f8[:](f8[:,:])')
def ShapleyValue( S ):
    """
    Computes the Shapley importance of each n covariate in a linear model from the (weighted) covariance
    (n+1) x (n+1) matrix. Returns a vector of length n giving the average amount of R2 attributed 
    to the nth variable. 
    """
    n_cov = S.shape[1]-1
    model = np.zeros(2**n_cov) ### storage space to memoize the models computed
    shapley =np.zeros(n_cov)   ### storage space for the shapley importances
    
    model[0] = 0.  ### no covariates in model, base case, R2 for no covariates
    for i in xrange(2**n_cov):
        
        k = NVars(i)
        if model[i] == 0: model[i] = Rsq(S,i,n_cov)
        
        for ij in xrange(n_cov):
            ### add the ij variable to the i. if its already in i, continue, else compute new model
            j = (1<<ij)  
            if i == i|j: continue
                
            if model[i|j] == 0:  model[i|j] = Rsq(S,i|j,n_cov) 

            ### compute the improvement in R2 given the addition of the jth variable and average over
            ### permutations possible
            shapley[ ij ] += factorial(k) * 1.*factorial(n_cov - k -1)/factorial(n_cov)* (model[i|j]-model[i])/model[0]
            
    return shapley


def KruskalReg(corr, targets):
    """
    Find the linear regression coefficients given desired net effects. A wrapper for a bounded L-BFGS-B
    optimizer.
    """
    fit = optimize.minimize( np.sum( (np.multiply(x.dot(corr[1:,1:]), x) - targets)**2 ), 
                method='L-BFGS-B',
                x0 = np.array([0.001]*corr[1:,1:].shape[0]), 
                bounds = [(0,None)]*corr[1:,1:].shape[0] )

    return fit



    