import numpy as np
from scipy import optimize
from numba import jit
from math import factorial

from ..utils import *

__all__ = ['Nvars','Rsq','ShapleyValue','ConstrainedRegression']

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
            shapley[ ij ] += factorial(k) * 1.*factorial(n_cov - k -1)/factorial(n_cov)* (model[i|j]-model[i])/S[0,0]  
    return shapley





class ConstrainedRegression():

    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0
        self.importances = None

    def constrained_optimization(self, corr):
        """
        Find the linear regression coefficients given desired net effects. A wrapper for a bounded L-BFGS-B
        optimizer.
        """
        fit = optimize.minimize( lambda x: np.sum( (np.multiply(x.dot(corr[1:,1:]), x) - self.importances)**2 ), 
                    method='L-BFGS-B',
                    x0 = np.array([0.001]*corr[1:,1:].shape[0]), 
                    bounds = [(0,None)]*corr[1:,1:].shape[0] )
        return fit


    def fit(self, X, y, weights = None ):
        """
        Fit a linear regression with prescribed importances defined by the Shapley value for each covariate
        """
        if weights is None: weights = np.ones(y.shape[0])
        data = np.hstack((y.reshape(y.shape[0],1),X))

        S = cov(data, weights)
        corr = cor(S)
        wsd = np.sqrt(S.diagonal())

        self.importances = ShapleyValue(S)
        model = self.constrained_optimization(corr)

        if self.fit_intercept:
            w = np.diagflat( weights/np.sum(weights),k=0)
            wmean = np.sum(w.dot(data), axis=0)
            self.intercept_ = wmean[0] - wsd[0]*np.sum(wmean[1:]*model.x/wsd[1:])

        self.coef_ = wsd[0]*model.x/wsd[1:] 

        return model

    def decision_function(self, X):
        """
        Decision function of the linear model
        """
        return self.predict(X)

    def predict(self, X):
        """
        Predict using the linear model
        """
        return self.intercept_ + X.dot(self.coef_)


    def score(self,X, y, weights = None):
        """
        Returns the coefficient of determination R^2 of the prediction.
        """

        if weights is None: weights = np.ones(y.shape[0])

        y_mean = weights/np.sum(weights) *y
        y_pred = self.predict(X)
        u = ((y - y_pred)**2).sum()
        v = ((y - y_mean)**2).sum()
        return 1.-u/v

    