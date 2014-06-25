from .shapley import *
import numpy as np


### test R2 routine

## mock data
X = np.random.rand(15,3)
X[:,0] = 0.5 + .25*X[:,1] + np.random.rand(15)
X[:,2] = X[:,1]*.01
weights = np.random.rand(10+5)
wgts = np.diagflat(weights,k=0)

w = np.diagflat( weights/np.sum(weights),k=0)
X = np.sqrt(w).dot(X - np.sum( w.dot(X),axis=0))
S = 1./(1. - np.sum(w**2) ) * X.T.dot(X)

import statsmodels.api as sm
ww = weights/np.sum(weights)

### weighted linear regression using the two covariates
model = sm.WLS(X[:,0],X[:,1:], weights = ww).fit()
model.rsquared == R2(S, 2, 2) ## R2 with two covariates in it. These should be equal