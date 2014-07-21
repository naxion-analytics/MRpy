import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .base import *
from ..utils import *

__all__ = ['ExtraTreesRegression']

class ExtraTreesRegression(ConstrainedRegression):
    """
    Fit a constrained regression model with the importances from an extra tree classifier
    """
    
    def fit(self, X, y, weights = None, **kwargs):
        if weights is None: weights = np.ones(y.shape[0])
        data = np.hstack((y.reshape(y.shape[0],1),X))
        
        S = wcov(data, weights)
        corr = wcorr(data, weights)
        wsd = np.sqrt(S.diagonal())
        
        ExtraTrees = ExtraTreesRegressor(**kwargs)
        ExtraTrees.fit(X,y, sample_weight=weights)
        
        Rsquare = ( S[0,1:].dot(np.linalg.inv(S[1:,1:]).dot(S[1:,0])) )/S[0,0]
        
        # assign proportion of Rsquare to each covariate dep. on importance
        self.importances = ExtraTrees.feature_importances_ * Rsquare 
        model = self.constrained_optimization( corr )
        
        if self.fit_intercept:
            w = np.diagflat( weights/np.sum(weights),k=0)
            wmean = np.sum(w.dot(data), axis=0)
            self.intercept_ = wmean[0] - wsd[0]*np.sum(wmean[1:]*model.x/wsd[1:])

        self.coef_ = wsd[0]*model.x/wsd[1:] 
        
        return self