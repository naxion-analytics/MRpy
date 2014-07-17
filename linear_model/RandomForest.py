from sklearn.ensemble import ExtraTreesClassifier
from .base import *

class ExtraTreesRegression(ConstrainedRegression):
    """
    Fit a constrained regression model with the importances from an extra tree classifier
    """
    
    def fit(self, X,y,weights = None, **kwargs):
        if weights is None: weights = np.ones(y.shape[0])
        data = np.hstack((y.reshape(y.shape[0],1),X))
        
        S = wcov(data, weights)
        corr = wcorr(data, weights)
        wsd = np.sqrt(S.diagonal())
        
        ExtraTrees = ExtraTreesClassifier(**kwargs)
        ExtraTrees.fit(X,y,sample_weight=weights)
        
        self.importances = ExtraTrees.feature_importances_
        model = self.constrained_optimization( corr )
        
        if self.fit_intercept:
            w = np.diagflat( weights/np.sum(weights),k=0)
            wmean = np.sum(w.dot(data), axis=0)
            self.intercept_ = wmean[0] - wsd[0]*np.sum(wmean[1:]*model.x/wsd[1:])

        self.coef_ = wsd[0]*model.x/wsd[1:] 
        
        return self