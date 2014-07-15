import numpy as np
from scipy import optimize
from abc import abstractmethod

from ..utils import *

__all__ = ['ConstrainedRegression']

class ConstrainedRegression():
    """
    Base class for constrained regression models.
    """

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

    @abstractmethod
    def fit(self, X, y, weights = None ):
        """
        Fit  model
        """

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

    