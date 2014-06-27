import numpy as np


def cov(data,weights=None):
	""" function to compute weighted covariance matrix correctly. 
	"""
	
	if weights=None:
		weights = np.ones(data.shape[0],dtype=float)

	w = np.diagflat( weight/np.sum(weights), k=0)
	X = np.sqrt(w).dot(X - np.sum( w.dot(X),axis=0))

	return 1./(1.-np.sum(w**2)) * X.T.dot(X)