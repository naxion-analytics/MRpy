import numpy as np

__all__ = ['cov','cor']

def cov(data, weights=None):
	""" 
	Function to compute weighted covariance matrix correctly. 
	"""
	
	if weights == None:
		weights = np.ones(data.shape[0],dtype=float)

	w = np.diagflat( weights/np.sum(weights), k=0)
	data = np.sqrt(w).dot(data - np.sum( w.dot(data),axis=0))

	return 1./(1.-np.sum(w**2)) * data.T.dot(data)

def cor(covariance_matrix):
	std = np.diagflat( np.sqrt( covariance_matrix.diagonal())**-1 )
	return std.dot(covariance_matrix).dot(std)