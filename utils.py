import numpy as np
from numba import jit

__all__ = ['wcov','wcorr']


@jit('f8[:,:](f8[:,:],f8[:])')
def wcov(data, weights):
	""" 
	Function to compute weighted covariance matrix correctly. 
	"""
	w = np.diagflat( weights/np.sum(weights), k=0)
	data = np.sqrt(w).dot(data - np.sum( w.dot(data),axis=0))

	return 1./(1.-np.sum(w**2)) * data.T.dot(data)


@jit('f8[:,:](f8[:,:], f8[:])')
def wcorr(data, weights):
	"""
	weighted correlation function
	"""
	std = np.diagflat( np.sqrt( wcov(data, weights).diagonal())**-1 )
	return std.dot(wcov(data, weights) ).dot(std)