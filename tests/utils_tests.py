import numpy as np
from mocks import *
from ..utils import *


def cov_unweighted_test():
	y = np.array( data.cov()) ### unweighted covariance
	x = cov(np.array(data), weights = None)
	return len(y[np.where(abs(y-x)>1e-12)]) == 0

def cov_weighted_test():
	from statsmodels.stats.weightstats import DescrStatsW

	dd = DescrStatsW( np.array(data) ,weights=weights)
	y = dd.cov
	x = cov(np.array(data), weights=weights ) 
	return len(y[np.where(abs(y-x)>1e-12)]) == 0



if __name__ == '__main__':
	if cov_unweighted_test():
		print 'utils.cov passed unweighted test'
	if cov_weighted_test():
		print 'utils.cov passed weighted test'
