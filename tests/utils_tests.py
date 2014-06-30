import numpy as np
from mocks import *
from ..utils import *


def cov_unweighted_test():
	y = np.array( data.cov()) ### unweighted covariance
	x = cov(np.array(data), weights = None)
	if len(y[np.where(abs(y-x)>1e-12)]) == 0:
		return 'PASSED UNWEIGHTED TEST'
	else:
		return 'FAILED UNWEIGHTED TEST'

def cov_weighted_test():
	from statsmodels.stats.weightstats import DescrStatsW

	dd = DescrStatsW( np.array(data) ,weights=weights)
	y = dd.cov
	x = cov(np.array(data), weights=weights ) 

	if len(y[np.where(abs(y-x)>1e-12)]) == 0:
		return 'PASSED WEIGHTED TEST'
	else:
		return 'FAILED WEIGHTED TEST'



if __name__ == '__main__':
	cov_unweighted_test()
	cov_weighted_test()