from ..shapley.ShapleyValue import *
import numpy as np

def NVars_test():
	y = NVars(2519)
	if y == 8:
		return 'PASSED NVARS TEST'
	else:
		return 'FAILED NVARS TEST'


def Rsq_weighted_test():
	import statsmodels.api as sm

	y = sm.WLS(d[:,0],d[:,1:],weights=weights).fit()
	x = Rsq(cov(d, weights=weights),15,4)
	if abs(x-y.rsquared) < 5e-4:
		return 'PASSED R-SQUARED WEIGHTED TEST'
	else:
		return 'FAILED R-SQUARED WEIGHTED TEST'

def Rsq_unweighted_test():
	import statsmodels.api as sm

	y = sm.OLS(d[:,0],d[:,1:]).fit()
	x = Rsq(cov(d),15,4)
	if abs(x-y.rsquared) < 5e-4:
		return 'PASSED R-SQUARED UNWEIGHTED TEST'
	else:
		return 'FAILED R-SQUARED UNWEIGHTED TEST'

if __name__ == '__main__':
	NVars_test()
	Rsq_weighted_test()
	Rsq_unweighted_test()
