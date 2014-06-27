import numpy as np


#### random data with weights
X = np.random.rand(15,3)
X[:,0] = 0.5 + .25*X[:,1] + np.random.rand(15)
X[:,2] = X[:,1]*.01
weights = np.random.rand(10+5)