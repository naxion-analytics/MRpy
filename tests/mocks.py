import numpy as np
import pandas as pd

#### random data with weights
np.random.seed(42)
data = pd.DataFrame(np.random.randn(1000,5))
weights = abs(np.random.rand(1000))


known_covariance_weighted = [[ ]]
known_covariance_unweighted = [[ ]]
known_shapley = []
known_R2 = None