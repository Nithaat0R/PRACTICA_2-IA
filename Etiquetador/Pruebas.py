import numpy as np
import utils

X = np.random.rand(10, 10, 3)

if X.dtype != np.float64:
    X = X.astype(np.float64)

if X.ndim == 3:
    X = X.reshape(-1,3)
