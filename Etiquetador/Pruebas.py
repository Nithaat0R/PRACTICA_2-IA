import numpy as np
import utils

X = np.random.rand(10, 10, 3)

if X.dtype != np.float64:
    X = X.astype(np.float64)

if X.ndim == 3:
    X = X.reshape(-1,3)

centroids = []

for i in X.flatten():
    count = 0
    while count < len(centroids) and i != centroids[count]:
        count = count + 1
                
    if count == len(centroids):
        centroids.append(i)
                    
print(centroids)