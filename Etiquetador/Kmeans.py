__authors__ = [1668101]
__group__ = 'group'

import numpy as np
import utils
import math

class KMeans:

    def __init__(self, X, K=1, options=None):

        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X):

        #Si X contiene algun valor que no sea float, cambia todos los valores a tipo float.
        if X.dtype != np.float64:
            self.X = X.astype(np.float64) 
        
        #Si la matriz tiene 3 dimensiones la reordena para tener forma (N, 3).
        if X.ndim == 3:
            self.X = X.reshape(-1,3) 

    def _init_options(self, options=None):

        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):

        if self.options['km_init'] == 'first':
            #Recoge los indices de los arrays de X que no se repitan pero están ordenados en orden creciente de los arrays
            aux = np.unique(self.X, axis=0, return_index=True)[1]
            #Ordena los indices para obtener los arrays en el mismo orden que en X
            aux = self.X[np.sort(aux)]
            #Selecciona los primeros K arrays 
            self.centroids = aux[:self.K]
            self.old_centroids = self.centroids.copy()

        elif self.options['km_init'] == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.labels = np.random.randint(self.K, size=self.X.shape[0])

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


def distance(X, C):

    aux = np.array([])

    #Itera los valores de X y C de 3 en 3 porque tenemos que calcular las distancias entre puntos 3D
    for i1, i2, i3 in X:
        for j1, j2, j3 in C:
            #Calcula la distancia euclidiana entre los puntos y la coloca en la array
            aux = np.append(aux, math.sqrt((i1-j1)**2 + (i2-j2)**2 + (i3-j3)**2))
    #Cuando todos los valores estan dentro de la array, la convierte en una matriz N x K
    aux = aux.reshape(X.shape[0], C.shape[0])

    return aux


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
