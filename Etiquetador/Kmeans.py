__authors__ = ['1668101','1665124']
__group__ = '_'

import numpy as np
import utils

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
            options['tolerance'] = 0.1
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

        #Calcula la matriz de distancias
        distances = distance(self.X, self.centroids)
        self.labels = np.array([])
        self.labels = self.labels.astype(np.int64) 

        #Calcula el indice del valor minimo y lo guarda en self.labels
        for dist in distances:
            self.labels = np.append(self.labels, np.argmin(dist))

    def get_centroids(self):

        #Guarda el valor de centroids en old_centroids
        self.old_centroids = self.centroids.copy()
        self.centroids = np.array([])
        #Por cada centroide calcula el punto central de los puntos mas cercanos a un centroide
        k = 0
        while k < self.K:
            i = 0
            aux = np.array([])
            while i < len(self.X):
                #Comprueba si el punto actual es tiene como centroide mas cercano al actual y lo guarda en una array
                if  self.labels[i] == k:
                    aux = np.append(aux, self.X[i])
                i = i + 1
            #Separa la array por puntos, los suma todos y los divide entre el numero actual de puntos
            aux = aux.reshape(-1, 3)
            count = len(aux)
            aux = np.sum(aux, axis=0)
            aux = aux/count
            #Guarda el centroide en self.centroids y calcula el siguiente centroide
            self.centroids = np.append(self.centroids, aux)
            k = k + 1
        
        self.centroids = self.centroids.reshape(-1,3)

    def converges(self):

        #Itera por cada centroide
        for i in range(len(self.centroids)):
            #Calcula el margen de error
            min = np.subtract(self.old_centroids[i], self.options['tolerance'])
            max = np.add(self.old_centroids[i], self.options['tolerance'])
            #Comprueba si el centroide actual está dentro de los margenes
            compare1 = np.greater_equal(min, self.centroids[i])
            compare2 = np.greater_equal(self.centroids[i], max)
            compare = compare1*compare2
            #Si algún centroide está fuera de los margenes, devuelve false
            if (compare[0] and compare[1] and compare[2]) == True:
                return False
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
    for i in X:
        i = i.astype(np.longdouble) 
        for j in C:
            j = j.astype(np.longdouble)
            #Calcula la distancia euclidiana entre los puntos y la coloca en la array
            aux = np.append(aux, np.linalg.norm(i-j))

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
