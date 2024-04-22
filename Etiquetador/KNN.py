__authors__ = ['1668101','1665124', '1667459']
__group__ = '_'


import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):

        #Si train_data contiene valores que no sean tipo float los transforma
        if train_data.dtype != np.float64:
            train_data = train_data.astype(np.float64)
        
        #Redimensiona la matriz para que tenga forma (X, 4800)
        self.train_data = train_data.reshape(-1, 4800)

    def get_k_neighbours(self, test_data, k):

        #Reordena el tamaño de la matriz de datos por imagenes
        test_data = test_data.reshape(-1,4800)
        #Calcula las distancias entre los datos de la muestra y los datos por defecto
        distances = cdist(test_data, self.train_data)
        self.neighbors = np.array([])
        #Por cada imagen ordena las distancias y guarda las k primeras etiquetas
        for i in distances:
            #Consigue los indices ordenados
            aux  = np.argsort(i)
            aux = aux.astype(np.int64)
            j = 0
            #Guarda las etiquetas de las k distancias mas bajas
            while j < k:
                self.neighbors = np.append(self.neighbors, self.labels[aux[j]])
                j += 1
        #Reordena el tamaño de la matriz de vecinos por imagenes
        self.neighbors = self.neighbors.reshape(-1, k)

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
