__authors__ = ['1668101','1665124','1667459']
__group__ = '324'


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

        ret = np.array([])
        #Por cada imagen selecciona el valor que mas se repite
        for i in self.neighbors:
            max = 0
            count = 0
            #Itera en cada valor de la imagen para ver si es el que más se repite
            for j in i:
                #Si el valor actual se repite mas que el guardado guarda el valor actual
                if max < np.count_nonzero(i == j):
                    label = i[count]
                    max = np.count_nonzero(i == j)
                count += 1
            #Guarda el label que mas se repite en una array
            ret = np.append(ret, label)
        return ret

    def predict(self, test_data, k):
        #Codigo original del archivo
        self.get_k_neighbours(test_data, k)
        return self.get_class()
