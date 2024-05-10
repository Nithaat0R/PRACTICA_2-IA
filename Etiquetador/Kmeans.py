__authors__ = ['1668101','1665124', '1667459']
__group__ = '324'

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
            options['tolerance'] = 0.5
        if 'max_iter' not in options:
            options['max_iter'] = 100
        if 'fitting' not in options:
            options['fitting'] = 20  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):

        if self.options['km_init'] == 'first':
            #Recoge los indices de los arrays de X que no se repitan pero están ordenados en orden creciente de los arrays
            aux = np.unique(self.X, axis=0, return_index=True)[1]
            #Ordena los indices para obtener los arrays en el mismo orden que en X
            aux = self.X[np.sort(aux)]
            aux = aux.astype(np.float64)
            #Selecciona los primeros K arrays 
            self.centroids = aux[:self.K]
            self.old_centroids = self.centroids.copy()

        elif self.options['km_init'] == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

        elif self.options['km_init'] == 'kmeans++':
            #Afegeix el primer node
            self.centroids.append(self.X[0])
            n = 1
            while n < self.X:
                #Calcula la distancia entre nodes
                dist = distance(self.X, np.array(self.centroids))
                #Busca la distancia minima entre les distancies i la probabiitat
                #de ser seleccionat
                min_dist = np.min(dist, axis=1)
                prob = min_dist / np.sum(min_dist)
                #Escull un node aleatori
                new_centroid_ind = np.random.choice(len(self.X), p=prob)
                #Comprova si el node seleccionat es diferent als que ja existeixen
                if self.add_centroids(new_centroid_ind):
                    n = n + 1


    def get_labels(self):

        #Calcula la matriz de distancias
        distances = distance(self.X, self.centroids)
        #Calcula el centroide mas cercano a cada punto
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):

        #Guarda el valor de centroids en old_centroids
        self.old_centroids = self.centroids.copy()
        self.centroids = np.array([])
        #Por cada centroide calcula el punto central de los puntos mas cercanos a un centroide
        for k in range(self.K):
            #Guarda todos los puntos mas cercanos a k
            aux = self.X[self.labels == k]
            #Calcula la media de los puntos para encontrar la nueva k
            self.centroids = np.append(self.centroids, np.mean(aux, axis = 0))
        self.centroids = self.centroids.reshape(-1,3)

    def converges(self):

        #Itera por cada centroide
        for i in range(len(self.centroids)):
            #Calcula el margen de error
            min = np.subtract(self.old_centroids[i], self.options['tolerance'])
            max = np.add(self.old_centroids[i], self.options['tolerance'])
            #Comprueba si el centroide actual está dentro de los margenes
            compare1 = np.greater_equal(self.centroids[i], min)
            compare2 = np.greater_equal(max, self.centroids[i])
            compare = compare1*compare2
            #Si algún centroide está fuera de los margenes, devuelve false
            if compare[0]*compare[1]*compare[2] == False:
                return False
        return True

    def fit(self):
        
        #Inicializa los centroides
        self._init_centroids()
        #Calcula los centroides mientras que el converges sea False.
        #El caso 0 es una excepción porque al inicializar, centroides y old_centroides tienen el mismo valor
        while (not self.converges() or self.num_iter == 0) and self.num_iter < self.options['max_iter']:
            #Calcula los nuevos puntos mas cercanos
            self.get_labels()
            #Calcula los nuevos centroides
            self.get_centroids()
            self.num_iter += 1
        #Hay que volver a colocar esto a 0 sino no se puede volver a usar luego :c
        self.num_iter = 0

    def withinClassDistance(self):

        wcd = 0
        #Calculem les distancies entre tots els punts i centroides.
        dist = distance(self.X, self.centroids)
        i = 0
        #Per cada punt suma la distancia amb el node més proper al quadrat.
        while i < self.X.shape[0]:
            wcd += dist[i, self.labels[i]]**2
            i += 1
        #Divideix la suma de totes les distancies per trobar la mitjana
        wcd /= i

        return wcd

    def interClassDistance(self):
        icd = 0
        for i in range(self.K):
            for j in range(i+1, self.K):
                #Calcula la distancia euclediana al quadrat del parell de nodes
                icd += np.linalg.norm(self.centroids[i] - self.centroids[j])**2
        #Calculem la mitjana de las distancies inter-clase
        icd /= (self.K * (self.K - 1)) / 2
        return icd

    def fisherCoefficient(self):
        #Calcula el coeficient de fisher
        wcd = self.withinClassDistance()
        icd = self.interClassDistance()
        return icd / wcd

    def find_bestK(self, max_K):
        
        cond = False
        #Calculamos WCD para K = 2
        self.K = 2
        self.fit()
        wcd = self.withinClassDistance()
        k = 3
        #Mientras que k sea inferior a max_k y no se cumpla la condicion, se aumenta la K optima
        while k < max_K and not cond:
            old_WCD = wcd
            #Calculamos el nuevo WCD
            self.K = k
            self.fit()
            wcd = self.withinClassDistance()
            #Calculamos el porcentaje de diferencia entre WCD y old_WCD
            dec = 100*(wcd/old_WCD)

            #Si llegamos a un decrecimiento estabilizado salimos del bucle
            if 100 - dec < self.options['fitting']:
                cond = True
                self.K -= 1
            else:
                k += 1
    


def distance(X, C):

    aux = np.zeros([X.shape[0], C.shape[0]])
    for j in range(0, C.shape[0]):
        aux[:,j] = np.linalg.norm(X - C[j,:], ord=2, axis=1)

    return aux


def get_colors(centroids):

    #Calcula las probabilidades de cada color
    prob = utils.get_color_prob(centroids)
    #Ordena los indices de todas las probabilidades de menor a mayor
    aux = np.argmax(prob,axis=1)  
    return utils.colors[aux]
