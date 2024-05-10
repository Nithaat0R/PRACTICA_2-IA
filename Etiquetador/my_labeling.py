__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
<<<<<<< Updated upstream

=======
from Kmeans import *
from KNN import *
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> Stashed changes

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
<<<<<<< Updated upstream
=======

def retrieval_by_color(images, labels, colors):

    indices = np.array([])
    
    #Depende si se le pasa uno o mas colores hará un recorrido u otro
    if type(colors) is list:
        #Por cada color busca todos los indices de las imagenes que contengan ese color
        for color in colors:
            #Por cada imagen comprueba si contiene el color que buscamos
            for i in range(len(labels)):
                if color in labels[i]:
                    #Si la imagen tiene el color buscado añade el indice de la imagen a la lista
                    indices = np.append(indices, i)
        #Elimina los indices repetidos (Si contiene mas de un color de la lista)
        indices = np.unique(indices)
    else:
        #Por cada imagen comprueba si contiene el color que buscamos
        for i in range(len(labels)):
            if colors in labels[i]:
                #Si la imagen tiene el color buscado añade el indice de la imagen a la lista
                indices = np.append(indices, i)
    #Cambia el tipo de dato de los indices a int (Se cambian a float por el append)
    indices = indices.astype(int)
    #Devuelve las imagenes correspondientes a los indices que hemos encontrado
    ret = images[indices]

    return ret

#Llama a la función retrieval_by_color para poder probarla
images1 = retrieval_by_color(imgs,color_labels, ['Blue','Black'])

def retrieval_by_shape(images, labels, shape):

    indices = np.array([])
    
    #Por cada imagen comprueba si contiene la forma que buscamos
    for i in range(len(labels)):
        if shape == labels[i]:
            #Si la imagen tiene la forma buscada añade el indice de la imagen a la lista
            indices = np.append(indices, i)
    #Cambia el tipo de dato de los indices a int (Se cambian a float por el append)
    indices = indices.astype(int)
    #Devuelve las imagenes correspondientes a los indices que hemos encontrado
    ret = images[indices]

    return ret

#Llama a la función retrieval_by_shape para poder probarla
images2 = retrieval_by_shape(imgs, class_labels, 'Dresses')

def retrieval_combined(images, color_labels, shape_labels, colors, shape):
    
    #Ejecuta la función retrieval_by_color.
    #Ejecuta el contenido en vez de llamar a la función porque necesitamos datos que se calculan dentro de la función para llamar a la siguiente

    indices = np.array([])
    
    if type(colors) is list:
        for color in colors:
            for i in range(len(color_labels)):
                if color in color_labels[i]:
                    indices = np.append(indices, i)
        indices = np.unique(indices)
    else:
        for i in range(len(color_labels)):
            if colors in color_labels[i]:
                indices = np.append(indices, i)

    indices = indices.astype(int)
    imgs = images[indices]

    #Elimina los labels de forma que se correspondan con las imagenes que hemos eliminado anteriormente
    shape_labels = shape_labels[indices]

    #Llama a la función retrieval_by_shape con las imagenes nuevas y los labels actualizados
    imgs = retrieval_by_shape(imgs, shape_labels, shape)

    return imgs

#Llama a la función retrieval_by_shape para poder probarla
images3 = retrieval_combined(imgs, color_labels, class_labels, ['Blue','Black'], 'Dresses')

km = KMeans(train_imgs[0], 10)

def Kmean_statistics(kmeans, kmax):
        
        decList = []
        #Calculamos WCD para K = 2
        kmeans.K = 2
        kmeans.fit()
        wcd = kmeans.withinClassDistance()
        k = 3
        #Mientras que k sea inferior a max_k y no se cumpla la condicion, se aumenta la K optima
        for k in range(3,kmax):
            old_WCD = wcd
            #Calculamos el nuevo WCD
            kmeans.K = k
            kmeans.fit()
            wcd = kmeans.withinClassDistance()
            #Calculamos el porcentaje de diferencia entre WCD y old_WCD
            dec = 100*(wcd/old_WCD)
            decList = np.append(decList, dec)
        
        #Establim que es x i y
        x = np.arange(2,kmax-1,1)
        y = decList[x-2]
        
        #Creem el gràfic
        plt.plot(x,y)
        plt.xlabel('Number of K')
        plt.ylabel('DEC')
        plt.title('Kmean STATISTICS')
        plt.show()
        
Kmean_statistics(km, 30)

def Get_shape_accuracy(etiquetesKnn, groundtruth): #Li pasem el groundtruth o el array de labels?
    #Inicialitzem el comptador de coincidències en 100% és a dir en totes.
    comptador = len(etiquetesKnn)
    #Anem un a un i en cas de que no coincideixin restem un al total
    for i in range(len(etiquetesKnn)):
        if etiquetesKnn[i] != groundtruth['class_labels'][i]:
            comptador = comptador - 1
   
    #Retornem el calcul del percentatge
    return comptador/len(etiquetesKnn)*100

def Get_color_accuracy(etiquetesKmeans, groundtruth):
    #Inicialitzem el comptador de coincidències en 100% és a dir en totes.
    comptador = len(etiquetesKmeans)
    
    #Iterem cada llista de colors
    for i in range(len(etiquetesKmeans)):
        #Inicialitzem el comptador de coincidències en 0
        coinc = 0
        #Iterem dins de la llista de etiquetes correctes
        for j in range(len(groundtruth['color_labels'][i])):
            #Itereme a la vegada per la llista de etiquetes que ens ha retornat el Kmeans
            f = 0
            tr = False
            #Quan troba una coincidència para de iterar, suma al 1 al comptador i es pasa a la següente posicio del llistat de correctes. Sinó
            #Pasa a la següent posició
            while f < len(etiquetesKmeans[i]) and tr == False:
                if etiquetesKmeans[i][f] == len(groundtruth['color_labels'][i][j]):
                    coinc = coinc + 1
                    tr = True
                else:
                    f = f + 1
        #Cada vegada que acabem una de les llistes de etiquetes restem al total el percentatge de aquesta equivalent dels que hem errat.
        comptador = comptador - (len(groundtruth['color_labels'][i])-coinc/len(groundtruth['color_labels'][i]))
   
    #Retornem el calcul del percentatge
    return comptador/len(etiquetesKmeans)*100
>>>>>>> Stashed changes
