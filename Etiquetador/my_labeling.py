__authors__ = ['1668101','1665124','1667459']
__group__ = '324'

import utils
from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans import *
from KNN import *
import numpy as np
import matplotlib.pyplot as plt

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
    wcdList = []
    iterList = []
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

        #kmeans.fit()
        kmeans._init_centroids()
        while (not kmeans.converges() or kmeans.num_iter == 0) and kmeans.num_iter < kmeans.options['max_iter']:
            kmeans.get_labels()
            kmeans.get_centroids()
            kmeans.num_iter += 1
        iterList = np.append(iterList, kmeans.num_iter)
        kmeans.num_iter = 0

        wcd = kmeans.withinClassDistance()
        #Calculamos el porcentaje de diferencia entre WCD y old_WCD
        dec = 100*(wcd/old_WCD)
        decList = np.append(decList, dec)
        wcdList = np.append(wcdList, wcd)
    
    #Creem el gràfic
    x = np.arange(2,kmax-1,1)
    fig, axs = plt.subplots(3)
        
    
    axs[0].plot(x, decList[x-2])
    axs[0].set_xlabel('Number of K')
    axs[0].set_ylabel('DEC')
    axs[0].set_title('DEC')

    
    axs[1].plot(x, wcdList[x-2])
    axs[1].set_xlabel('Number of K')
    axs[1].set_ylabel('WCD')
    axs[1].set_title('WCD')

    
    axs[2].plot(x, iterList[x-2])
    axs[2].set_xlabel('Number of K')
    axs[2].set_ylabel('Iterations')
    axs[2].set_title('Iterations')

    
    fig.tight_layout()
    plt.show()
        
Kmean_statistics(km, 30)

knn = KNN(utils.rgb2gray(train_imgs), train_class_labels)


def Get_shape_accuracy(etiquetesKnn, groundtruth): #Li pasem el groundtruth o el array de labels?
    #Inicialitzem el comptador de coincidències en 100% és a dir en totes.
    comptador = len(etiquetesKnn)
    #Anem un a un i en cas de que no coincideixin restem un al total
    for i in range(len(etiquetesKnn)):
        if etiquetesKnn[i] != groundtruth[i]:
            comptador = comptador - 1
   
    #Retornem el calcul del percentatge
    return comptador/len(etiquetesKnn)*100

preds = knn.predict(utils.rgb2gray(test_imgs), 2)
shapeAcc = Get_shape_accuracy(preds, test_class_labels)
print("Shape accuracy: ", shapeAcc)

def Get_color_accuracy(etiquetesKmeans, groundtruth):
    Netiquetes = 0
    count = 0
    for i in range(len(etiquetesKmeans)):
        repetits = []
        for color in etiquetesKmeans[i]:
            if color in groundtruth[i] and color not in repetits:
                count += 1
                repetits = np.append(repetits, color)
        
        Netiquetes += len(groundtruth[i])
    
    return (count/Netiquetes)*100

KMprova = KMeans(test_imgs[0], 4)
KMprova.fit()
TestColors = get_colors(KMprova.centroids)

for i in range(len(test_imgs) - 1):
    KMprova = KMeans(test_imgs[i + 1], 4)
    KMprova.fit()
    colors = get_colors(KMprova.centroids)
    TestColors = np.vstack((TestColors, colors))

colorAcc = Get_color_accuracy(TestColors, test_color_labels)
print("Color accuracy: ", colorAcc)

def Get_K_accuracy(test_imgs, groundtruth, maxK, heuristic):
    encerts = 0
    for i in range(len(test_imgs)):
        KMprova = KMeans(test_imgs[i], 4)
        KMprova.find_bestK(maxK, heuristic)
        if KMprova.K == len(groundtruth[i]):
            encerts += 1
    
    return (encerts/len(test_imgs))*100

kAcc = Get_K_accuracy(test_imgs, test_color_labels, 4, 'fisher')
print("K accuracy: ", kAcc)