__authors__ = ['1668101','1665124', '1667459']
__group__ = '324'

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans import *
import numpy as np


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
images = retrieval_by_color(imgs,color_labels, ['Blue','Black'])