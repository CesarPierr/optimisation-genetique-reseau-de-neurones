
from keras.datasets import mnist
from keras.utils import np_utils
from numpy import random

from neurones2 import *

#importation des données qui vont servir à entrainer le reseau
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#formatage des données initiales 
# 
# on transforme en vecteur 
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
#on change le type pour la division
x_train = x_train.astype('float32')
#on rapporte à des données entre 0 et 1 pour les couches de gros
x_train /= 255
#on change les parametre de sortie de 0 à 9 en [0,0,0,0,0,0,0,0,1,0] si la reponse est 8
y_train = np_utils.to_categorical(y_train)

#idem sur les fonctions test
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

from random import randint
import numpy as np
from fonction import *

l_fonctions = [[tanh,tanh_prime],[relu,relu_prime],[atan,atan_prime],[softplus,softplus_prime]]



class Population :

    def __init__(self) :
        self.nb_layers = randint(3,10)
        self.nb_trans = randint(5,40)
        self.nb_neurones_couches = [randint(20,200) for i in range(self.nb_layers-1)]
        self.f_acti = l_fonctions[2]
        self.learning_rate = 0.1

    def fitness(self) :
        net = Network()
        net.add(FCLayer(28*28, self.nb_neurones_couches[0]))
        net.add(ActivationLayer(self.f_acti[0], self.f_acti[1]))

        for i in range(self.nb_layers-2):
            net.add(FCLayer(self.nb_neurones_couches[i], self.nb_neurones_couches[i+1]))
            net.add(ActivationLayer(self.f_acti[0], self.f_acti[1]))


        net.add(FCLayer(self.nb_neurones_couches[-1], 10))                   
        net.add(ActivationLayer(self.f_acti[0], self.f_acti[1]))
        net.use(mse, mse_prime)
        net.fit(x_train[0:1000], y_train[0:1000], epochs=self.nb_trans, learning_rate=self.learning_rate)
        out = net.predict(x_test[0:3])
        print("\n")
        print("predicted values : ")
        print(out, end="\n")
        print("true values : ")
        print(y_test[0:3])
