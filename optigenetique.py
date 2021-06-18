
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
x_test = x_test.astype('float128')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

from random import randint
import numpy as np
from fonction import *
import time

l_fonctions = [[tanh,tanh_prime],[atan,atan_prime]]



class Population :

    def __init__(self) :
        self.nb_layers = randint(2,10)
        self.nb_neurones_couches = [randint(20,200) for i in range(self.nb_layers-1)]
        self.f_acti = l_fonctions[randint(0,1)]
        self.learning_rate = randint(1,500)/100
        self.accuracy = 1

    def fitness(self,iter) : 
        net = Network()
        net.add(FCLayer(28*28, self.nb_neurones_couches[0]))
        net.add(ActivationLayer(self.f_acti[0], self.f_acti[1]))

        for i in range(self.nb_layers-2):
            net.add(FCLayer(self.nb_neurones_couches[i], self.nb_neurones_couches[i+1]))
            net.add(ActivationLayer(self.f_acti[0], self.f_acti[1]))


        net.add(FCLayer(self.nb_neurones_couches[-1], 10))                   
        net.add(ActivationLayer(self.f_acti[0], self.f_acti[1]))
        net.use(mse, mse_prime)
        t1 = time.time()
        err = net.fit(x_train[0:1000], y_train[0:1000], epochs=iter, learning_rate=self.learning_rate)
        t2 = time.time()
        self.accuracy = err
        return err*(t2-t1)/iter


    def __repr__(self) :
        return "nb layers : " + str(self.nb_layers) +"\n" + "nb neurones/couches" + str(self.nb_neurones_couches) + "\n" +"fonction : " + str(self.f_acti)

    def mutation(self,ngeneration) :
        mut_f = randint(1,int(50/ngeneration))
        mut_layers = randint(1,int(10*ngeneration/self.accuracy))
        mut_learning_rate = randint(1,5)
        if mut_f == 1 :
            self.f_acti = l_fonctions[randint(0,1)]
        if mut_layers == 1 and self.nb_layers < 10 :
            self.nb_layers += 1
            self.nb_neurones_couches.append(randint(20,200))
        if mut_layers == 2 and self.nb_layers > 2 :
            self.nb_layers -= 1
            self.nb_neurones_couches = self.nb_neurones_couches[:-1]
        for i in range(len(self.nb_neurones_couches)) :
            mut_nb_neurones = randint(1,30)
            if mut_nb_neurones == 1 :
                self.nb_neurones_couches[i] = randint(20,200)
        return self
    
    def crossover(self,ind2) :
        #choose nb layers
        new = Population()
        chose_layer = randint(1,2)

        if chose_layer == 1 :
            new.nb_layers = self.nb_layers
        else : 
            new.nb_layers = ind2.nb_layers
        #choose the nb of neurones from the layer
        chose_neurones = [randint(0,1) for i in range(new.nb_layers-1)]
        for k,elem in enumerate(chose_neurones) :
            if elem == 0 :
                try :
                    new.nb_neurones_couches[k] = self.nb_neurones_couches[k]
                except :
                    new.nb_neurones_couches[k] = ind2.nb_neurones_couches[k]
            else : 
                try :
                    new.nb_neurones_couches[k] = ind2.nb_neurones_couches[k]
                except :
                    new.nb_neurones_couches[k] = self.nb_neurones_couches[k]


def calculate_median(l):
    l = sorted(l)
    l_len = len(l)
    ind = l_len//2
    return l[ind]

n_pop = 50
n_gen = 4
n_trans_init = 10
l_fit2 = []
pop = [Population() for i in range(n_pop)]
for i in range(n_gen):
    n_trans = int(n_trans_init*(1+i*0.5))
    pop = [ind.mutation(i+1) for ind in pop]
    resul = [indiv.fitness(n_trans) for indiv in pop]
    newpop = []
    m = calculate_median(resul)
    l_fit2 = []
    for k,elem in enumerate(resul) :
        if elem < m :
            newpop.append(pop[k])
            l_fit2.append(resul[k])
    pop = newpop

for i in range(len(pop)) : 
    print("un individu est : ", pop[i])
    print("sa fitness est : ",l_fit2[i],"\n") 

    
