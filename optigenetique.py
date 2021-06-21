
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
import copy
l_fonctions = [[tanh,tanh_prime],[atan,atan_prime]]



class Population :

    def __init__(self) :
        self.nb_layers = randint(2,5)
        self.nb_neurones_couches = [randint(20,100) for i in range(self.nb_layers-1)]
        self.f_acti = l_fonctions[randint(0,1)]
        self.learning_rate = [0.05 ,0.1, 0.15 ,0.2][randint(0,3)]
        self.accuracy = 1

    def fitness(self,iter,num,den,n_gen) : 
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
        print("generation ", n_gen,"/",10," : ",round(((num+1)/den)*100,2),"%",end= "")
        err = net.fit(x_train[0:1000], y_train[0:1000], epochs=iter, learning_rate=self.learning_rate)
        t2 = time.time()
        print(" temps d'entrainement du reseau : ",t2-t1)
        self.accuracy = err
        return err*(t2-t1)/iter


    def __repr__(self) :
        return "\n nb layers : " + str(self.nb_layers) +"\n" + "nb neurones/couches" + str(self.nb_neurones_couches) + "\n" +"fonction : " + str(self.f_acti) + "\n" + str(self.learning_rate)

    def mutation(self,ngeneration) :
        new = self
        mut_f = randint(1,int(20/ngeneration))
        mut_layers = randint(1,10)
        mut_learning_rate = randint(1,8)
        if mut_f == 1 :
            new.f_acti = l_fonctions[randint(0,1)]
        if mut_layers == 1 and new.nb_layers < 5 :
            new.nb_layers += 1
            new.nb_neurones_couches.append(randint(20,100))
        if mut_layers == 2 and new.nb_layers > 2 :
            new.nb_layers -= 1
            new.nb_neurones_couches = new.nb_neurones_couches[:-1]
        if mut_learning_rate == 1 :
            new.learning_rate = [0.05 ,0.1, 0.15 ,0.2][randint(0,3)]

        for i in range(len(new.nb_neurones_couches)) :
            mut_nb_neurones = randint(1,20)
            if mut_nb_neurones == 1 :
                new.nb_neurones_couches[i] = randint(20,100)
        return new
    
    def crossover(self,ind2) :

        #choose nb layers/learning rate
        new = copy.copy(Population())
        chose_layer = randint(1,2)
        chose_learning_rate = randint(1,2)
        if chose_layer == 1 :
            new.nb_layers = self.nb_layers
        else : 
            new.nb_layers = ind2.nb_layers

        new.nb_neurones_couches = [0 for i in range(new.nb_layers-1)]
        if chose_learning_rate == 1 :
            new.learning_rate = self.learning_rate
        else : 
            new.learning_rate = ind2.learning_rate

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
        return new

def calculate_limite(l,n):
    l = sorted(l)
    return l[n]

def mutation(pop,n,gen) :
    ret = []
    for i in range(n) :
        a = pop[randint(0,len(pop)-1)]
        ret.append(a.mutation(gen))
    return ret

def crossover(pop,n) :
    ret = []
    for i in range(n) :
        a = pop[randint(0,len(pop)-1)]
        b = pop[randint(0,len(pop)-1)]
        ret.append(a.crossover(b))
    return ret

def del_doublons(pop) :
    l = len(pop)
    doub = []
    new = []
    for i in pop :
        if i.nb_neurones_couches in doub :
            continue
        else :
            doub.append(i.nb_neurones_couches)
            new.append(i)
    miss = len(new) - l
    complete = [Population() for i in range(miss)]
    new += complete
    return new
    
#parametres initiaux
n_pop = 15
n_gen = 10
n_lim = 5
n_mut = 5
n_cross = 5

n_trans_init = 20


l_fit2 = []
pop = [Population() for i in range(n_pop)]
x = [1,2,3,4,5,6,7,8,9,10]
moy_20 = []
meilleur = []
moy = []
for i in range(n_gen):
    n_trans = n_trans_init

    mut = mutation(pop[:],n_mut,i+1)
    cross = crossover(pop[:],n_cross)
    pop = pop + mut + cross
    pop = del_doublons(pop)
    taille = len(pop)
    resul = [indiv.fitness(n_trans,j,taille,i+1) for j,indiv in enumerate(pop)]
    newpop = []
    
    m = calculate_limite(resul,n_lim)

    l_fit2 = []

    for k,elem in enumerate(resul) :
        if elem < m :
            newpop.append(pop[k])
            l_fit2.append(resul[k])
    moy_20.append(sum(l_fit2)/len(l_fit2))
    meilleur.append(min(l_fit2))
    moy.append(sum(resul)/len(resul))
    pop = newpop
import matplotlib.pyplot as plt

for i in range(len(pop)) : 
    print("un individu est : ", pop[i])
    print("sa fitness est : ",l_fit2[i],"\n") 
plt.plot(x,moy_20,label = "moy 20 meilleurs")
plt.legend()
plt.plot(x,moy,label = "moyenne totale")
plt.legend()
plt.plot(x,meilleur,label = "meilleur resultat")
plt.legend()
plt.show()
    
