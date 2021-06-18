#definition des fonctions d'activation
from math import atan,log,exp,tanh
import numpy as np
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2


def atan(x) :
    return np.arctan(x)

def atan_prime(x) :
    return 1/(np.square(x) + 1)

def softplus(x) :
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    return 1/(1 + np.exp(-x))

