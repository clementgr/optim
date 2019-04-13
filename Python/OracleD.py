import numpy as np

##### Fonctions fournies dans le cadre du projet

# Donnees du probleme
from Probleme_R import *
from Structures_N import *

# Affichage des resultats
from Visualg import Visualg

# Verification des resultats
from HydrauliqueP import HydrauliqueP
from HydrauliqueD import HydrauliqueD
from Verification import Verification

def F(q):
    return np.dot(q,r*q*np.absolute(q))/3 + np.dot(pr,np.matmul(Ar,q))

def Lagrangien(q,lbd):
    return F(q) + np.dot(lbd, Ad @ q - fd)


def X(lbd):
    return np.matmul(np.transpose(Ar),pr) + np.matmul(np.transpose(Ad),lbd)

def Q_chapeau(lbd):
    q_chapeau = np.zeros(n)
    x = X(lbd)
    for i in range(n):
        q_chapeau[i] = -np.sign(x[i])*np.sqrt(np.abs(x[i])/r[i])
    return q_chapeau

def fonction_duale(lbd):
    return(Lagrangien(Q_chapeau(lbd), lbd))

def G(lbd):
    return np.matmul(Ad, Q_chapeau(lbd)) - fd

def H(lbd):
    x = X(lbd)
    print('X(lbd) = {}'.format(x))
    D = np.zeros((n,n))
    for k in range(n):
        D[k,k] = np.sign(x[k]) / 2 / np.sqrt(r[k] * np.abs(x[k]))
    return np.matmul(Ad, np.matmul(D, np.transpose(Ad)))

def OracleDG(lbd,ind):
    if (ind == 2):
        return((fonction_duale(lbd), None, ind))
    elif (ind == 3):
        return((None, G(lbd), ind))
    elif (ind == 4):
        return((fonction_duale(lbd), G(lbd), ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')

def OracleDH(lbd,ind):
    if (ind == 2):
        return((fonction_duale(lbd), None, None, ind))
    elif (ind == 3):
        return((None, G(lbd), None, ind))
    elif (ind == 4):
        return((fonction_duale(lbd), G(lbd), None, ind))
    elif (ind == 5):
        return((None, None, H(lbd), ind))
    elif (ind == 6):
        return((None, G(lbd), H(lbd), ind))
    elif (ind == 7):
        return((fonction_duale(lbd), G(lbd), H(lbd), ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')

if __name__ == '__main__':
    ind = 5
    #lbd = np.array([+0.08, -1.30, +0.13, +0.09, +0.16, +0.14, +0.12, +0.07, +0.17, +0.11, +0.25, +0.01, +0.13])
    #lbd = np.ones(md)
    print(OracleDH(lbd, ind))
