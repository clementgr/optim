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
    return F(q) + np.dot(lbd, np.matmul(Ad,q) - fd)

def X(lbd):
    return np.matmul(np.transpose(Ar),pr) + np.matmul(np.transpose(Ad),lbd)

def Q_chapeau(lbd):
    q_chapeau = np.zeros(n)
    x = X(lbd)
    for i in range(n):
        q_chapeau[i] = - np.sign(x[i]) * np.sqrt(np.abs(x[i])/r[i])
    return q_chapeau

def Phi(lbd):
    return(Lagrangien(Q_chapeau(lbd), lbd))

def gradient_Phi(lbd):
    return np.matmul(Ad, Q_chapeau(lbd)) - fd

def hessien_Phi(lbd):
    x = X(lbd)
    D = np.zeros((n,n))
    for k in range(n):
        D[k,k] = -1 / (2 * np.sqrt(r[k] * np.abs(x[k])))
    return np.matmul(Ad, np.matmul(D, np.transpose(Ad)))

def OracleDG(lbd,ind):
    if (ind == 2):
        return((-Phi(lbd), None, ind))
    elif (ind == 3):
        return((None, -gradient_Phi(lbd), ind))
    elif (ind == 4):
        return((-Phi(lbd), -gradient_Phi(lbd), ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')

def OracleDH(lbd,ind):
    if (ind == 2):
        return((-Phi(lbd), None, None, ind))
    elif (ind == 3):
        return((None, -gradient_Phi(lbd), None, ind))
    elif (ind == 4):
        return((-Phi(lbd), -gradient_Phi(lbd), None, ind))
    elif (ind == 5):
        return((None, None, -hessien_Phi(lbd), ind))
    elif (ind == 6):
        return((None, -gradient_Phi(lbd), -hessien_Phi(lbd), ind))
    elif (ind == 7):
        return((-Phi(lbd), -gradient_Phi(lbd), -hessien_Phi(lbd), ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')
