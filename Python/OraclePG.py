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
    return(np.vdot(q,r*q*np.abs(q))/3 + np.vdot(pr, np.dot(Ar,q)))

def Q(qc):
    return(q0 + np.dot(B,qc))

def gradient_inter(q):
    return(r*q*np.abs(q) + np.dot(np.transpose(Ar),pr))

def G(qc):
    return(np.dot(np.transpose(B), gradient_inter(Q(qc))))

def H(qc):
    diag = np.zeros((n,n))
    q = Q(qc)
    for i in range(n):
        diag[i,i] = r[i]*q[i]
    H_int = 2*np.dot(diag, B)
    return(np.dot(np.transpose(B), H_int))

def OraclePG(qC,ind):
    if (ind == 2):
        critere = F(Q(qC))
        return((critere, None, ind))
    elif (ind == 3):
        gradient = G(qC)
        return((None, gradient, ind))
    elif (ind == 4):
        critere, gradient = F(Q(qC)), G(qC)
        return((critere, gradient, ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')


def OraclePH(qC,ind):
    if (ind == 2):
        critere = F(Q(qC))
        return((critere, None, None, ind))
    elif (ind == 3):
        gradient = G(qC)
        return((None, gradient, None, ind))
    elif (ind == 4):
        critere, gradient = F(Q(qC)), G(qC)
        return((critere, gradient, None, ind))
    elif (ind == 5):
        hessienne = H(qC)
        return((None, None, hessienne, ind))
    elif (ind == 6):
        hessienne = H(qC)
        return((None, gradient, hessienne, ind))
    elif (ind == 7):
        critere = F(Q(qC))
        gradient = G(qC)
        hessienne = H(qC)
        return((critere, gradient, hessienne, ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')
