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
    #return np.vdot(q,r*q*np.abs(q))/3 + np.dot(pr,np.dot(Ar,q))

def Q(qc):
    return q0+np.matmul(B,qc)
    #return(q0 + np.dot(B,qc))

def gradient_F_tilde(q):
    return r*q*np.absolute(q) + np.matmul(np.transpose(Ar),pr)
    #return(r*q*np.abs(q) + np.dot(np.transpose(Ar),pr))

def G(qc):
    return np.matmul(np.transpose(B),gradient_F_tilde(Q(qc)))
    #return(np.dot(np.transpose(B), gradient_F_tilde(Q(qc))))

def H(qc):
    return 2*np.matmul(np.transpose(B), np.matmul(np.diag(r*q), B))    
#def H(qc):
#    diag = np.zeros((n,n))
#    q = Q(qc)
#    for i in range(n):
#        diag[i,i] = r[i]*q[i]
#    H_int = 2*np.dot(diag, B)
#    return(np.dot(np.transpose(B), H_int))

def OraclePG(qC,ind):
    if (ind == 2):
        return((F(Q(qC), None, ind))
    elif (ind == 3):
        return((None, G(qC), ind))
    elif (ind == 4):
        return((F(Q(qC)), G(qC), ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')

def OraclePH(qC,ind):
    if (ind == 2):
        return((F(Q(qC)), None, None, ind))
    elif (ind == 3):
        return((None, G(qC), None, ind))
    elif (ind == 4):
        return((F(Q(qC)), G(qC), None, ind))
    elif (ind == 5):
        return((None, None, H(qC), ind))
    elif (ind == 6):
        return((None, G(qC), H(qC), ind))
    elif (ind == 7):
        return((F(Q(qC)), G(qC), H(qC), ind))
    else:
        print('la valeur de ind ne correspond à aucune entrée possible')
