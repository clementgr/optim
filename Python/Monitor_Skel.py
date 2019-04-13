#!/usr/bin/python

import numpy as np

#############################################################################
#                                                                           #
#  MONITEUR D'ENCHAINEMENT POUR LE CALCUL DE L'EQUILIBRE D'UN RESEAU D'EAU  #
#                                                                           #
#############################################################################

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


##### Fonctions a ecrire dans le cadre du projet

# ---> Charger les fonctions associees a l'oracle du probleme,
#      aux algorithmes d'optimisation et de recherche lineaire

# gradient a pas fixe
from OraclePG import OraclePG
from Gradient_F import Gradient_F

# gradient a pas variable
# from OraclePG import OraclePG
# from Gradient_V import Gradient_V
# from Wolfe_Skel import Wolfe

# Gradient Polak-Ribiere
# from OraclePG import OraclePG
# from Polak_Ribiere import Polak_Ribiere
# from Wolfe_Skel import Wolfe

# Gradient BFGS Inverse
from OraclePG import OraclePG
from BFGS import BFGS
from Wolfe_Skel import Wolfe

# Newton a pas fixe
# from OraclePH import OraclePH
# from Newton_F import Newton_F

# Newton a pas variable
# from OraclePH import OraclePH
# from Newton_V import Newton
# from Wolfe_Skel import WolfePH


##### Initialisation de l'algorithme

# ---> La dimension du vecteur dans l'espace primal est n-md
#      et la dimension du vecteur dans l'espace dual est md

# primal
x0 = 0.1 * np.random.normal(size=n-md)

# dual
# x0 = 100 + np.random.normal(size=md)


##### Minimisation proprement dite

# ---> Executer la fonction d'optimisation choisie

# Gradient a pas fixe
# print()
# print("ALGORITHME DU GRADIENT A PAS FIXE")
# copt, gopt, xopt = Gradient_F(OraclePG, x0)

# Gradient a pas variable
# print()
# print("ALGORITHME DU GRADIENT A PAS VARIABLE")
# copt, gopt, xopt = Gradient_V(OraclePG, x0)

# Gradient BFGS Inverse
print()
print("ALGORITHME DE GRADIENT BFGS")
copt, gopt, xopt = BFGS(OraclePG, x0)

# Gradient Polak-Ribiere
#print()
#print("ALGORITHME DE GRADIENT POLAK-RIBIERE")
#copt, gopt, xopt = Polak_Ribiere(OraclePG, x0)

# Newton a pas fixe
#print()
#print("ALGORITHME DE NEWTON A PAS FIXE")
#copt, gopt, xopt = Newton_F(OraclePH, x0)

# Newton a pas variable
#print()
#print("ALGORITHME DE NEWTON A PAS VARIABLE")
#copt, gopt, xopt = Newton(OraclePH, x0)

##### Verification des resultats

# ---> La fonction qui reconstitue les variables hydrauliques
#      du reseau a partir de la solution du probleme s'appelle
#      HydrauliqueP pour le probleme primal, et HydrauliqueD
#      pour le probleme dual

# primal
qopt, zopt, fopt, popt = HydrauliqueP(xopt)

# dual
# qopt, zopt, fopt, popt = HydrauliqueD(xopt)


Verification(qopt, zopt, fopt, popt)
