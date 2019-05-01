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

from OraclePG import OraclePG
from Gradient_F import Gradient_F
from Gradient_V import Gradient_V
from Wolfe_Skel import Wolfe
from Polak_Ribiere import Polak_Ribiere
from BFGS import BFGS
from OraclePG import OraclePH
from Newton_F import Newton_F
from Newton_V import Newton
from Wolfe_Skel import WolfePH

from OracleD import OracleDG
from OracleD import OracleDH

##### Initialisation de l'algorithme

# ---> La dimension du vecteur dans l'espace primal est n-md
#      et la dimension du vecteur dans l'espace dual est md

# primal
x0 = 0.1 * np.random.normal(size=n-md)

# dual
# x0 = 100 + np.random.normal(size=md)


##### Minimisation proprement dite

# ---> Executer la fonction d'optimisation choisie

# PRIMAL

# Gradient à pas fixe
# print()
# print("ALGORITHME DU GRADIENT A PAS FIXE - PRIMAL")
# copt, gopt, xopt = Gradient_F(OraclePG, x0)

# Gradient à pas variable
# print()
# print("ALGORITHME DU GRADIENT A PAS VARIABLE - PRIMAL")
# copt, gopt, xopt = Gradient_V(OraclePG, x0)

# Gradient BFGS Inverse
# print()
# print("ALGORITHME DE GRADIENT BFGS - PRIMAL")
# copt, gopt, xopt = BFGS(OraclePG, x0)

# Gradient Polak-Ribiere
# print()
# print("ALGORITHME DE GRADIENT POLAK-RIBIERE - PRIMAL")
# copt, gopt, xopt = Polak_Ribiere(OraclePG, x0)

# Newton à pas fixe
# print()
# print("ALGORITHME DE NEWTON A PAS FIXE - PRIMAL")
# copt, gopt, xopt = Newton_F(OraclePH, x0)

# Newton à pas variable
# print()
# print("ALGORITHME DE NEWTON A PAS VARIABLE - PRIMAL")
# copt, gopt, xopt = Newton(OraclePH, x0)

# DUAL

# Gradient à pas fixe (modifier gradient_step = 0.65 dans l'initialisation dans Gradient_F.py)
# print()
# print("ALGORITHME DU GRADIENT A PAS FIXE - DUAL")
# copt, gopt, xopt = Gradient_F(OracleDG, x0)

# Gradient à pas variable
# print()
# print("ALGORITHME DU GRADIENT A PAS VARIABLE - DUAL")
# copt, gopt, xopt = Gradient_V(OracleDG, x0)

# Gradient BFGS Inverse (modifier gradient_step = 0.5 dans l'initialisation, et gradient_step = 10 dans la boucle dans BFGS.py)
# print()
# print("ALGORITHME DE GRADIENT BFGS - DUAL")
# copt, gopt, xopt = BFGS(OracleDG, x0)

# Gradient Polak-Ribiere (modifier gradient_step = 1 dans l'initialisation, et gradient_step = 1000 dans la boucle dans Polak_Ribiere.py)
# print()
# print("ALGORITHME DE GRADIENT POLAK-RIBIERE - DUAL")
# copt, gopt, xopt = Polak_Ribiere(OracleDG, x0)

# Newton à pas fixe
# print()
# print("ALGORITHME DE NEWTON A PAS FIXE - DUAL")
# copt, gopt, xopt = Newton_F(OracleDH, x0)

# Newton à pas variable
# print()
# print("ALGORITHME DE NEWTON A PAS VARIABLE - DUAL")
# copt, gopt, xopt = Newton(OracleDH, x0)

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
