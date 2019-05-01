#!/usr/bin/python

import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.linalg import inv
from time import process_time

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode de Newton a pas variable                                  #
#                                                                           #
#############################################################################

from Visualg import Visualg
from Wolfe_Skel import WolfePH

def Newton(Oracle, x0):

    ##### Initialisation des variables

    iter_max = 100
    gradient_step = 1
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0

    ##### Boucle sur les iterations

    for k in range(iter_max):

        # Valeur du critere et du gradient
        ind = 7
        critere, gradient, hessien, _ = Oracle(x, ind)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente - 2 calculs possible : inversion de la hessienne ou système linéaire
        # D = - dot(inv(hessien), gradient)
        D = np.linalg.solve(hessien, -gradient)

        # Pas du gradient par recherche linéaire
        gradient_step = 1 # Trouver un meilleur coefficient
        gradient_step, _ = WolfePH(gradient_step, x, D, Oracle)

        # Mise a jour des variables
        x = x + (gradient_step*D)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

    ##### Resultats de l'optimisation

    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
