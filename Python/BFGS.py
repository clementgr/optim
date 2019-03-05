#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
from time import process_time

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode BFGS Inverse                                              #
#                                                                           #
#############################################################################

from Visualg import Visualg
from Wolfe_Skel import Wolfe

def BFGS(Oracle, x0):

    ##### Initialisation des variables

    iter_max = 10000
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []
    
    Id = np.identity(x0.size)

    time_start = process_time()

    
    ##### Premiere iteration
    
    # Point de depart
    xp = x0
    
    # Valeur du critere et du gradient
    ind = 4
    critere, gradient_p, _ = Oracle(xp, ind)
    
    # Valeur de l'approximation de l'inverse de la Hessienne
    Wp = Id
    
    # Direction de descente
    D = -gradient_p
    
    # Pas du gradient par recherche linéaire
    gradient_step = 1 # Trouver un meilleur coefficient
    gradient_step, _ = Wolfe(gradient_step, xp, D, Oracle)
    
    # Point suivant
    xn = xp + (gradient_step*D)
    
    # Evolution du gradient, du pas, et du critere
    gradient_norm_list.append(norm(gradient_p))
    gradient_step_list.append(gradient_step)
    critere_list.append(critere)

    ##### Boucle sur les iterations

    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient_n, _ = Oracle(xn, ind)

        # Test de convergence
        gradient_norm = norm(gradient_n)
        if gradient_norm <= threshold:
            break
            
        # Valeur de l'approximation de l'inverse de la Hessienne
        delta_x = xn - xp
        delta_g = gradient_n - gradient_p
        delta_W = np.matmul(delta_x, np.transpose(delta_g))/np.dot(delta_x, delta_g)
        gradient_step = 1 # Trouver un meilleur coefficient
        gradient_step, _ = Wolfe(gradient_step, xn, D, Oracle)

        # Direction de descente
        D = -np.matmul(Wn, gradient_n)

        # Pas du gradient par recherche linéaire
        gradient_step = 1 # Trouver un meilleur coefficient
        gradient_step, _ = Wolfe(gradient_step, xn, D, Oracle)
        
        # Mise a jour des variables
        xp = xn
        gradient_p = gradient_n
        Wp = Wn
        xn = xp + (gradient_step*D)
        
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

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
