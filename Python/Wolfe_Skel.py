#!/usr/bin/python

import numpy as np
from numpy import dot
from OraclePG import *

########################################################################
#                                                                      #
#          RECHERCHE LINEAIRE SUIVANT LES CONDITIONS DE WOLFE          #
#                                                                      #
#          Algorithme de Fletcher-Lemarechal                           #
#                                                                      #
########################################################################

#  Arguments en entree
#
#    alpha  : valeur initiale du pas
#    x      : valeur initiale des variables
#    D      : direction de descente
#    Oracle : nom de la fonction Oracle
#
#  Arguments en sortie
#
#    alphan : valeur du pas apres recherche lineaire
#    ok     : indicateur de reussite de la recherche
#             = 1 : conditions de Wolfe verifiees
#             = 2 : indistinguabilite des iteres

def Wolfe(alpha, x, D, Oracle):

    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1 # c.f. methodes.pdf
    omega_2 = 0.9 # c.f. methodes.pdf

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    ind = 4

    ##### Algorithme de Fletcher-Lemarechal

    # Appel de l'oracle au point initial
    critere_x, gradient_x, _ = Oracle(x, ind)

    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x

    # Boucle de calcul du pas
    while ok == 0:

        # xn represente le point pour la valeur courante du pas,
        # xp represente le point pour la valeur precedente du pas.
        xp = xn
        xn = x + alpha_n*D

        # Appel de l'oracle au point courant
        critere_xn, gradient_xn, _ = Oracle(xn, ind)

        # Calcul des conditions de Wolfe
        cond1 = (omega_1*alpha_n*np.dot(gradient_x, D) - critere_xn + critere_x >= 0)
        cond2 = (np.dot(gradient_xn, D) - omega_2*np.dot(gradient_x, D) >= 0)

        # Test des conditions de Wolfe
        # - si la condition 1 n'est pas verifiée
        if (not cond1):
            alpha_max = alpha_n
            alpha_n = (alpha_min + alpha_max)/2
        # - si la condition 1 est vérifiée et que la condition 2 ne l'est pas
        elif (not cond2):
            alpha_min = alpha_n
            if alpha_max == np.inf:
                alpha_n = 2*alpha_min
            else:
                alpha_n = (alpha_min + alpha_max)/2
        # - si les deux conditions de Wolfe sont verifiées
        else:
            ok = 1

        # Test d'indistinguabilite
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok


def Wolfe_7(alpha, x, D, Oracle):

    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1 # c.f. methodes.pdf
    omega_2 = 0.9 # c.f. methodes.pdf

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    ind = 7

    ##### Algorithme de Fletcher-Lemarechal

    # Appel de l'oracle au point initial
    critere_x, gradient_x, _, _ = Oracle(x, ind)

    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x

    # Boucle de calcul du pas
    while ok == 0:

        # xn represente le point pour la valeur courante du pas,
        # xp represente le point pour la valeur precedente du pas.
        xp = xn
        xn = x + alpha_n*D

        # Appel de l'oracle au point courant
        critere_xn, gradient_xn, _, _ = Oracle(xn, ind)

        # Calcul des conditions de Wolfe
        cond1 = (omega_1*alpha_n*np.dot(gradient_x, D) - critere_xn + critere_x >= 0)
        cond2 = (np.dot(gradient_xn, D) - omega_2*np.dot(gradient_x, D) >= 0)

        # Test des conditions de Wolfe
        # - si la condition 1 n'est pas verifiée
        if (not cond1):
            alpha_max = alpha_n
            alpha_n = (alpha_min + alpha_max)/2
        # - si la condition 1 est vérifiée et que la condition 2 ne l'est pas
        elif (not cond2):
            alpha_min = alpha_n
            if alpha_max == np.inf:
                alpha_n = 2*alpha_min
            else:
                alpha_n = (alpha_min + alpha_max)/2
        # - si les deux conditions de Wolfe sont verifiées
        else:
            ok = 1

        # Test d'indistinguabilite
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok
