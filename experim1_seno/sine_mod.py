#
# Implementación de la evaluación del fitness para función seno
#
import numpy as np
import math as m


def eval_fitness(net):
    """
    Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
    :param net: The ANN of the phenotype to be evaluated
    :return fitness: El fitness se calcula como 1/e**(-MSE). De esta forma cuando
    el error cuadrático medio es 0 el fitness es 0 y conforme aumenta el fitness
    disminuye tendiendo a cero conforme el MSE tiende a infinito
    """

    n_eval_points = 90
    err = np.ndarray([n_eval_points])

    for i in range(n_eval_points):
        valor = i*(2*np.pi/n_eval_points)
        # Normalizo el seno entre 0 y 1 para que la red pueda predecirlo (neurona de
        # salida output en [0, 1]
        err[i] = ((np.sin(valor)+1)/2 - net.activate([valor]))**2

    mse = 1/n_eval_points * np.sum(err)
    fitness = m.exp(-mse)
    return fitness
