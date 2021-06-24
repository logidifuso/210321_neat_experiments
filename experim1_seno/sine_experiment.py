#
# This file provides the source code of the Single-Pole balancing experiment using on NEAT-Python library
#

import os  # The Python standard library import
import neat  # The NEAT-Python library imports
import sine_mod as sin  # El simulador de la función seno
import random
# import sys
import matplotlib.pyplot as plt
import warnings
import numpy as np

import common.visualize as vis
import common.utils as utils


# sys.path.append("../common")
# import visualize  # The helper used to visualize experiment results
# import utils


# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'seno')


def plot_salida(net, view=False, filename='salida.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    n_eval_points = 90
    salida = np.ndarray([n_eval_points])
    seno = np.ndarray([n_eval_points])
    for i in range(n_eval_points):
        valor = i * (2 * np.pi / n_eval_points)
        # val = net.activate([valor])
        salida[i] = net.activate([valor])[0]
        seno[i] = (np.sin(valor)+1)/2

    x = range(n_eval_points)

    plt.plot(x, salida, 'b-', label="salida")
    plt.plot(x, seno, 'r-', label="seno")

    plt.title("Salida vs Valor exacto")
    plt.xlabel("x")
    plt.ylabel("Salida")
    plt.grid()

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = sin.eval_fitness(net)
        genome.fitness = fitness


def evaluate_best_net(net, config):
    """
    Pequenna función para evaluar el mejor genoma encontrado durante la
    ejecución del experimento

    :param net: genema del mejor individuo encontrado
    :param config: ruta al archivo de configuración del experimento
    :return: True en caso de éxito; False en caso contrario
    """

    fitness = sin.eval_fitness(net)
    if fitness < config.fitness_threshold:
        return False
    else:
        return True


def run_experiment(config_file, n_generations):
    """
    Corre el experimento usando los valores para los hiper-parámetros definidos
    en el archivo de configuración.

    Genera una representación del genoma ganador usando las funciones disponibles
    en el módulo Visualize del proyecto

    :param config_file: Ruta al archivo de configuración con los valores a usar en
    el experimento.
    :param n_generations:
    :return:
    """

    seed = 1559231615
    random.seed(seed)

    # Carga de la configuración
    configuracion = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Creacción de la población
    p = neat.Population(configuracion)

    # Crea un stdout reporter para mostrar el progreso en el terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/spb-neat-checkpoint-'))

    # Ejecución durante n_generations
    best_genome = p.run(eval_genomes, n=n_generations)

    # Muestra info del mejor genoma
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Comprobación de si el mejor genoma es un hit
    net = neat.nn.FeedForwardNetwork.create(best_genome, configuracion)
    print("\n\nRe-evaluación del mejor individuo")
    hit = evaluate_best_net(net, configuracion)
    if hit:
        print("ÉXITO!!!")
    else:
        print("FRACASO!!!")

    # Visualiza los resultados del experimento
    node_names = {-1: 'x', 0: 'output'}
    vis.draw_net(configuracion, best_genome, True, node_names=node_names, directory=out_dir, fmt='svg')
    vis.plot_stats_sine(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    vis.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))
    plot_salida(net, view=True, filename=os.path.join(out_dir, 'salida.svg'))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'sine_config.ini')

    # Limpia los resultados de la ejecución anterior (si los hubiera) o inicia la carpeta a usar
    # para guardar los resultados
    utils.clear_output(out_dir)

    # Run the experiment
    run_experiment(config_path, n_generations=35)
