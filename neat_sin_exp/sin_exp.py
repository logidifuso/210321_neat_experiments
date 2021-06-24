import os,signal,sys
import shutil

import neat

import math
import numpy as np

import multiprocessing
from CheckpointerBest import CheckpointerBest

import common.visualize as vis
import common.utils as utils


# The current working directory
local_dir = os.path.dirname(__file__)
# local_dir = "./"                                    # Directoria actual

# out_dir = os.path.join(local_dir, 'checkpoints') #Original, lo reemplazo por:
# The directory to store outputs
outputs_dir = os.path.join(local_dir, 'outputs')
graphs_dir = os.path.join(outputs_dir, 'graphs')

# ===================================================================================
# Crea muestras del seno escogiendo 45 muestras de ángulo entre 0 y 360 al azar para
# usar en las evaluaciones del fitness
# ------------------------------------------------------------------------------------
# create full sin list 1 step degrees
degrees2radians = np.radians(np.arange(0,360,1))
# samples
sample_count = 45
xx = np.random.choice(degrees2radians,sample_count,replace=False)
yy = np.sin(xx)
# ====================================================================================

def eval_fitness(net):
    global gens   # TODO: se utiliza está variable global o no?

    error_sum = 0.0
    outputs = []
    accs = []

    def _imp():
        _fitness = 0
        for xi, xo in zip(xx, yy):
            output = net.activate([xi])
            xacc = 1 - abs(xo - output)
            _fitness += xacc

        _fitness = np.mean((_fitness / len(xx)))

        return _fitness

    fitness = (_imp()) * 100
    fitness = np.round(fitness, decimals=4)
    fitness = max(fitness, -1000.)

    return fitness


def eval_genomes_mp(genomes, config):
    net = neat.nn.FeedForwardNetwork.create(genomes, config)
    genomes.fitness = eval_fitness(net)
    return genomes.fitness


def eval_genomes_single(genomes, config):
    # single process
    for genome_id, genome in genomes:
        # net = RecurrentNet.create(genome, config,1)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net)


def createPoolAndConfig(config_file, checkpoint):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.

    if checkpoint is not None:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    return p, config


def run_experiment(config_file, checkpoint=None, mp=False):
    best_genome = None

    p, config = createPoolAndConfig(config_file, checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # TODO: Experimentos previos grabo un checkpoint sólo cada 5 (var de add reporter). Aquí igual??
    p.add_reporter(CheckpointerBest(filename_prefix="".join((outputs_dir, '/sin_exp-checkpoint-'))))
    pe = None

    # this part required to handle keyboard intterrupt correctly, and return population and config to evaluate test set.
    try:

        if mp:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes_mp)

            signal.signal(signal.SIGINT, original_sigint_handler)

            """set_trace()
            return"""
            best_genome = p.run(pe.evaluate, 113551)
        else:

            best_genome = p.run(eval_genomes_single, 1113551)
            # print('\nBest genome:\n{!s}'.format(best_genome))
    except:
        print("Stopping the Jobs. ", sys.exc_info())
        if mp:
            pe.pool.terminate()
            pe.pool.join()
            print("pool ok")

        return p, config

    return p, config


# TODO: Dónde y para qué se utiliza?
def evaluate_best(p, best_genome, config):

    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    accs = []
    for xi, xo in zip(xx, yy):
        output = net.activate([xi])
        xacc = 1 - (abs(xo-output))
        accs.append(xacc)

    print("\nmean acc {}\n".format(
        np.round(np.array(accs).mean() * 100., decimals=2)
    ))


if __name__ == '__main__':
    # Indica la ruta al archivo de configuración. Siempre que el archivo
    # de configuración se encuentre en el mismo folder el script se ejecutará
    # correctamente independientemente de cúal sea la carpeta de trabajo actual
    config_path = os.path.join(local_dir, 'sin_config.ini')

    # os.makedirs(out_dir, exist_ok=True)  # Crea carpeta de salida #Original. Reemplazado por:
    # Limpia los resultados de la ejecución anterior (si los hubiera) o inicia la carpeta a usar
    # para guardar los resultados
    utils.clear_output(graphs_dir)

    # Corre experimento. Para continuar a partir de un checkpont particular usa cp, eg. cp=1024
    cp = None
    if cp is not None:
        ret = run_experiment(config_path,
                             checkpoint="".join((outputs_dir, '/sin_exp-checkpoint-{}'.format(cp))),
                             mp=True)
    else:
        ret = run_experiment(config_path, mp=True)
    # TODO: Annadir los gráficos de la mejor red, etc..
########################################################################################################################

    # Run the experiment
    run_experiment(config_path, n_generations=35)


