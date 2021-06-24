#from __future__ import print_function

import gzip
import random
# import time
import neat

'''
try:
    import cPickle as pickle  # pylint: disable=import-error
except ImportError:
    import pickle  # pylint: disable=import-error
'''
import pickle
# from neat.population import Population
from neat.reporting import BaseReporter


class CheckpointerBest(BaseReporter):
    """
    Clase que override los métodos de la clase neat.reporting.BaseReporter.BaseReporter
    BaseReporter es de hecho una clase abstracta simplemente proveyendo un conjunto de
    métodos dummy (no hacen nada, solo "pass") para ser overriden, que define la interfaz esperada por
    la clare ReporterSet
    """

    def __init__(self, filename_prefix='neat-checkpoint-'):

        self.filename_prefix = filename_prefix

        self.current_generation = None
        self.last_best_fitness = -1

    def start_generation(self, generation):
        self.current_generation = generation

    # def end_generation(self, config, population, species_set):

    def post_evaluate(self, config, population, species, best_genome):

        if best_genome is not None:
            if self.last_best_fitness < best_genome.fitness:
                self.last_best_fitness = best_genome.fitness
                self.save_checkpoint(config, population, species, self.current_generation)
        else:
            print("no best genome")

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.filename_prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod  # decoramos para poder llamarlo si necesidad de instanciar la clase primero
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return neat.Population(config, (population, species_set, generation))
