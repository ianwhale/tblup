import abc
import numpy as np
from copy import deepcopy
from tblup.evaluator import BlupParallelEvaluator


def get_local_search(args, population):
    """
    Returns the local search class specified by the
    :param args: argparse.Namespace
    :param population: tblup.Population
    :return: tblup.LocalSearch object
    """
    if args.local_search == args.LOCAL_SEARCH_KNOCKOUT:
        return KnockoutLocalSearch(population)

    raise NotImplementedError("Local search method {} not implemented.".format(args.local_search))


class LocalSearch(abc.ABC):
    """
    Base class for local search classes.
    """
    def __init__(self, population):
        """
        Constructor.
        :param population: tblup.Population, final popuation.
        """
        self.population = population

    @abc.abstractmethod
    def search(self):
        raise NotImplementedError()


class KnockoutLocalSearch(LocalSearch):
    """
    Knockout local search. Removes indices if they increase fitness.
    """
    def __init__(self, population):
        """
        Constructor.
        :param population: tblup.Population, final population.
        """
        super(KnockoutLocalSearch, self).__init__(population)

        assert issubclass(population.evaluator.__class__,
                          BlupParallelEvaluator), "Knockout only implemented for BLUP regressors."

    def search(self):
        """
        Execute local search.
        :return: tuple, (np.array, float), (knocked out genome, fitness value)
        """
        evaluator = self.population.evaluator
        best = deepcopy(max(self.population, key=lambda individual: individual.fitness))
        genome = self.population.evaluator.snp_remover.combine_with_removed(best.genome)
        best_fitness = best.fitness
        data, labels = np.load(self.population.evaluator.data_path), np.load(self.population.evaluator.labels_path)

        mask = np.ones(len(genome), dtype=bool)
        for i in range(len(genome)):
            mask[i] = False

            fitness = evaluator.blup(genome[mask], evaluator.training_indices, evaluator.validation_indices,
                                     data, labels, evaluator.h2)

            if fitness > best_fitness:
                # Keep the index masked.
                best_fitness = fitness

            else:
                # Change the index back, we don't want to exclude this one.
                mask[i] = True

        return genome[mask], best_fitness
