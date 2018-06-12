import abc
import random
from copy import deepcopy
from tblup import exclusive_randrange


def get_evolver(args):
    """
    Gets the evolver type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.Evolver
    """
    if args.de_strategy == "de_rand_1":
        return DERandOneEvolver(args.dimensionality, args.crossover_rate, args.mutation_intensity)


class Evolver(abc.ABC):
    """
    Abstract base evolver base class.
    """
    @abc.abstractmethod
    def evolve(self, population):
        """
        :param population: tblup.Population
        """
        raise NotImplementedError()


class DERandOneEvolver(Evolver):
    """
    Standard differential evolution scheme.
    "DE/rand/1" mutation.
        - Get random all random mutators and create a candidate vector.
    """
    def __init__(self, dimensionality, crossover_rate, mutation_intensity):
        """
        :param dimensionality: int, dimensionality of the problem.
        :param crossover_rate: float, [0, 1], probability to crossover.
        :param mutation_intensity: float, (0, inf), how much to mutate the candidate by.
        """
        self.dimensionality = dimensionality
        self.crossover_rate = crossover_rate
        self.mutation_intensity = mutation_intensity

    @staticmethod
    def derandone(population, mi, cr, dimensionality):
        next_pop = []

        pop_len = len(population)

        for i in range(pop_len):
            parent = population[i]

            # Get mutators.
            a = exclusive_randrange(0, pop_len, [i])
            b = exclusive_randrange(0, pop_len, [i, a])
            c = exclusive_randrange(0, pop_len, [i, a, b])

            a, b, c = population[a], population[b], population[c]

            # Create candidate from mutators and parent.
            candidate = deepcopy(parent)

            fixed = random.randrange(0, len(parent))

            for j in range(len(parent)):
                if j == fixed or random.random() < cr:
                    mutant = round(a[j] + mi * (b[j] - c[j]))  # Round for integer solutions only.

                    # Bound solutions.
                    if mutant >= dimensionality:
                        mutant = dimensionality - 1

                    elif mutant < 0:
                        mutant = 0

                    candidate[j] = mutant

            next_pop.append(candidate)

        return next_pop

    def evolve(self, population):
        """
        Create the next population.
        :param population: tblup.Population, current population object.
        :return: list, list of tblup.Individual.
        """
        if population.generation % 5 == 0:
            mi = 5

        else:
            mi = self.mutation_intensity

        return self.derandone(population, mi, self.crossover_rate, self.dimensionality)


class SaDE(Evolver):
    """
    Self-adaptive Differential Evolution.
    The evolution scheme described by formulas (4) and (5) in:
    """
    pass