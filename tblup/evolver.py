import abc
import random
from copy import deepcopy
from tblup.utils import exclusive_randrange


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

    def evolve(self, population):
        """
        Create the next population.
        :param population: tblup.Population, current population object.
        :return: list, list of tblup.Individual.
        """
        next_pop = []

        pop_len = len(population)

        for i in range(pop_len):
            parent = population[i]

            # Get mutators.
            a, b, c = exclusive_randrange(0, pop_len, i), \
                exclusive_randrange(0, pop_len, i), \
                exclusive_randrange(0, pop_len, i)

            # Create candidate from mutators and parent.
            candidate = deepcopy(parent)

            fixed = random.randrange(0, len(parent))

            for j in range(len(parent)):
                if j == fixed or random.random() < self.crossover_rate:
                    mutant = round(a[j] + self.mutation_intensity * (b[j] - c[j]))  # Round for integer solutions only.

                    # Bound solutions.
                    if mutant >= self.dimensionality:
                        mutant = self.dimensionality - 1

                    elif mutant < 0:
                        mutant = 0

                    candidate[j] = mutant

            next_pop.append(candidate)

        return next_pop
