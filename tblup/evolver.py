import abc
import random
import numpy as np
from copy import deepcopy
from numpy.random import normal
from tblup import exclusive_randrange


def get_evolver(args):
    """
    Gets the evolver type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.Evolver
    """
    if args.de_strategy == "de_rand_1":
        return DERandOneEvolver(args.dimensionality, args.crossover_rate, args.mutation_intensity)

    if args.de_strategy == "de_currenttobest_2":
        return DECurrentToBestTwoEvolver(args.dimensionality, args.crossover_rate, args.mutation_intensity)

    if args.de_strategy == "sade":
        return SaDE(args.dimensionality)

    raise NotImplementedError("Evolver with config description {} is not implemented.".format(args.de_strategy))


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
    def de_rand_one(population, mi, cr, dimensionality, parent_idx):
        """
        :param population: tblup.Population, current population.
        :param mi: float, mutation intensity (also known as F).
        :param cr: float, crossover rate.
        :param dimensionality: int, dimensionality of the problem.
        :param parent_idx: int, index of parent.
        :return: tblup.Individual, the candidate individual.
        """
        pop_len = len(population)

        parent = population[parent_idx]

        # Get mutators.
        a = exclusive_randrange(0, pop_len, [parent_idx])
        b = exclusive_randrange(0, pop_len, [parent_idx, a])
        c = exclusive_randrange(0, pop_len, [parent_idx, a, b])

        a, b, c = population[a], population[b], population[c]

        # Create candidate from mutators and parent.
        candidate = deepcopy(parent)

        fixed = random.randrange(0, len(parent))

        for j in range(len(parent)):
            if j == fixed or random.random() < cr:
                mutant = round(a[j] + mi * (b[j] - c[j]))  # Round for integer solutions only.

                # Bound solutions.
                mutant = np.clip(mutant, 0, dimensionality - 1)
                candidate[j] = int(mutant)

        return candidate

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

        next_pop = []
        for i in range(len(population)):
            next_pop.append(self.de_rand_one(population, mi, self.crossover_rate, self.dimensionality, i))

        return next_pop


class DECurrentToBestTwoEvolver(Evolver):
    """
    DE/Current to best/2 strategy.
    Mutant vector V_i described by, for random vectors X_a and X_b, and target (parent) X_i,
        V_i = X_i + F * (X_{best} - X_i)
    Where X_{best} is the current best
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
    def de_currenttobest_two(population, mi, cr, dimensionality, parent_idx, best=None):
        """
        Actual DE step.
        :param population: tblup.Population, current population.
        :param mi: float, mutation intensity (also known as F).
        :param cr: float, crossover rate.
        :param dimensionality: int, dimensionality of the problem.
        :param parent_idx: int, index of parent.
        :param best: None | tblup.Individual, if not provided, will be determined.
        :return: tblup.Individual, the candidate individual.
        """
        pop_len = len(population)

        if best is None:
            best = max(population, key=lambda individual: individual.fitness)

        parent = population[parent_idx]

        # Get mutators.
        a = exclusive_randrange(0, pop_len, [parent_idx])
        b = exclusive_randrange(0, pop_len, [parent_idx, a])

        a, b = population[a], population[b]

        # Create candidate from mutators and parent.
        candidate = deepcopy(parent)

        fixed = random.randrange(0, len(parent))

        for j in range(len(parent)):
            if j == fixed or random.random() < cr:
                mutant = round(candidate[j] + mi * (best[j] - candidate[j]) + mi * (a[j] - b[j]))

                # Bound solutions.
                mutant = np.clip(mutant, 0, dimensionality - 1)
                candidate[j] = int(mutant)

        return candidate

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

        next_pop = []

        best = max(population, key=lambda individual: individual.fitness)
        for i in range(len(population)):
            next_pop.append(self.de_currenttobest_two(population,
                                                      mi,
                                                      self.crossover_rate,
                                                      self.dimensionality,
                                                      i,
                                                      best=best))

        return next_pop


class SaDE(Evolver):
    """
    Self-adaptive Differential Evolution.
    Described with detail in:
    Qin, A. K. and Suganthan P. N. (2005) Self-adaptive Differential Evolution Algorithm for Numerical Optimization.
    2005 IEEE Congress on Evolutionary Computation.
    """

    mi_m = 0.5  # Constant mean for mutation intensity normal distribution.
    mi_std = 0.3  # Constant standard deviation for mutation intensity normal distribution.
    cr_std = 0.1  # Constant standard deviation for crossover normal distribution.
    recalculate_mean_interval = 25  # Recalculate cr_m at generations that are multiples of this constant.
    regenerate_crs_interval = 5  # Regenerate new crossover rates at generations that are multiples of this constant.
    initial_learning_period = 50  # Learn ns and nf parameters for this many initial generations, then reset them.

    def __init__(self, dimensionality):
        """
        Constructor.
        :param dimensionality: int, dimensionality of the problem.
        """
        self.dimensionality = dimensionality
        self.previous_population_uids = None  # The previous population's globally unique ids.
        self.cr_m = 0.5  # Initial crossover rate mean.
        self.p = 0.5  # Initial probability of using strategy 1, else use strategy 2.
        self.crs = []
        self.successful_crs = {}  # Crossover rates that created an individual that entered the population.

        self.strategy_one_indices = None  # Indices in the population that generated a candidate with strategy 1.

        # Number of successes/failures method 1 or 2 has.
        self.ns_1, self.ns_2, self.nf_1, self.nf_2 = 0, 0, 0, 0

    def should_regenerate_crs(self, generation):
        """Should we regenerate the crossover rates?"""
        return len(self.crs) == 0 or generation % self.regenerate_crs_interval == 0

    def regenerate_crs(self, population):
        """
        Create new crossover rates.
        :param population: tblup.Population, current population.
        :return: list, list of floats
        """
        self.crs = [self.generate_cr() for _ in range(len(population))]

    def should_recalculate_cr_m(self, generation):
        """Should we recalculate the mean of the crossover rates?"""
        return generation != 0 and generation % self.recalculate_mean_interval == 0

    def recalculate_cr_m(self):
        """Recalculate the mean of the crossover rates."""
        if len(self.successful_crs) == 0:
            return

        self.cr_m = np.mean(list(self.successful_crs.keys()))
        self.successful_crs = {}  # Reset the record of successful crossover values.

    def generate_mi(self):
        """Generate a mutation intensity (F), where F follows N(mi_m, mi_std), and F in [0, 2]."""
        return np.clip(normal(self.mi_m, self.mi_std), 0, 2)

    def generate_cr(self):
        """Generate a crossover rate, where CR follows N(cr_m, cr_std), and is in [0, 1]."""
        return np.clip(normal(self.cr_m, self.cr_std), 0, 1)

    def recalculate_p(self, population):
        """Recalculate the probability of using strategy 1 if needed."""
        if population.generation < self.initial_learning_period:
            return  # Do not recalculate.

        self.p = (self.ns_1 * (self.ns_2 + self.nf_2)) \
                 / (self.ns_2 * (self.ns_1 + self.nf_1) + self.ns_1 * (self.ns_2 + self.nf_2))

    def count_outcomes(self, population):
        """
        Count the number of successes and failures each strategy had.
        :param population: tblup.Population, current population.
        """
        if self.strategy_one_indices is None:
            return  # Do nothing, we are in an initialization state.

        if population.generation == self.initial_learning_period:
            # The learning period is over, we need to reset the counters.
            self.ns_1, self.ns_2, self.nf_1, self.nf_2 = 0, 0, 0, 0

        # We know the UIDs of the previous population, and want to know which strategies produced new individuals.
        for i, [previous_uid, current_individual] in enumerate(zip(self.previous_population_uids, population)):
            if previous_uid == current_individual.uid:
                # This is a "failure" case, the strategy used did not create a new individual.
                if i in self.strategy_one_indices:
                    self.nf_1 += 1

                else:
                    self.nf_2 += 1

            else:
                # This is a "success" case, the strategy did create a new individual.
                self.successful_crs[self.crs[i]] = True
                if i in self.strategy_one_indices:
                    self.ns_1 += 1

                else:
                    self.ns_2 += 1

    def evolve(self, population):
        """
        Create the next population. See original paper for more details not described with inline comments.
        :param population: tblup.Population
        :return: list, list of individuals.
        """
        if self.previous_population_uids is None:
            self.previous_population_uids = [individual.uid for individual in population]

        # Count the number of success and failures that each method had.
        self.count_outcomes(population)
        self.recalculate_p(population)

        if self.should_recalculate_cr_m(population.generation):
            self.recalculate_cr_m()

        if self.should_regenerate_crs(population.generation):
            self.regenerate_crs(population)

        self.previous_population_uids = [individual.uid for individual in population]

        # Same mutation intensity (a.k.a. F) used for each individual.
        mi = self.generate_mi()

        next_pop = []
        self.strategy_one_indices = set()

        best = max(population, key=lambda indv: indv.fitness)
        for i in range(len(population)):
            if random.random() < self.p:
                # Do strategy 1, i.e. DE/rand/1
                self.strategy_one_indices.add(i)
                indv = DERandOneEvolver.de_rand_one(population, mi, self.crs[i], self.dimensionality, i)

            else:
                # Do strategy 2, i.e. DE/current to best/2
                indv = DECurrentToBestTwoEvolver.de_currenttobest_two(population, mi, self.crs[i],
                                                                      self.dimensionality, i, best=best)

            next_pop.append(indv)

        return next_pop
