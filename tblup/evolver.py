import os
import abc
import csv
import random
import numpy as np
from math import ceil
from copy import deepcopy
from numpy.random import normal
from scipy.stats import cauchy
from tblup import exclusive_randrange


def get_evolver(args):
    """
    Gets the evolver type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.Evolver
    """
    if args.de_strategy == "de_rand_1":
        return DERandOneEvolver(args.dimensionality, args.crossover_rate, args.mutation_intensity, args.clip)

    if args.de_strategy == "de_currenttobest_1":
        return DECurrentToBestOneEvolver(args.dimensionality, args.crossover_rate, args.mutation_intensity, args.clip)

    if args.de_strategy == "sade":
        return SaDE(args.dimensionality, args.clip)

    if args.de_strategy == "mde_pbx":
        return MDE_pBX(args.dimensionality, args.generations, args.clip)

    raise NotImplementedError("Evolver with config option {} is not implemented.".format(args.de_strategy))


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
    def __init__(self, dimensionality, crossover_rate, mutation_intensity, clip=True):
        """
        :param dimensionality: int, dimensionality of the problem.
        :param crossover_rate: float, [0, 1], probability to crossover.
        :param mutation_intensity: float, (0, inf), how much to mutate the candidate by.
        :param clip: bool, true to clip indices at [0, dimensionality).
        """
        self.dimensionality = dimensionality
        self.crossover_rate = crossover_rate
        self.mutation_intensity = mutation_intensity
        self.clip = clip

    @staticmethod
    def de_rand_one(population, mi, cr, dimensionality, parent_idx, clip=True):
        """
        :param population: tblup.Population, current population.
        :param mi: float, mutation intensity (also known as F).
        :param cr: float, crossover rate.
        :param dimensionality: int, dimensionality of the problem.
        :param parent_idx: int, index of parent.
        :param clip: bool, true to clip indices at [0, dimensionality).
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
                if clip:
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
            next_pop.append(self.de_rand_one(population, mi, self.crossover_rate, self.dimensionality, i, self.clip))

        return next_pop


class DECurrentToBestOneEvolver(Evolver):
    """
    DE/Current to best/1 strategy.
    Mutant vector V_i described by, for random vectors X_a and X_b, and target (parent) X_i,
        V_i = X_i + F * (X_{best} - X_i)
    Where X_{best} is the current best
    """
    def __init__(self, dimensionality, crossover_rate, mutation_intensity, clip=True):
        """
        :param dimensionality: int, dimensionality of the problem.
        :param crossover_rate: float, [0, 1], probability to crossover.
        :param mutation_intensity: float, (0, inf), how much to mutate the candidate by.
        """
        self.dimensionality = dimensionality
        self.crossover_rate = crossover_rate
        self.mutation_intensity = mutation_intensity
        self.clip = clip

    @staticmethod
    def de_currenttobest_one(population, mi, cr, dimensionality, parent_idx, best=None, clip=True):
        """
        Actual DE step.
        :param population: tblup.Population, current population.
        :param mi: float, mutation intensity (also known as F).
        :param cr: float, crossover rate.
        :param dimensionality: int, dimensionality of the problem.
        :param parent_idx: int, index of parent.
        :param best: None | tblup.Individual, if not provided, will be determined.
        :param clip: bool, true to clip indices at [0, dimensionality).
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
                if clip:
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
            next_pop.append(self.de_currenttobest_one(population,
                                                      mi,
                                                      self.crossover_rate,
                                                      self.dimensionality,
                                                      i,
                                                      best=best))

        return next_pop


class DECurrentToGrBestOneEvolver(Evolver):
    """
    DE/current-to-gr_best/1 strategy.
    Similar to DE/current-to-best/1, but uses a a random choice among the q% best vectors.
    """
    def __init__(self, dimensionality, crossover_rate, mutation_factor, q=0.15):
        """
        Constructor.
        :param dimensionality: int, dimensionality of the problem.
        :param crossover_rate: float, rate of crossover (Cr).
        :param mutation_factor: float, mutation factor (F).
        """
        self.dimensionality = dimensionality
        self.crossover_rate = crossover_rate
        self.mutation_factor = mutation_factor
        self.q = q

    @staticmethod
    def get_q_best(population, q):
        """
        Get the indices of the q best individuals.
        :param population: tblup.Population, current population.
        :param q: float, top q percent of individuals will be returned.
        :return: list, list of tblup.Individuals.
        """
        assert 0 < q <= 1, "q should be in (0, 1]."

        n = int(len(population) * q)
        return np.argsort([indiv.fitness for indiv in population])[-n:]

    def evolve(self, population):
        next_pop = []

        qbest = self.get_q_best(population, self.q)

        for i in range(len(population)):
            best = population[np.asscalar(np.random.choice(qbest, 1))]

            next_pop.append(DECurrentToBestOneEvolver.de_currenttobest_one(population,
                            self.crossover_rate,
                            self.mutation_factor,
                            self.dimensionality,
                            i,
                            best=best))

        return next_pop


class AdaptiveEvolver(Evolver):
    """
    Base class for adaptive evolver, does book keeping for "successful" parameters. In other words, parameters that
    produce candidate vectors that enter the next population.
    """
    def __init__(self):
        self.successful_fs = []  # Successful mutation factor values.
        self.successful_crs = []  # Successful crossover rate values.
        self.previous_pop_uids = None  # Previous population's unique ids.

        self.crs = []  # Crossover rates for the current generation.
        self.fs = []  # Mutation factors for the current generation.

    def should_report(self):
        """By default, aways report statistics about adapting parameters."""
        return True

    def report(self, population):
        """Report statistics about adaptive evolver."""
        directory = os.path.dirname(population.monitor.results_file)
        params_file, ext = os.path.splitext(os.path.basename(population.monitor.results_file))
        params_file = os.path.join(directory, params_file + "_params" + ext)

        if population.generation == 1:
            # Write header.
            with open(params_file, "w") as f:
                csv.writer(f).writerow(self.get_header())

        with open(params_file, "a") as f:
            csv.writer(f).writerow(self.get_params_row())

    @abc.abstractmethod
    def get_header(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params_row(self):
        raise NotImplementedError()

    def evolve(self, population):
        """
        Create the next population. See original paper for more details not described with inline comments.
        :param population: tblup.Population
        """
        if self.should_report():
            self.report(population)

        if self.previous_pop_uids is None:  # On the initial generation, set the uids before logic checks.
            self.previous_pop_uids = [individual.uid for individual in population]

        # Count the number of success and failures that each method had.
        self.count_outcomes(population)

        if self.should_regenerate_crs(population):
            self.regenerate_crs(population)

        if self.should_regenerate_fs(population):
            self.regenerate_fs(population)

        self.previous_pop_uids = [individual.uid for individual in population]

    def count_outcomes(self, population):
        """
        Count the successful mutation factors and crossover rates.
        :param population: tblup.Population, current population.
        """
        # We know the UIDs of the previous population, and want to know which strategies produced new individuals.
        for i, [previous_uid, current_individual] in enumerate(zip(self.previous_pop_uids, population)):
            if previous_uid != current_individual.uid:
                # A successful parameter, record it as such.
                self.successful_crs.append(self.crs[i])
                self.successful_fs.append(self.fs[i])

    @abc.abstractmethod
    def should_regenerate_crs(self, population):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_cr(self):
        raise NotImplementedError()

    def regenerate_crs(self, population):
        """
        Create new crossover rates.
        :param population: tblup.Population, current population.
        :return: list, list of floats
        """
        self.crs = [self.generate_cr() for _ in range(len(population))]

    @abc.abstractmethod
    def should_regenerate_fs(self, population):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_f(self):
        raise NotImplementedError()

    def regenerate_fs(self, population):
        """
        Create new mutation factors.
        :param population: tblup.Population, current population.
        :return: list, list of floats
        """
        self.fs = [self.generate_f() for _ in range(len(population))]


class SaDE(AdaptiveEvolver):
    """
    Self-adaptive Differential Evolution.
    Described with detail in:
    Qin, A. K. and Suganthan P. N. (2005) Self-adaptive Differential Evolution Algorithm for Numerical Optimization.
    2005 IEEE Congress on Evolutionary Computation.
    """

    f_m = 0.5  # Constant mean for mutation factor normal distribution.
    f_std = 0.3  # Constant standard deviation for mutation factor normal distribution.
    cr_std = 0.1  # Constant standard deviation for crossover normal distribution.
    recalculate_mean_interval = 25  # Recalculate cr_m at generations that are multiples of this constant.
    regenerate_crs_interval = 5  # Regenerate new crossover rates at generations that are multiples of this constant.
    initial_learning_period = 50  # Learn ns and nf parameters for this many initial generations, then reset them.

    header = ["p", "cr_m"]  # Header to use in the parameter reporting file.

    def __init__(self, dimensionality, clip=True):
        """
        Constructor.
        :param dimensionality: int, dimensionality of the problem.
        :param clip: bool, true to clip indices at [0, dimensionality).
        """
        super(SaDE, self).__init__()
        
        self.dimensionality = dimensionality
        self.clip = clip
        self.cr_m = 0.5  # Initial crossover rate mean.
        self.p = 0.5  # Initial probability of using strategy 1, else use strategy 2.

        self.strategy_one_indices = set()  # Indices in the population that generated a candidate with strategy 1.

        # Number of successes/failures method 1 or 2 has.
        self.ns_1, self.ns_2, self.nf_1, self.nf_2 = 0, 0, 0, 0

    def get_header(self):
        """Header for parameter csv output."""
        return ["cr_m", "p"]

    def get_params_row(self):
        """Row for parameter csv output."""
        return [self.cr_m, self.p]

    def should_regenerate_crs(self, generation):
        """Should we regenerate the crossover rates?"""
        return len(self.crs) == 0 or generation % self.regenerate_crs_interval == 0

    def should_recalculate_cr_m(self, generation):
        """Should we recalculate the mean of the crossover rates?"""
        return generation != 0 and generation % self.recalculate_mean_interval == 0

    def recalculate_cr_m(self):
        """Recalculate the mean of the crossover rates."""
        if len(self.successful_crs) == 0:
            return

        self.cr_m = np.mean(self.successful_crs)

    def generate_f(self):
        """Generate a mutation intensity (F), where F follows N(mi_m, mi_std), and F in [0, 2]."""
        return np.clip(normal(self.f_m, self.f_std), 0, 2)

    def generate_cr(self):
        """Generate a crossover rate, where CR follows N(cr_m, cr_std), and is in [0, 1]."""
        return np.clip(normal(self.cr_m, self.cr_std), 0, 1)

    def should_regenerate_fs(self, population):
        """SaDE does not generate an F value for each individual, only once per generation."""
        return False

    def recalculate_p(self, population):
        """Recalculate the probability of using strategy 1 if needed."""
        if population.generation < self.initial_learning_period:
            return  # Do not recalculate.

        self.p = (self.ns_1 * (self.ns_2 + self.nf_2)) \
            / (self.ns_2 * (self.ns_1 + self.nf_1) + self.ns_1 * (self.ns_2 + self.nf_2))

    def count_outcomes(self, population):
        """
        Override parent function.
        Count the number of successes and failures each strategy had.
        :param population: tblup.Population, current population.
        """
        super(SaDE, self).count_outcomes(population)

        if population.generation == self.initial_learning_period:
            # The learning period is over, we need to reset the counters.
            self.ns_1, self.ns_2, self.nf_1, self.nf_2 = 0, 0, 0, 0

        # We know the UIDs of the previous population, and want to know which strategies produced new individuals.
        for i, [previous_uid, current_individual] in enumerate(zip(self.previous_pop_uids, population)):
            if previous_uid == current_individual.uid:
                # This is a "failure" case, the strategy used did not create a new individual.
                if i in self.strategy_one_indices:
                    self.nf_1 += 1
                else:
                    self.nf_2 += 1

            else:
                # This is a "success" case, the strategy did create a new individual.
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
        if self.should_recalculate_cr_m(population.generation):
            self.recalculate_cr_m()

        super(SaDE, self).evolve(population)

        self.recalculate_p(population)

        # Same mutation intensity (a.k.a. F) used for each individual.
        f = self.generate_f()

        next_pop = []
        self.strategy_one_indices = set()

        best = max(population, key=lambda indv: indv.fitness)
        for i in range(len(population)):
            if random.random() < self.p:
                # Do strategy 1, i.e. DE/rand/1
                self.strategy_one_indices.add(i)
                indv = DERandOneEvolver.de_rand_one(population, f, self.crs[i], self.dimensionality, i, clip=self.clip)

            else:
                # Do strategy 2, i.e. DE/current to best/2
                indv = DECurrentToBestOneEvolver.de_currenttobest_one(population, f, self.crs[i],
                                                                      self.dimensionality, i, best=best, clip=self.clip)

            next_pop.append(indv)

        return next_pop


class MDE_pBX(AdaptiveEvolver):
    """
    MDE_pBX strategy.
    Described with detail in:
    Islam, Sk. M., Das, S., Ghosh, S. Roy, S., and Suganthan P. N. (2012) An Adaptive Differential Evolution Algorithm
    With Novel Mutation and Crossover Strategies for Global Numerical Optimization. IEEE Transactions on Systems, Man,
    and Cybernetics—Part B: Cybernetics.
    """

    f_scale = 0.1  # Scale factor for mutation factor cauchy distribution.
    cr_std = 0.1  # Standard deviation for crossover rate normal distribution.
    group_q = 0.15  # q parameter for the q best vectors used in the current-to-gr_best/1 scheme

    def __init__(self, dimensionality, generations, clip=True):
        """
        Constructor.
        :param dimensionality: int, dimensionality of the problem.
        :param generations: int, maximum number of generations.
        :param clip: bool, true to clip indices at [0, dimensionality).
        """
        super(MDE_pBX, self).__init__()

        self.dimensionality = dimensionality
        self.clip = clip
        self.g_max = generations

        self.cr_m = 0.6  # Initial crossover mean.
        self.f_m = 0.5  # Initial mutation factor mean.

        self.p = None

    def get_header(self):
        """Header for parameter csv output."""
        return ["cr_m", "f_m"]

    def get_params_row(self):
        """Row for parameter csv output."""
        return [self.cr_m, self.f_m]

    def should_regenerate_fs(self, population):
        """Always regenerate mutation values."""
        return True

    def should_regenerate_crs(self, population):
        """Always regenerate crossover rates."""
        return True

    def generate_cr(self):
        """Generate crossover rate from normal distribution. Regenerate until cr in [0, 1]."""
        cr = normal(self.cr_m, self.cr_std)
        while cr < 0 or cr > 1:
            cr = normal(self.cr_m, self.cr_std)
        return cr

    def generate_f(self):
        """Generate mutation factor from Cauchy distribution. Regenerate until f in [0, 1]."""
        f = cauchy.rvs(loc=self.f_m, scale=self.f_scale)
        while f < 0 or f > 1:
            f = cauchy.rvs(loc=self.f_m, scale=self.f_scale)
        return f

    def recalculate_cr_m(self):
        """Recalculate cr_m with formula (12a) in Islam et al."""
        if len(self.successful_crs) == 0:
            return

        w_cr = self.get_weight_factor(0.9, 0.1)
        self.cr_m = w_cr * self.cr_m + (1 - w_cr) * self.mean_pow(self.successful_crs)
        self.successful_crs = []  # Reset record of successful crossover rates.

    def recalculate_f_m(self):
        """Recalculate f_m with formula (9a) in Islam et al."""
        if len(self.successful_fs) == 0:
            return

        w_f = self.get_weight_factor(0.8, 0.2)
        self.f_m = w_f * self.f_m + (1 - w_f) * self.mean_pow(self.successful_fs)
        self.successful_fs = []  # Reset record of succesful mutation factors.

    def recalculate_p(self, population):
        """Recalculate p with formula (7) in Islam et al."""
        self.p = ceil((len(population) / 2) * (1 - (population.generation / self.g_max)))

    @staticmethod
    def mean_pow(vals, n=1.5):
        """
        See formula (10) in Islam et al.
        Mean_pow is defined here a little differently in the original paper. However since we know that all the values
        in the sum will be positive (they are crossover rates or mutation factors) and n is positive, we can simplify.
        :param vals: list, list of postive numbers.
        :param n: float, parameter to the power mean.
        :return:
        """
        assert n > 0, "n must be a positive number."

        d = pow(len(vals), -n)
        return sum(vals) / d

    @staticmethod
    def get_weight_factor(p, q):
        """
        See formula (9b) and (12b) in Islam et al.
        Weight factor function.
        :param p: float, additive parameter.
        :param q: float, multiplicative parameter.
        :return: float, weight factor.
        """
        return p + q * random.random()  # p + q * rand(0,1)

    def evolve(self, population):
        """
        Create the next population. See original paper for more details not described with inline comments.
        :param population: tblup.Population
        :return: list, list of individuals.
        """
        self.recalculate_cr_m()
        self.recalculate_f_m()
        self.recalculate_p(population)

        super(MDE_pBX, self).evolve(population)

        sorted_indices = np.argsort([indiv.fitness for indiv in population])

        q_best = sorted_indices[-int(len(population) * self.group_q):]
        p_best = sorted_indices[-self.p:]

        next_pop = []
        for i in range(len(population)):
            gr_choice = population[np.asscalar(np.random.choice(q_best, 1))]
            parent_idx = np.asscalar(np.random.choice(p_best, 1))

            indiv = DECurrentToBestOneEvolver.de_currenttobest_one(population,
                                                                   self.fs[i],
                                                                   self.crs[i],
                                                                   self.dimensionality,
                                                                   parent_idx,
                                                                   best=gr_choice,
                                                                   clip=self.clip)

            next_pop.append(indiv)

        return next_pop
