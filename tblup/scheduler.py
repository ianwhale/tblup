import abc
import random
import numpy as np
from math import log, floor
from tblup.utils import exclusive_randrange
from tblup.individual import IndexIndividual, RandomKeyIndividual


def get_scheduler(args):
    """
    Gets the scheduler type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.FeatureScheduler
    """
    complexifier = None

    if args.individual == args.INDIVIDUAL_TYPE_INDEX:
        complexifier = IndexComplexifier()

    elif args.individual == args.INDIVIDUAL_TYPE_RANDOM_KEYS:
        complexifier = RandomKeyComplexifier()

    if complexifier is None:
        raise NotImplementedError("Complexifier for individual {} is not implemented.".format(args.individual))

    if args.initial_features is None:
        return FeatureScheduler(args.initial_features, args.features, args.generations, complexifier)

    if args.feature_scheduling == args.FEATURE_SCHEDULING_STEPWISE:
        return StepwiseScheduler(args.initial_features, args.features, args.generations, complexifier)

    if args.feature_scheduling == args.FEATURE_SCHEDULING_ADAPTIVE:
        return AdaptiveScheduler(args.initial_features, args.features, args.generations, complexifier)


class FeatureScheduler(object):
    """
    Feature scheduler base class, doesn't do anything.
    """
    def __init__(self, initial_features, final_features, generations, complexifier):
        assert isinstance(complexifier, Complexifier)

        self.initial = initial_features
        self.final = final_features
        self.generations = generations
        self.complexifier = complexifier

    def should_step(self, population, generation):
        return False

    def step(self, population):
        pass


class StepwiseScheduler(FeatureScheduler):
    """
    Scheduler doubles the genome length at regular intervals throughout the search.
    """
    def __init__(self, initial_features, final_features, generations, complexifier):
        """
        Constructor.
        :param initial_features: int, starting number of features.
        :param final_features: int, ending number of features.
        :param generations: int, total number of generations.
        :param complexifier: tblup.Complexifier
        """
        super(StepwiseScheduler, self).__init__(initial_features, final_features, generations, complexifier)

        # Number of times we will double the genome length plus once more to
        # make genome length the correct final size.
        # 2^x * init = final  ==>  log_2(final / init) = x
        self.step_count = floor(log((final_features / initial_features), 2))

        # How often to step.
        self.step_interval = generations // (self.step_count + 1)

        step_intervals = []
        current = self.step_interval
        for _ in range(self.step_count):
            step_intervals.append(current)
            current += self.step_interval

        self.step_intervals = step_intervals

    def should_step(self, population, generation):
        """
        Do the step if we are on a predefined interval, and there are steps to do left.
        :param population: list, list of tblup.Individuals, current population.
        :param generation: int, current generation.
        :return: bool, True if we should step.
        """
        if self.step_intervals:
            if generation == self.step_intervals[0]:
                self.step_intervals.pop(0)
                return True

        return False

    def step(self, population):
        """
        Update the number of features.
        :param population: list, list of tblup.Individuals, current population.
        :param generation: int, current generation.
        """
        self.complexifier.step(self, population)
        self.step_count -= 1


class AdaptiveScheduler(StepwiseScheduler):
    """
    This scheduler updates individuals if the past individual with the maximum fitness hasn't changed
    in some predetermined number of generations.
    """
    def __init__(self, initial_features, final_features, generations, complexifier, memory=50):
        """
        Constructor.
        For this scheduler, we always double the features on a step, so we just need to figure out how often to double.
        :param initial_features: int, starting number of features.
        :param final_features: int, ending number of features.
        :param generations: int, total number of generations.
        :param complexifier: tblup.Complexifier
        :param memory: int, how many generations to remember the previous maximum fitness before complexifying.
        """
        super(AdaptiveScheduler, self).__init__(initial_features, final_features, generations, complexifier)

        self.prev = float('-inf')
        self.count = 0
        self.memory = memory

    def should_step(self, population, generation):
        """
        If we have seen the same maximum fitness 50 generations in a row, do the step.
        :param population: list, list of tblup.Individuals, current population.
        :param generation: int, current generation.
        :return: bool, True if we should step.
        """
        if len(self.step_intervals) == 0:
            return False

        max_fitness = max(population, key=lambda x: x.fitness).fitness

        if self.prev < max_fitness:
            self.prev = max_fitness
            self.count = 0

        else:
            self.count += 1

        if self.count >= self.memory - 1:
            self.step_intervals.pop(0)  # If we step, we don't want to "overstep" by stepping at the next interval.
            self.step_count -= 1
            self.prev = float('-inf')  # Reset the max fitness as recombination will likely disrupt fitnesses.
            return True

        # Still step if we need to at the predefined intervals.
        return super().should_step(population, generation)


class Complexifier(abc.ABC):
    """
    Complexifier service to allow for complexifying different types of individuals.
    """
    @abc.abstractmethod
    def step(self, scheduler, population):
        raise NotImplementedError()


class RandomKeyComplexifier(Complexifier):
    """
    Complexifies random key individuals.
    """
    def step(self, scheduler, population):
        if len(scheduler.step_intervals) == 0 and 2 * len(population[0]) > scheduler.final:
            for indv in population.population:
                indv.fill(scheduler.final, population.dimensionality)

        else:
            # Indices that the genomes decode to.
            indices = {individual.uid: individual.genome for individual in population}

            new_length = 2 * len(population[0])
            n = len(population)
            new_pop = []
            for _ in population:
                idx_1 = random.randrange(0, n)
                idx_2 = exclusive_randrange(0, n, [idx_1])

                indv_1, indv_2 = population[idx_1], population[idx_2]
                indices_1, indices_2 = indices[indv_1.uid], indices[indv_2.uid]

                new_indv = RandomKeyIndividual(new_length, population.dimensionality)

                # Set the internal genome to be the values of the genomes we are complexifying with.
                internal = new_indv.get_internal_genome()
                internal[indices_1] = indv_1[indices_1]
                internal[indices_2] = indv_2[indices_2]
                new_indv.set_internal_genome(internal)

                new_pop.append(new_indv)

            population.population = new_pop


class IndexComplexifier(Complexifier):
    """
    Complexifies index based individuals.
    """
    def step(self, scheduler, population):
        """
        Complexify a population of tblup.IndexIndividuals
        :param scheduler: tblup.FeatureScheduler
        :param population: tblup.Population
        """
        # If we're on the last step, and can't double the individual size, we just fill with random indices.
        if len(scheduler.step_intervals) == 0 and 2 * len(population[0]) > scheduler.final:
            for indv in population.population:
                indv.fill(scheduler.final)

        else:
            new_length = 2 * len(population[0])  # Our new desired length is twice the old length.

            # For performance, turn top genomes into sets.
            as_set = []
            n = len(population)
            for individual in population:
                as_set.append(set(individual))

            next_pop = []
            for i in range(len(population)):
                idx_1 = random.randrange(0, n)
                idx_2 = exclusive_randrange(0, n, [idx_1])

                first = as_set[idx_1]
                second = as_set[idx_2]

                union = first.union(second)

                # Fill the new genome with random indices until it is of the desired length.
                while len(union) < new_length:
                    union.add(random.randrange(0, population.dimensionality))  # Add will ensure uniqueness.

                indv = IndexIndividual(new_length, population.dimensionality, genome=np.array(union))

                next_pop.append(indv)

            population.population = next_pop

