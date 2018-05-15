import random
from math import log, floor
from tblup.utils import exclusive_randrange
from tblup.individual import IndexIndividual


def get_scheduler(args):
    """
    Gets the scheduler type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.FeatureScheduler
    """
    if args.initial_features is None:
        return FeatureScheduler(args.initial_features, args.features, args.generations)

    if args.feature_scheduling == "stepwise":
        return StepwiseFeatureScheduler(args.initial_features, args.features, args.generations)


class FeatureScheduler(object):
    """
    Feature scheduler base class, doesn't do anything.
    """
    def __init__(self, initial_features, final_features, generations):
        self.initial = initial_features
        self.final = final_features
        self.generations = generations

    def do_step(self, population, generation):
        return False

    def step(self, population):
        pass


class StepwiseFeatureScheduler(FeatureScheduler):
    """
    Scheduler doubles the genome length at regular intervals throughout the search.
    """
    def __init__(self, initial_features, final_features, generations):
        """
        Constructor.
        :param initial_features: int, starting number of features.
        :param final_features: int, ending number of features.
        :param generations: int, total number of generations.
        """
        super(StepwiseFeatureScheduler, self).__init__(initial_features, final_features, generations)

        # Number of times we will double the genome length plus once more to
        # make genome length the correct final size.
        # 2^x * init = final => log_2(final / init) = x
        self.step_count = floor(log((final_features / initial_features), 2))
        self.step_interval = generations // (self.step_count + 1)

        step_intervals = []
        current = self.step_interval
        for _ in range(self.step_count):
            step_intervals.append(current)
            current += self.step_interval

        self.step_intervals = step_intervals

    def do_step(self, population, generation):
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
        # If we're on the last step, and can't double the individual size, we just fill with random indices.
        if len(self.step_intervals) == 0 and 2 * len(population[0]) > self.final:
            for indv in population.population:
                indv.fill(self.final, population.dimensionality)
            return

        cut = len(population)  # Only select from the top cut of the population.
        new_length = 2 * len(population[0])  # Our new desired length is twice the old length.
        population.population.sort(reverse=True, key=lambda x: x.fitness)

        # For performance, turn top genomes into sets.
        as_set = []
        for i in range(cut):
            as_set.append(set(population[i].genome))

        next_pop = []
        for i in range(len(population)):
            # Get two unique indexes from the top cut of the population.
            idx_1 = random.randrange(0, cut)
            idx_2 = exclusive_randrange(0, cut, idx_1)

            first = as_set[idx_1]
            second = as_set[idx_2]

            union = first.union(second)

            # Fill the new genome with random indices until it is of the desired length.
            while len(union) < new_length:
                union.add(random.randrange(0, population.dimensionality))  # Add will ensure uniqueness.

            indv = IndexIndividual(new_length, population.dimensionality, genome=list(union))

            next_pop.append(indv)

        population.population = next_pop
        self.step_count -= 1


class AdaptiveScheduler(StepwiseFeatureScheduler):
    """
    This scheduler updates individuals if the past individual with the maximum fitness hasn't changed
    in some predetermined number of generations.
    """
    def __init__(self, initial_features, final_features, generations, memory=50):
        """
        Constructor.
        For this scheduler, we always double the features on a step, so we just need to figure out how often to double.
        :param initial_features: int, starting number of features.
        :param final_features: int, ending number of features.
        :param generations: int, total number of generations.
        """
        super(AdaptiveScheduler, self).__init__(initial_features, final_features, generations)

        self.prev = float('-inf')
        self.count = 0
        self.memory = memory

    def do_step(self, population, generation):
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
        return super().do_step(population, generation)
