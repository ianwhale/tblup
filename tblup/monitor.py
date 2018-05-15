import os
import csv
import pickle
from math import sqrt
from os.path import join, isdir, isfile


class Monitor:
    """
    Class for monitoring the population statistics.
    """

    ROUND_DECIMALS = 4

    def __init__(self, args):
        """
        Constructor.
        :param args: object, argparse.Namespace.
        """
        results = join(".", "results")
        if not isdir(results):
            os.mkdir(results)

        subdir = self.make_subdir(args)

        results = join(results, subdir)
        if not isdir(results):
            os.mkdir(results)

        results_file = join(results, str(args.seed).zfill(3) + "_results")
        archive_file = join(results, str(args.seed).zfill(3) + "_archive")

        # Be sure to not overwite a file.
        i = 1
        temp_res = results_file
        temp_arch = archive_file
        while isfile(temp_res + ".csv") or isfile(temp_arch + ".json"):
            temp_res = results_file + "_" + str(i)
            temp_arch = archive_file + "_" + str(i)

            i += 1

        self.results_file = temp_res + ".csv"
        self.archive_file = temp_arch + ".pkl"

        header = ["generation", "max_fitness", "min_fitness", "median_fitness", "mean_fitness", "stdev_fitness", "len"]
        with open(self.results_file, "w") as f:
            csv.writer(f).writerow(header)

    def make_subdir(self, args):
        """
        Make the name of the directory that is going to hold all the results for a particular experiment.
        :param args: object, argparse.Namespace.
        :return: string
        """
        option_list = [str(args.regressor)]

        if args.de_strategy != "de_rand_1":
            option_list.append(str(args.de_strategy))

        if args.feature_scheduling is not None:
            option_list.append("f" + str(args.initial_features))
            option_list.append(str(args.feature_scheduling))
            option_list.append("i" + str(args.initial_features))

        option_list.append("n" + str(args.population_size))
        option_list.append("g" + str(args.generations))
        option_list.append("cr" + str(args.crossover_rate).replace(".", ""))
        option_list.append("mi" + str(args.mutation_intensity).replace(".", ""))

        return "_".join(option_list)

    def report(self, population):
        """
        Write statistics out to file.
        :param population: tblup.Population.
        """
        with open(self.results_file, "a") as f:
            csv.writer(f).writerow(self.gather_stats(population))

    def set_override(self, obj):
        """
        Override for json decoder to handle frozenset.
        :raises: TypeError.
        :return: list
        """
        if isinstance(obj, frozenset):
            return str(list(obj))
        raise TypeError

    def save_archive(self, population):
        """
        Save the archive out to a JSON file.
        :param population: tblup.Population, the current population.
        """
        with open(self.archive_file, "wb") as f:
            pickle.dump(population.evaluator.archive, f)

    def gather_stats(self, population):
        """
        Gather statistics.
        :param population: tblup.Population.
        :return list: a row in the results file.
        """
        sorted_pop = sorted(population, key=lambda x: x.fitness)

        median_idx = len(population) / 2.0
        if int(median_idx) == median_idx:
            # Population length was even, use middle index.
            median_fitness = sorted_pop[int(median_idx)].fitness

        else:
            # Population length was odd, use average of two middle values.
            median_fitness = (sorted_pop[int(median_idx)].fitness + population[int(median_idx) + 1].fitness) / 2

        max_fitness = population[-1].fitness
        min_fitness = population[0].fitness

        fitness_sum = 0
        sum_of_squares = 0
        for indv in population:
            fitness_sum += indv.fitness
            sum_of_squares += indv.fitness * indv.fitness

        n = len(population)
        stdev_fitness = sqrt(n * sum_of_squares - (fitness_sum * fitness_sum) / (n * (n - 1)))
        mean_fitness = fitness_sum / n

        current_length = len(population[0])

        return [
            population.generation,
            round(max_fitness, self.ROUND_DECIMALS),
            round(min_fitness, self.ROUND_DECIMALS),
            round(median_fitness, self.ROUND_DECIMALS),
            round(mean_fitness, self.ROUND_DECIMALS),
            round(stdev_fitness, self.ROUND_DECIMALS),
            current_length
        ]
