import abc
import random

class Individual(abc.ABC):
    """
    Individual base class.
    """
    def __init__(self, length, dimensionality):
        """
        Constructor.
        :param length: int, how long an individual is.
        :param dimensionality: int, the dimensionality of the problem.
        """
        self.length = length
        self.dimensionality = dimensionality


class IndexIndividual(Individual):
    """
    Individual whose genome is a list of column indices in a data matrix.
    """
    def __init__(self, length, dimensionality, genome=None):
        """
        Constructor.
        :param length: int, length of the individual.
        :param dimensionality: int, the number of columns in the data
        :param genome: list, optional list representing the genome.
        """
        super(IndexIndividual, self).__init__(length, dimensionality)

        if genome:
            self.genome = genome

        else:
            self.genome = random.sample(range(dimensionality), self.length)

        self.fitness = float("-inf")

    def __len__(self):
        return len(self.genome)

    def __getitem__(self, item):
        return self.genome[item]

    def __setitem__(self, key, value):
        self.genome[key] = value

    def fill(self, new_size):
        genome_set = set(self.genome)

        while len(genome_set) < new_size:
            rand_features = random.sample(range(self.dimensionality), new_size - len(genome_set))
            genome_set.update(rand_features)

        self.genome = list(genome_set)
