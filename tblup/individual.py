import abc
import random
from tblup import uid
from copy import deepcopy


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
        self.uid = next(uid)  # Get next globally unique id.
        self.length = length
        self.dimensionality = dimensionality

    def __deepcopy__(self, memo):
        """
        Deepcopy override. Need to update uid.
        Nothing should need to be actually deep copied, if so, override this and use an upcall in a derived class.
        :param memo: dict
        :return: tblup.Individual.
        """
        # Do __new__ to avoid calling constructor.
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__dict__.update(self.__dict__)

        # Get a new uid.
        cp.uid = next(uid)

        return cp


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

    def __deepcopy__(self, memo):
        """
        Deepcopy override. Upcalls parent to get new uid then deepcopies the genome.
        :param memo: dict
        :return: tblup.IndexIndividual
        """
        cp = super(IndexIndividual, self).__deepcopy__(memo)
        cp.genome = deepcopy(self.genome)
        return cp

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
