import abc
import random
from tblup import uid
from copy import deepcopy


def get_individual(args):
    """
    Get the desired individual type.
    :param args: argparse.Namespace
    :return: callable, tblup.Individual constructor.
    """
    if args.individual == "index":
        return IndexIndividual

    if args.individual == "nullable":
        return NullableIndexIndividual

    raise NotImplementedError("Individual with config option {} not implemented.".format(args.individual))


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
            self._genome = genome

        else:
            self._genome = random.sample(range(dimensionality), self.length)

        self.fitness = float("-inf")

    @property
    def genome(self):
        return self._genome

    def __deepcopy__(self, memo):
        """
        Deepcopy override. Upcalls parent to get new uid then deepcopies the genome.
        :param memo: dict
        :return: tblup.IndexIndividual
        """
        cp = super(IndexIndividual, self).__deepcopy__(memo)
        cp._genome = deepcopy(self._genome)
        return cp

    def __len__(self):
        return len(self._genome)

    def __getitem__(self, item):
        return self._genome[item]

    def __setitem__(self, key, value):
        self._genome[key] = value

    def fill(self, new_size):
        genome_set = set(self._genome)

        while len(genome_set) < new_size:
            rand_features = random.sample(range(self.dimensionality), new_size - len(genome_set))
            genome_set.update(rand_features)

        self._genome = list(genome_set)


class NullableIndexIndividual(IndexIndividual):
    """
    Same as index individual, but handles the case when an index is outside [0, dimensionality) by removing it from
    the genome. This effectively allows the search to chose index subsets that are smaller than the desired features.
    """
    @property
    def genome(self):
        """
        Remove indices in the genome that are outside [0, dimensionality).
        :return: list
        """
        return [gene for gene in self._genome if 0 <= gene < self.dimensionality]

    def __len__(self):
        """
        Return the length of the self.genome property.
        :return: int
        """
        length = 0
        for gene in self._genome:
            length += 1 if 0 <= gene < self.dimensionality else 0
        return length
