import abc
import random
import numpy as np
from tblup import uid
from copy import deepcopy


def get_individual(args):
    """
    Get the desired individual type.
    :param args: argparse.Namespace
    :return: callable, tblup.Individual constructor.
    """
    if args.individual == args.INDIVIDUAL_TYPE_INDEX:
        return IndexIndividual

    if args.individual == args.INDIVIDUAL_TYPE_NULLABLE:
        return NullableIndexIndividual

    if args.individual == args.INDIVIDUAL_TYPE_RANDOM_KEYS:
        return RandomKeyIndividual

    if args.individual == args.INDIVIDUAL_TYPE_COEVOLE:
        return CoevolutionIndividual

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

    @abc.abstractmethod
    def fill(self, new_size):
        raise NotImplementedError()


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

        if genome is not None:
            self._genome = genome

        else:
            self._genome = np.random.randint(0, dimensionality, length)

        self.fitness = float("-inf")

    @property
    def genome(self):
        return self._genome.astype(int)

    def get_internal_genome(self):
        return self._genome

    def set_internal_genome(self, genome):
        self._genome = genome

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

        self._genome = np.array(genome_set)


class RandomKeyIndividual(IndexIndividual):
    """
    Individual whose genome is real valued, and the length of the dimensionality.
    Indices into the matrix are obtained by sorted based on the value of a key.
        - I.e., we obtain the indices we want in the matrix by getting the top-N indices that would sort the genome.
    """
    def __init__(self, length, dimensionality, genome=None):
        """
        Constructor
        :param length: int, here we interpret this as how many indices will be selected after sorting.
        :param dimensionality: int, actual length of the individual.
        :param genome: list, optional list representing the genome.
        """
        super(RandomKeyIndividual, self).__init__(length, dimensionality)

        if genome is not None:
            self._genome = genome

        else:
            self._genome = np.random.uniform(size=dimensionality)

    @property
    def genome(self):
        return np.argsort(self._genome)[-int(self.length):]

    def __len__(self):
        return int(self.length)

    def fill(self, new_size):
        """
        "Fills" the genome with new indices. Ideally we'd do random indices, but we instead add the next largest entries
        in the sorted genome will be the "next best" found in the search. So we just add those.
        :param new_size: int
        """
        self.length = new_size


class CoevolutionIndividual(RandomKeyIndividual):
    """
    Individual that actively evolves the number of features to select during the search.
    """
    def __init__(self, length, dimensionality, genome=None):
        """
        Constructor
        :param length: int, here we interpret this as how many indices will be selected after sorting.
        :param dimensionality: int, actual length of the individual.
        :param genome: list, optional list representing the genome.
        """
        super(CoevolutionIndividual, self).__init__(length, dimensionality, genome=genome)

        self.length = np.random.randint(20, 2000)  # TODO: Make these hyperparameters.

    def get_internal_genome(self):
        """
        Override to include the number of features in the evolution process.
        :return: np.array.
        """
        return np.append(self._genome, self.length)

    def set_internal_genome(self, genome):
        """
        Override to peel off the last element for the num_features member.
        :param genome:
        """
        if len(genome) == self.dimensionality + 1:
            self.length = genome[-1] if genome[-1] > 1 else 1  # Protect from lengths <= 0.
            self._genome = np.delete(genome, -1)

        elif len(genome) == self.dimensionality:
            self._genome = genome

        else:
            raise RuntimeError("Genome of invalid length, must be dimensionality d or d + 1.")


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
        condition = np.logical_and(0 <= self._genome, self._genome < self.dimensionality)
        return np.extract(condition, self._genome).astype(int)

    def __len__(self):
        """
        Return the length of the self.genome property.
        :return: int
        """
        length = 0
        for gene in self._genome:
            length += 1 if 0 <= gene < self.dimensionality else 0
        return length
