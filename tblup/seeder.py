import abc
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression


def get_seeder(args, evaluator):
    """
    Get generator for the seeded population.
    :param args: tblup.TBLUPArgumentParser
    :param evaluator: tblup.Evaluator, needed to get the training indices so we don't overfit to the testing data.
    :return: None | generator
    """
    if args.seeder is None:
        return None

    length = args.features if args.initial_features is None else args.initial_features
    metric = None
    strategy = None
    seeder = None

    if args.seeder_metric == args.SEED_METRIC_P_VALUE:
        metric = p_value

    if metric is None:
        raise NotImplementedError("Metric {} not implemented.".format(args.metric))

    if args.seeder == args.SEED_STRATEGY_TOP_SNPS:
        strategy = TopSNPsSeedStrategy(evaluator, metric, args.geno, args.pheno, )

    if strategy is None:
        raise NotImplementedError("Strategy {} not implemented.".format(args.strategy))

    if args.individual == args.INDIVIDUAL_TYPE_INDEX or args.individual == args.INDIVIDUAL_TYPE_NULLABLE:
        seeder = IndexSeeder(strategy, length)

    elif args.individual == args.INDIVIDUAL_TYPE_RANDOM_KEYS:
        seeder = RandomKeySeeder(strategy, length, args.dimensionality)

    if seeder is None:
        raise NotImplementedError("Seeder {} not implemented.".format(args.seeder))

    return seeder


##################################
# Seeders, differ per individual #
##################################


class Seeder(abc.ABC):
    """
    Abstract base class for a seeder.
    All seeders should be iterators to allow for the functionality described in the tblup.Population constructor.
    """
    def __init__(self, strategy, length):
        """
        Constructor.
        :param strategy: tblup.SeedStrategy
        :param length: int, length of individuals.
        """
        assert isinstance(strategy, SeedStrategy)

        self.strategy = strategy
        self.length = length

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError()

    def __iter__(self):
        self.strategy.reset()
        return self


class IndexSeeder(Seeder):
    """
    Seeder for index individuals.
    """
    def __next__(self):
        return self.strategy.get_next_indices(self.length)


class RandomKeySeeder(Seeder):
    """
    Seeder for random key individuals.
    """
    def __init__(self, strategy, length, dimensionality):
        """
        Constructor.
        :param strategy: tblup.SeedStrategy
        :param length: int, length of genome.
        :param dimensionality: int, dimensionality of problem.
        """
        super(RandomKeySeeder, self).__init__(strategy, length)

        self.dimensionality = dimensionality

    def __next__(self):
        genome = np.random.rand(self.dimensionality)
        genome[self.strategy.get_next_indices(self.length)] = 1
        return genome


######################
# Seeding strategies #
######################


class SeedStrategy(abc.ABC):
    """
    Abstract base class for a seeding strategy.
    """

    N_SPLITS = 5

    def __init__(self, evaluator, metric, geno_path, pheno_path):
        """
        Constructor.
        :param evaluator: tblup.Evaluator.
        :param metric: callable, metric function that returns a list of values describing the "goodness" of each index.
            - "Goodness" should be defined in ascending order, i.e. lower values should be less "good".
        :param geno_path: path to genotype data.
        :param pheno_path: path to phenotype data.
        """
        try:
            self.training_indices = evaluator.training_indices

        except AttributeError:
            raise AttributeError("The provided evaluator {} does not calculate training indices, which are needed "
                                 "for a seeder to filter the data.".format(evaluator.__class__.__name__))

        self.metric = metric
        self.indices = self.get_sorted_indices(geno_path, pheno_path)

    @abc.abstractmethod
    def get_next_indices(self, length):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    def get_sorted_indices(self, geno_path, pheno_path):
        """
        Get the sorted indices based on the metric given in constructor.
        :param geno_path: path to genotype data.
        :param pheno_path: path to phenotype data.
        :return: np.array, integers corresponding to indices in order defined by the metric.
        """
        X, y = np.load(geno_path), np.load(pheno_path)

        #
        # Cross-validate the scoring metric on the training data only to not contaminate the dataset.
        #
        scores = np.zeros(X.shape[1])
        for train, _ in KFold(n_splits=self.N_SPLITS).split(X):
            scores += self.metric(X[train], y[train].ravel())

        return np.argsort(scores, axis=0)


class TopSNPsSeedStrategy(SeedStrategy):
    """
    Top SNPs seeding strategy. Individuals are filled in with the best SNPs in order, determined by the provided metric.
    If  more indices are needed than
    """
    def __init__(self, evaluator, metric, geno_path, pheno_path):
        """
        See parent doc.
        """
        super(TopSNPsSeedStrategy, self).__init__(evaluator, metric, geno_path, pheno_path)

        self.current_index = 0

    def get_next_indices(self, length):
        """
        Returns the next best indices.
        Return random indices if the max length is reached.
        :param length: int, length of indices needed to construct an individual.
        :return: np.array
        """
        n = self.current_index
        self.current_index += length

        if self.current_index > len(self.indices):
            return np.random.choice(self.indices, length, replace=False)
        else:
            return self.indices[n:n + length]

    def reset(self):
        """
        Reset the current index count.
        """
        self.current_index = 0

###################
# Seeding metrics #
###################


def p_value(X, y):
    """
    P-value metric (GWAS).
    :param X: np.ndarray, (N x P), matrix of genotypes.
    :param y: np.ndarray, (N x 1), vector of phenotypes.
    :return: np.ndarray, (N x 1), vector of p_values, negated to reflect that a smaller p-value is better.
    """
    _, p_values = f_regression(X, y)
    return -1 * p_values


def f_score(X, y):
    """
    F score metric (GWAS). Same as p-value but gives effect values instead.
    :param X: np.ndarray, (N x P), matrix of genotypes.
    :param y: np.ndarray, (N x 1), vector of phenotypes.
    :return: np.ndarray, (N x 1), vector of effect values. Higher is more significant.
    """
    f_scores, _ = f_regression(X, y)
    return f_scores
