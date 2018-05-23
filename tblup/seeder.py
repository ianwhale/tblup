import random
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


def get_seeder(args, evaluator):
    """
    Get generator for the seeded population.
    :param args: object, argparse.Namespace
    :param evaluator: tblup.Evaluator, needed to get the training indices so we don't overfit to the testing data.
    :return: None | generator
    """
    if args.seeder is None:
        return None

    if args.seeder == "half-half":
        return half_half_seeder(args, evaluator)

    if args.seeder == "one-elite":
        return one_elite(args, evaluator)

    raise NotImplementedError("Seeder {} not implemented.".format(args.seeder))


def get_top_n(args, evaluator):
    """
    Get the best features basted on the filtering metric.
    :param args: object, argparse.Namespace
    :param evaluator: tblup.Evaluator
    :return: list, list of ints corresponding to the indices of the best features determined by the
    """
    n = args.initial_features if args.initial_features is not None else args.features

    if args.seeder_metric == "p_value":
        metric = f_regression

    else:
        raise NotImplementedError("No option for filtering metric {}.".format(args.seeder_metric))

    try:
        evaluator.training_indices

    except AttributeError:
        raise AttributeError("The provided regressor {} does not calculate training indices, "
                             "which are needed for a seeder to filter the data.".format(args.regressor))

    X = np.load(args.geno)[evaluator.training_indices]
    y = np.load(args.pheno)[evaluator.training_indices]

    return [int(x) for x in SelectKBest(metric, n).fit(X, y.ravel()).get_support(True)]


def half_half_seeder(args, evaluator):
    """
    The entire population has half random and half random selected features from the best features.
    :param args: object, argparse.Namespace
    :param evaluator: tblup.Evaluator
    :yield: list, list of indices representing a genome.
    """
    best_features = get_top_n(args, evaluator)
    n = args.initial_features if args.initial_features is not None else args.features

    for i in range(args.population_size):
        genome = set(random.sample(best_features, n // 2))  # Get n / 2 random indices from the filtered results.

        while len(genome) < n:
            rand_features = random.sample(range(args.dimensionality), n // 2)
            genome.update(rand_features)

        yield list(genome)


def one_elite(args, evaluator):
    """
    Entire population is random except one who is all the best features.
    :param args: object, argparse.Namespace
    :param evaluator: tblup.Evaluator
    :yield: list, list of indices representing a genome.
    """
    best_features = get_top_n(args, evaluator)
    n = args.initial_features if args.initial_features is not None else args.features

    yield best_features

    for i in range(args.population_size - 1):
        yield random.sample(range(args.dimensionality), n)
