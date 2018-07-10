import random
import numpy as np
from sklearn.model_selection import KFold
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

    if args.seeder == "half_half":
        return half_half_seeder(args, evaluator)

    if args.seeder == "one_elite":
        return one_elite(args, evaluator)

    if args.seeder == "top_snps":
        return top_snps(args, evaluator)

    raise NotImplementedError("Seeder {} not implemented.".format(args.seeder))


def get_top_n(args, evaluator, n=None):
    """
    Get the best features basted on the filtering metric.
    :param args: object, argparse.Namespace
    :param evaluator: tblup.Evaluator
    :param n: int | None, optional number of features to select.
    :return: list, list of ints corresponding to the indices of the best features determined by the
    """
    if n is None:
        n = args.initial_features if args.initial_features is not None else args.features

    if args.seeder_metric == "p_value":
        # Wrap the f_regression metric to get p-values only.
        def wrap(X, y):
            _, p_vals = f_regression(X, y)
            return -1 * p_vals  # Lower p-value is better so we multiply by -1.

        metric = wrap

    elif args.seeder_metric == "f_score":
        # Wrap the f_regression metric to get F-scores only.
        def wrap(X, y):
            scores, _ = f_regression(X, y)
            return scores

        metric = wrap

    else:
        raise NotImplementedError("No option for filtering metric {}.".format(args.seeder_metric))

    try:
        evaluator.training_indices

    except AttributeError:
        raise AttributeError("The provided regressor {} does not calculate training indices, "
                             "which are needed for a seeder to filter the data.".format(args.regressor))

    X = np.load(args.geno)[evaluator.training_indices]
    y = np.load(args.pheno)[evaluator.training_indices]

    #
    # Cross-validate our scoring metric to not contaminate it on the whole dataset.
    #
    scores = np.zeros(X.shape[1])
    for train, _ in KFold(n_splits=5).split(X):
        scores += metric(X[train], y[train].ravel())

    return np.argsort(scores, axis=0)[-n:]


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
            genome.add(random.randrange(0, args.dimensionality))

        yield np.array(list(genome))


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
        yield np.random.choice(range(args.dimensionality), n)


def top_snps(args, evaluator):
    """
    Entire population is the top-N SNPs as defined by the given metric, where N = pop_size * features_selected
    :param args: object, argparse.Namespace
    :param evaluator: tblup.Evaluator
    :yield: list, list of indices representing a genome.
    """
    n = args.initial_features if args.initial_features is not None else args.features
    num_features = n * args.population_size
    best_features = get_top_n(args, evaluator, n=num_features)

    start = 0
    while n + start < len(best_features):
        yield best_features[start:n + start]
        start += n

    # Start seeding randomly if we go over size.
    indices = range(0, args.dimensionality - 1)
    while True:
        yield np.random.choice(indices, n, replace=False)
