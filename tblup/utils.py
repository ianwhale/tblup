import os
import random
import pickle
import numpy as np


def make_grm(geno):
    """
    Make the genomic relationship matrix.
    - Expects rows as individuals and columns as markers.
    :param geno: np.ndarray, 2D, genotype marker matrix. (N x P)
    :return: np.ndarray, 2D, genomic relationship matrix. (N X N)
    """
    p = np.mean(geno, axis=0) / 2  # Row means over 2.
    P = 2 * (p - 0.5)
    W = (geno - 1) - P  # Subtract P from each column of G, where G is put into {-1, 0, 1} format from {0, 1, 2}.
    WtW = np.matmul(W, np.transpose(W))
    return WtW / (2 * np.sum(p * (1 - p)))


def exclusive_randrange(begin, end, exclude):
    """
    Get a random integer in a range [begin, end), excluding a particular list of numbers.
    :param begin: int, beginning of range.
    :param end: int, end of range.
    :param exclude: list, exclude this from the range.
    :return: int, random number not in exclude.
    """
    r = random.randrange(begin, end)
    exclude = set(exclude)

    assert len(exclude) < (end - begin), "Exclusion range larger than random range."

    while r in exclude:
        r = random.randrange(begin, end)
    return r


def build_kwargs(args):
    """
    Builds the arguements for a tblup.Population object as keyword arguments.
    :param args: object, argparse arguments.
    :return: dict, kwargs
    """
    # Import inside function to avoid circular imports in other modules.
    from tblup import get_seeder
    from tblup import get_evolver
    from tblup import get_evaluator
    from tblup import get_scheduler
    from tblup import get_individual
    from tblup import get_stop_condition
    from tblup import Monitor
    from tblup import DifferentialEvolutionSelector

    args.dimensionality = get_dimensionality(args)

    d = {
        "evolver": get_evolver(args),
        "evaluator": get_evaluator(args),
        "selector": DifferentialEvolutionSelector(),
        "individual": get_individual(args),
        "scheduler": get_scheduler(args),
        "length": features_logic(args),
        "dimensionality": args.dimensionality,
        "num_individuals": args.population_size,
        "monitor": Monitor(args),
        "record_testing": args.record_testing,
        "stop_condition": get_stop_condition(args),
        "coevolve_gamma": args.coevolve_gamma
    }

    d["seeded_initial"] = get_seeder(args, d["evaluator"])

    return d


def features_logic(args):
    """
    Some logic to handle initial features
    :param args: object, argpase.Namespace
    :return: int, number of features in initial population.
    """
    if args.feature_scheduling == args.FEATURE_SCHEDULING_PROGRESSIVE_CUTS:
        return args.features * args.cuts_multiplier
    else:
        return args.initial_features if args.initial_features else args.features


def get_dimensionality(args):
    """
    Get the number of columns in the provided dataset.
    :param args: object, argparse.Namespace
    :return: int
    """
    geno_cols = np.load(args.geno).shape[1]

    return geno_cols


def id_gen():
    """
    Unique id generator.
    :yield int:
    """
    i = 0
    while True:
        yield i
        i = i + 1


# Hacky way to get unique ids across the whole program.
# Individuals aren't created in parallel, if they are, this won't work.
uid = id_gen()
