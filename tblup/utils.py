import os
import json
import random
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
    Get a random integer in a range [begin, end), excluding a particular number.
    :param begin: int, beginning of range.
    :param end: int, end of range.
    :param exclude: int, exclude this from the range.
    :return: int, random number != exclude
    """
    r = random.randrange(begin, end)
    while r == exclude:
        r = random.randrange(begin, end)
    return r


def build_kwargs(args):
    """
    Builds the arguements for a tblup.Population object as keyword arguments.
    :param args: object, argparse arguments.
    :return: dict, kwargs
    """
    from tblup import get_evolver
    from tblup import get_evaluator
    from tblup import IndexIndividual
    from tblup import get_scheduler
    from tblup import Monitor
    from tblup import DifferentialEvolutionSelector

    args.dimensionality = get_dimensionality(args)

    return {
        "evolver": get_evolver(args),
        "evaluator": get_evaluator(args),
        "selector": DifferentialEvolutionSelector(),
        "individual": IndexIndividual,
        "scheduler": get_scheduler(args),
        "length": args.initial_features if args.initial_features else args.features,
        "dimensionality": args.dimensionality,
        "num_individuals": args.population_size,
        "monitor": Monitor(args),
        "seeded_initial": get_seeded(args)
    }


def get_dimensionality(args):
    """
    Get the number of columns in the provided dataset.
    :param args: object, argparse.Namespace
    :return: int
    """
    train_geno_cols = np.load(args.geno_train).shape[1]
    test_geno_cols = np.load(args.geno_test).shape[1]

    assert train_geno_cols == test_geno_cols, "Training and testing data must have the same number of features."
    return train_geno_cols


def get_seeded(args):
    """
    Get the seeded population from the designated file.
    Designated file should be a json file.
    :param args: object, argparse.Namespace
    :return: None | list
    """
    out = None
    if args.seed_population is not None and os.path.isfile(args.seed_population):
        with open(args.seed_population, "r") as f:
            out = json.load(f)

    return out
