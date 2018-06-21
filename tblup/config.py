import argparse


def boollike(v):
    """See: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="TBLUP Python Implementation")

#
# General
#
parser.add_argument("-s", "--seed", type=int, default=0, help="value of random seed")
parser.add_argument("-p", "--processes", type=int, default=4, help="number of parallel processes, -1 to use maximum")
parser.add_argument("-o", "--output", default=None, help="override automatic results directory name")

#
# Data
#
parser.add_argument("--geno", default="./data/geno.npy", help="training genotype .npy file (m x n matrix)")
parser.add_argument("--pheno", default="./data/pheno.npy",
                    help="training phenotype .npy file (m x 1 vector)")
parser.add_argument("--splitter", default=None, help="a custom train/test split function, available types: "
                                                     "pca")
parser.add_argument("--pca_outliers", type=boollike, default="false", help="only has an effect when splitter is pca, "
                                                                           "if false, the training data will be  "
                                                                           "the pca inliers "
                                                                           "if true, the training data will be the pca "
                                                                           "outliers")

#
# Regression
#
parser.add_argument("--regressor", default="blup", help="type of regression scheme, available types: "
                                                        "blup, intracv_blup, intercv_blup")
parser.add_argument("-h2", "--heritability", type=float, default=0.16, help="heritability of trait being predicted")
parser.add_argument("--cv_folds", type=int, default=5, help="number of folds to use in cross-validation")

#
# Evolutionary
#
parser.add_argument("--generations", type=int, default=100, help="number of generations to run")
parser.add_argument("--population_size", type=int, default=50, help="number of individuals in population")
parser.add_argument("--features", type=int, default=100, help="number of features to select")
parser.add_argument("--initial_features", type=int, default=None, help="features in initial population")
parser.add_argument("--feature_scheduling", default=None,
                    help="scheduling scheme for increasing features (used if initial features is supplied)"
                         "available types: stepwise, adaptive")
parser.add_argument("-de", "--de_strategy", default="de_rand_1", help="type of differential evolution scheme"
                                                                      "available types: de_rand_1, de_currenttobest_1, "
                                                                      "sade, mde_pbx")
parser.add_argument("-cr", "--crossover_rate", type=float, default=0.8, help="probability of crossover")
parser.add_argument("-mi", "--mutation_intensity", type=float, default=0.5, help="mutation intensity")
parser.add_argument("--seeder", default=None, help="seeder to use, available types: half_half, one_elite")
parser.add_argument("--seeder_metric", default="p_value", help="the metric the seeder will use to filter the data "
                                                               "available types: p_value, f_score")
