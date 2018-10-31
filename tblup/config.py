import argparse


def boollike(v):
    """See: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class TBLUPArgumentParser(argparse.ArgumentParser):
    """
    Defines constants for common names around the system and extends the real argument parser.
    """
    def parse_args(self, args=None, namespace=None):
        namespace = super(TBLUPArgumentParser, self).parse_args(args=args, namespace=namespace)

        namespace.SEED_STRATEGY_TOP_SNPS = "top_snps"

        namespace.SEED_METRIC_P_VALUE = "p_value"

        namespace.INDIVIDUAL_TYPE_RANDOM_KEYS = "randkeys"
        namespace.INDIVIDUAL_TYPE_INDEX = "index"
        namespace.INDIVIDUAL_TYPE_NULLABLE = "nullable"
        namespace.INDIVIDUAL_TYPE_COEVOLE = "coevolve"

        namespace.REGRESSOR_TYPE_BLUP = "blup"
        namespace.REGRESSOR_TYPE_INTRACV_BLUP = "intracv_blup"
        namespace.REGRESSOR_TYPE_INTERCV_BLUP = "intercv_blup"
        namespace.REGRESSOR_TYPE_MONTECV_BLUP = "montecv_blup"

        namespace.FEATURE_SCHEDULING_STEPWISE = "stepwise"
        namespace.FEATURE_SCHEDULING_ADAPTIVE = "adaptive"
        namespace.FEATURE_SCHEDULING_PROGRESSIVE_CUTS = "progressive_cuts"

        namespace.LOCAL_SEARCH_KNOCKOUT = "knockout"

        return namespace


parser = TBLUPArgumentParser(description="TBLUP Python Implementation")

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
                                                        "blup, intercv_blup, intracv_blup, montecv_blup")
parser.add_argument("--remove_snps", type=boollike, default="false", help="at h(1 + alpha), the r largest indices in "
                                                                          "the highest fitness individual will be "
                                                                          "disallowed from the fitness evaluation")
parser.add_argument("--removal_r", type=int, default=None, help="r largest indices to remove, if None, uses --features")
parser.add_argument("--heritability", type=float, default=0.4, help="narrow-sense heritability of trait "
                                                                    "being predicted")
parser.add_argument("--cv_folds", type=int, default=5, help="number of folds to use in cross-validation")

#
# Evolutionary
#
parser.add_argument("--generations", type=int, default=100, help="number of generations to run")
parser.add_argument("--population_size", type=int, default=50, help="number of individuals in population")
parser.add_argument("--features", type=int, default=100, help="number of features to select,"
                                                              " treated as initial length in coevolution individuals.")
parser.add_argument("--initial_features", type=int, default=None, help="features in initial population")
parser.add_argument("--feature_scheduling", default=None,
                    help="scheduling scheme for increasing features (used if initial features is supplied)"
                         "available types: stepwise, adaptive, progressive_cuts")
parser.add_argument("--cuts_multiplier", type=int, default=10,
                    help="multiplier for progressive_cuts scheduling strategy")
parser.add_argument("--de_strategy", default="de_rand_1", help="type of differential evolution scheme"
                                                                      "available types: de_rand_1, de_currenttobest_1, "
                                                                      "sade, mde_pbx")
parser.add_argument("--crossover_rate", type=float, default=0.8, help="probability of crossover")
parser.add_argument("--mutation_intensity", type=float, default=0.5, help="mutation intensity")
parser.add_argument("--seeder", default=None, help="seeder to use, available types: top_snps")
parser.add_argument("--seeder_metric", default="p_value", help="the metric the seeder will use to filter the data "
                                                               "available types: p_value")
parser.add_argument("--individual", default="randkeys", help="type of individual available types: index, nullable, "
                                                             "randkeys, coevolve")
parser.add_argument("--coevolve_gamma", default=1.0, type=float, help="weight parameter for size of subset")
parser.add_argument("--clip", type=boollike, default="false", help="if true, clip at the dimensionality bounds [0, d) "
                                                                   "if false, no clipping will occur")
parser.add_argument("--record_testing", type=boollike, default="false", help="if true, record testing error in search "
                                                                             "if false, do not")
parser.add_argument("--local_search", default=None, help="local search functionality available types: knockout")
parser.add_argument("--stop_condition", default=None, help="stopping condition. h2 prefixes are "
                                                           "based on the h2/sqrt(h2) threshold. Available options: "
                                                           "h2_max, h2_min, h2_median, h2_mean")
parser.add_argument("--h2_alpha", default=0.0, type=float, help="Changes h2/sqrt(h2) threshold to be * 1.alpha larger")