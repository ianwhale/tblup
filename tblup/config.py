import argparse

parser = argparse.ArgumentParser(description="TBLUP Python Implementation")

#
# General
#
parser.add_argument("-s", "--seed", type=int, default=0, help="value of random seed")
parser.add_argument("-p", "--processes", type=int, default=4, help="number of parallel processes, -1 to use maximum")

#
# Data
#
parser.add_argument("--geno_train", default="./data/geno_train.npy", help="training genotype .npy file (m x n matrix)")
parser.add_argument("--pheno_train", default="./data/pheno_train.npy",
                    help="training phenotype .npy file (m x 1 vector)")
parser.add_argument("--geno_test", default="./data/geno_test.npy", help="testing genotype .npy file (p x n matrix)")
parser.add_argument("--pheno_test", default="./data/pheno_test.npy", help="testing phenotype .npy file (p x 1 vector)")

#
# Regression
#
parser.add_argument("--regressor", default="ridge", help="type of regression scheme, available types: gblup")
parser.add_argument("-h2", "--heritability", type=float, default=0.4, help="heritability of trait being predicted")

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
parser.add_argument("-cr", "--crossover_rate", type=float, default=0.8, help="probability of crossover")
parser.add_argument("-mi", "--mutation_intensity", type=float, default=0.5, help="mutation intensity")
