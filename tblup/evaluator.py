import os
import abc
import random
import numpy as np
import multiprocessing as mp
from tblup import make_grm
import tblup.sharearray as sa
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge


def get_evaluator(args):
    """
    Gets the evaluator type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.Evaluator
    """
    if args.regressor == "blup":
        return BlupParallelEvaluator(args.geno, args.pheno, args.heritability, n_procs=args.processes)

    if args.regressor == "intracv_blup":
        return IntraGCVBlupParallelEvaluator(args.geno, args.pheno, args.heritability,
                                              n_procs=args.processes, n_folds=args.cv_folds)

    if args.regressor == "intercv_blup":
        return InterGCVBlupParallelEvaluator(args.geno, args.pheno, args.heritability,
                                             n_procs=args.processes, n_folds=args.cv_folds)


class Evaluator(abc.ABC):
    """
    Abstract base class for Evaluators.
    """
    def __init__(self, data_path, labels_path):
        """
        Constructor.
        :param data_path: string, path to desired data file (SNP or otherwise).
        :param labels_path: string, path to labels.
        """
        assert os.path.isfile(data_path), "Argument for data_path {} not found.".format(data_path)
        assert os.path.isfile(labels_path), "Argument for labels_path {} not found.".format(labels_path)

        self.data_path = data_path
        self.labels_path = labels_path

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def evaluate(self, population, generation):
        """
        :param population: tblup.Population
        :param generation: int
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def genomes_to_evaluate(self, population):
        raise NotImplementedError()


class ParallelEvaluator(Evaluator):
    """
    Base class for evaluators that use multiprocessing that ensures the data is put into shared memory.
    """
    def __init__(self, data_path, labels_path, n_procs=-1):
        """
        Constructor.
        :param data_path: string, path to desired data file. Should be in npy format.
        :param labels_path: string, path to labels.
        :param n_procs: int, number of processes to use.
            If the argument is -1, MPEvaluator will use the max parallelism specified by mp.cpu_count()
        """
        super(ParallelEvaluator, self).__init__(data_path, labels_path)

        # IDs to use in the sharearray identifiers, process safe for use in shared environments.
        self.data_id = "data" + str(os.getpid())
        self.labels_id = "labels" + str(os.getpid())

        self.n_procs = n_procs

        self.pool = None

    def __enter__(self):
        """
        Create the multiprocessing pool.
        Only happens once to reduce overhead.
        """
        self.pool = mp.Pool(processes=mp.cpu_count() if self.n_procs == -1 else self.n_procs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Free the shared memory allocated in evaluations and close the processing pool.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        sa.free(self.data_id)
        sa.free(self.labels_id)

    @property
    def data(self):
        """
        Handle loading the data into shared memory if needed.
        :return: np.ndarray
        """
        return sa.cache(self.data_id, np.load(self.data_path), verbose=False)

    @property
    def labels(self):
        """
        Handle loading the labels into shared memory if needed.
        :return: np.ndarray
        """
        return sa.cache(self.labels_id, np.load(self.labels_path), verbose=False)

    def genomes_to_evaluate(self, population):
        raise NotImplementedError()

    def evaluate(self, population, generation):
        """
        :param population: tblup.Population
        :param generation: int
        """
        if self.pool is None:
            raise AttributeError("Pool was not set up.")

        pass


class BlupParallelEvaluator(ParallelEvaluator):
    """
    Evaluates individuals using multiprocessing and genomic best linear unbiased predictor.

    Expects the individuals to have nonnegative integer valued genomes.
    """

    TRAIN_TEST_SPLIT = 0.8   # 80% of the data will be training, 20% will be testing.
    TRAIN_VALID_SPLIT = 0.8  # Of the training data, 20% will be validation.

    def __init__(self, data_path, labels_path, r, n_procs=-1):
        """
        Constructor.
        :param data_path: string, path to the training data.
        :param labels_path: string, path to the labels for the training data.
        :param r: float, regularization parameter.
        """
        super(BlupParallelEvaluator, self).__init__(data_path, labels_path, n_procs=n_procs)
        
        # Store individuals we have evaluated already.
        # Indexing works as the hashed frozenset of the list => fitness.
        self.archive = {}

        self.r = r  # Regularization parameter.

        # Build training and testing indices.
        shape = np.load(data_path).shape
        self.n_samples, self.n_columns = shape[0], shape[1]
        indices = random.sample(range(self.n_samples), self.n_samples)
        n = int(len(indices) * self.TRAIN_TEST_SPLIT)
        self.training_indices = indices[:n]
        self.testing_indices = indices[n:]

    def blup(self, indices, train_indices, validation_indices):
        """
        Do BLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.

        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :param validation_indices: list, list of ints corresponding to which samples to use for validation.
        :return: float, prediction accuracy.
        """
        if self.n_columns > self.n_samples:
            # Do GBLUP. The GRM will be more efficient since we have more columns than samples.
            return self.gblup(indices, train_indices, validation_indices)

        else:
            # Do SNP-BLUP. Calculating the GRM is more costly in this case, so we just do normal ridge regression.
            return self.snp_blup(indices, train_indices, validation_indices)

    def gblup(self, indices, train_indices, validation_indices):
        """
        Do GBLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.

        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :param validation_indices: list, list of ints corresponding to which samples to use for validation.
        :return: float, prediction accuracy.
        """
        G = make_grm(self.data[:, indices])

        # Inverse the matrix using only the desired training samples.
        G_inv = G[train_indices, :][:, train_indices]
        G_inv.flat[:: G_inv.shape[0] + 1] += self.r  # Add regularization term to the diagonal of G.
        G_inv = np.linalg.inv(G_inv)

        prediction = np.matmul(np.matmul(G[:, train_indices], G_inv), self.labels[train_indices])

        return abs(pearsonr(self.labels[validation_indices], prediction[validation_indices])[0])

    def snp_blup(self, indices, train_indices, validation_indices):
        """
        Do SNP-BLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.

        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :param validation_indices: list, list of ints corresponding to which samples to use for validation.
        :return: float, prediction accuracy.
        """
        X = self.data[:, indices]
        y = self.labels

        X_train, X_valid = X[train_indices], X[validation_indices]
        y_train, y_valid = y[train_indices], y[validation_indices]

        p = np.mean(X_train, axis=0) / 2
        d = 2 * np.sum(p * (1 - p))
        l = (1 - self.r) / (self.r / d)

        X_train -= (2 * p)
        X_valid -= (2 * p)

        clf = Ridge(alpha=l)
        clf.fit(X_train, y_train)

        return abs(pearsonr(clf.predict(X_valid), y_valid)[0])

    def train_validation_indices(self, generation):
        """
        Just splits the data into two splits.
        :param generation: int, current generation. Not used here, used in subclasses.
        :return: (list, list), tuple of list of ints, (training, validation) indices.
        """
        n = int(len(self.training_indices) * self.TRAIN_VALID_SPLIT)
        return self.training_indices[n:], self.training_indices[:n]

    def __call__(self, genome, generation):
        """
        Call magic method, assign fitness to genome.
        :param genome: list, list of indexes into data matrix.
        :param generation: int, current generation.
        :return: float, fitness of genome.
        """
        train_indices, validation_indices = self.train_validation_indices(generation)
        return np.asscalar(self.blup(genome, train_indices, validation_indices))

    def __getstate__(self):
        """
        Magic method called when an object is pickled. Normally returns self.__dict__.
        This happens when a copy of self gets sent to a new process in the __call__ method.
        We cannot pickle the pool object and we do not want to pickle the archive, so we don't.
        :return: dict
        """
        safe_dict = {}

        for k, v in self.__dict__.items():
            if k != "archive" and k != "pool":
                safe_dict[k] = v

        return safe_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(BlupParallelEvaluator, self).__exit__(exc_type, exc_val, exc_tb)

    def genomes_to_evaluate(self, population):
        """
        Get the genomes that we haven't evaluated yet.
        :param population: tblup.Population.
        :return: list of lists
        """
        to_evaluate = []
        indices = []
        for i, indv in enumerate(population):
            unique = frozenset(indv.genome)

            if unique in self.archive:
                indv.fitness = self.archive[unique]

            else:
                indices.append(i)
                to_evaluate.append(list(unique))

        return to_evaluate, indices

    def evaluate(self, population, generation):
        """
        Evaluate the population with GBLUP.
        :param population: list, list of individuals..
        :param generation: int, current generation.
        :return: list, list of tblup.Individuals.
        """
        super(BlupParallelEvaluator, self).evaluate(population, generation)

        to_evaluate, indices = self.genomes_to_evaluate(population)

        results = []
        for genome in to_evaluate:
            results.append(
                self.pool.apply_async(self, (genome, generation))
            )

        for i, idx in enumerate(indices):
            population[idx].fitness = results[i].get()
            self.archive[frozenset(to_evaluate[i])] = population[idx].fitness

        return population

    def evaluate_testing(self, population):
        """
        Evaluate the whole population's testing accuracy.
        :param population: tblup.Population.
        :return:
        """
        results = []
        for indv in population:
            results.append(
                self.pool.apply_async(self.testing_function, (indv,))
            )

        accs = []
        for res in results:
            accs.append(res.get()[0])

        return accs

    def testing_function(self, genome):
        """
        Same as __call__, but using the testing indices.
        :param genome: list, list of indexes into data matrix.
        :return: float, testing accuracy feature subset..
        """
        return self.blup(genome, self.training_indices, self.testing_indices)


class InterGCVBlupParallelEvaluator(BlupParallelEvaluator):
    """
    Same as BlupParallelEvaluator, but with cross-validation between generations.
    This means that the validation set is changed each generation.

    InterGCV = intergenerational cross validation, or between generations.

    Only overrides the constructor and train_validation_indices.
    """
    def __init__(self, data_path, labels_path, r, n_procs=-1, n_folds=5):
        """
        See parent doc.
        :param n_folds: int, number of cross validation folds.
        """
        super(InterGCVBlupParallelEvaluator, self).__init__(data_path, labels_path, r, n_procs=n_procs)

        self.n_folds = n_folds
        self.fold_indices = self.make_fold_indices(self.training_indices, self.n_folds)

    @staticmethod
    def make_fold_indices(indices, n_folds):
        """
        Make the cross validation fold indices.
        Original: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_split.py#L421
        :param indices: list, list of ints, indices into some data.
        :param n_folds: int, how many folds to make.
        :return: list of lists, list of each index per fold.
        """
        sizes = [len(indices) // n_folds for _ in range(n_folds)]
        for i in range(len(indices) % n_folds):  # Account for remainder in division.
            sizes[i] += 1

        fold_indices = []
        current = 0
        for size in sizes:
            start, stop = current, current + size
            fold_indices.append(indices[start:stop])
            current = stop

        return fold_indices

    def train_validation_indices(self, generation):
        """
        Overriden method, changes the validation split every generation.
        :param generation: int, current generation.
        :return: (list, list), tuple of list of ints, (training, validation) indices.
        """
        train = []

        for i in range(self.n_folds):
            if i == generation % self.n_folds:
                continue

            train += self.fold_indices[i]

        return train, self.fold_indices[generation % self.n_folds]


class IntraGCVBlupParallelEvaluator(InterGCVBlupParallelEvaluator):
    """
    Same as BlupParallelEvaluator, but with k-fold cross-validation within a fitness evaluation.

    IntraGCV = intragenerational cross validation, or within generations.

    Only overrides the call method.
    """
    def __init__(self, data_path, labels_path, r, n_procs=-1, n_folds=5):
        """
        See parent doc.
        """
        super(IntraGCVBlupParallelEvaluator, self).__init__(data_path, labels_path, r, n_procs=n_procs, n_folds=n_folds)

    def __call__(self, genome, generation):
        """
        Call magic method, assign fitness to genome.
        Does k-fold cross-validated BLUP.
        :param genome: list, list of indexes into data matrix.
        :param generation: int, current generation. Not used for this override.
        :return: float, fitness of genome. Average Pearson's R over the folds.
        """
        fitness_sum = 0
        for k in range(self.n_folds):
            train_indices, validation_indices = self.train_validation_indices(k)
            fitness_sum += self.blup(genome, train_indices, validation_indices)

        return np.asscalar(fitness_sum / self.n_folds)
