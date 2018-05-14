import os
import abc
import numpy as np
import multiprocessing as mp
import tblup.sharearray.sharearray as sa
from tblup.utils import make_grm
from scipy.stats import pearsonr


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
    def evaluate(self, population):
        """
        :param population: tblup.Population
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
        def load_data():
            return np.load(self.data_path)
        return sa.cache(self.data_id, load_data, verbose=False)

    @property
    def labels(self):
        """
        Handle loading the labels into shared memory if needed.
        :return: np.ndarray
        """
        def load_labels():
            return np.load(self.labels_path)
        return sa.cache(self.labels_id, load_labels, verbose=False)

    def genomes_to_evaluate(self, population):
        raise NotImplementedError()

    def evaluate(self, population):
        """
        :param population: tblup.Population
        :return:
        """
        if self.pool is None:
            raise AttributeError("Pool was not set up.")

        pass


class GblupParallelEvaluator(ParallelEvaluator):
    """
    Evaluates individuals using multiprocessing and genomic best linear unbiased predictor.

    Expects the individuals to have nonnegative integer valued genomes.
    """

    TRAIN_PERCENTAGE = 0.8  # 80% of the data will be training, 20% will be validation.

    def __init__(self, data_path, labels_path, r, n_procs=-1):
        """
        Constructor.
        :param data_path: string, path to the training data.
        :param labels_path: string, path to the labels for the training data.
        :param r: float, regularization parameter.
        """
        super(GblupParallelEvaluator, self).__init__(data_path, labels_path, n_procs=n_procs)
        
        # Store individuals we have evaluated already.
        # Indexing works as the hashed string of the list => fitness.
        self.archive = {}

        self.r = r  # Regularization parameter.
        self.n_samples = self.data.shape[0]
        self.indices = np.random.permutation([i for i in range(self.n_samples)])

    def gblup(self, indices, train_indices, validation_indices):
        """
        Do GBLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.
        Note: for the uninitiated, GBLUP is basically ridge regression on a special matrix.
        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :return: (np.ndarray, np.ndarray), (GRM, the GBLUP solution)
        """
        G = make_grm(self.data[:, indices])

        # Inverse the matrix using only the desired training samples.
        G_inv = G[train_indices, :][:, train_indices]
        G_inv.flat[:: G_inv.shape[0] + 1] += self.r  # Add regularization term to the diagonal of G.
        G_inv = np.linalg.inv(G_inv)

        prediction = np.matmul(np.matmul(G[:, train_indices], G_inv), self.labels[train_indices])

        return abs(pearsonr(self.labels[validation_indices], prediction[validation_indices])[0])

    def train_validation_indices(self, generation):
        """
        Just splits the data into two splits.
        :param generation: int, current generation. Not used here, used in subclasses.
        :return: (list, list), tuple of list of ints, (training, validation) indices.
        """
        n = int(len(self.indices) * self.TRAIN_PERCENTAGE)
        return self.indices[n:], self.indices[:n]

    def __call__(self, genome, generation):
        """
        Call magic method, assign fitness to genome.
        :param genome: list, list of indexes into data matrix.
        :param generation: int, current generation.
        :return: float, fitness of genome.
        """
        train_indices, validation_indices = self.train_validation_indices(generation)
        return self.gblup(genome, train_indices, validation_indices)

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
        super(GblupParallelEvaluator, self).__exit__(exc_type, exc_val, exc_tb)

    def genomes_to_evaluate(self, population):
        """
        Get the genomes that we haven't evaluated yet.
        :param population: tblup.Population.
        :return: list of lists
        """
        to_evaluate = []
        indices = []
        for i, indv in enumerate(population):
            unique = sorted(set(indv.genome))

            as_str = str(unique)
            if as_str in self.archive:
                indv.fitness = self.archive[as_str]

            else:
                indices.append(i)
                to_evaluate.append(unique)

        return to_evaluate, indices

    def evaluate(self, population):
        """
        Evaluate the population with GBLUP.
        :param population: tblup.Population.
        :return:
        """
        super(GblupParallelEvaluator, self).evaluate(population)

        to_evaluate, indices = self.genomes_to_evaluate(population)

        results = []
        for genome in to_evaluate:
            results.append(
                self.pool.apply_async(self, (genome, population.generation))
            )

        for i, idx in enumerate(indices):
            population[idx].fitness = results[i].get()
            self.archive[str(to_evaluate[i])] = population[idx].fitness

        return population


class InterGCVGblupParallelEvaluator(GblupParallelEvaluator):
    """
    Same as GblupParallelEvaluator, but with cross-validation between generations.
    This means that the validation set is changed each generation.

    InterGCV = intergenerational cross validation, or between generations.

    Only overrides the constructor and train_validation_indices.
    """
    def __init__(self, data_path, labels_path, r, n_procs=-1, n_folds=5):
        """
        See parent doc.
        :param n_folds: int, number of cross validation folds.
        """
        super(GblupParallelEvaluator, self).__init__(data_path, labels_path, r, n_procs=n_procs)

        self.n_folds = n_folds
        self.fold_indices = self.make_fold_indices(self.indices, self.n_folds)

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


class IntraGCVGblupParallelEvaluator(InterGCVGblupParallelEvaluator):
    """
    Same as GblupParallelEvaluator, but with k-fold cross-validation within a fitness evaluation.

    IntraGCV = intragenerational cross validation, or within generations.

    Only overrides the call method.
    """
    def __init__(self, data_path, labels_path, r, n_procs=-1, n_folds=5):
        """
        See parent doc.
        """
        super(IntraGCVGblupParallelEvaluator, self).__init__(data_path, labels_path, r, n_procs=n_procs,
                                                             n_folds=n_folds)

    def __call__(self, genome, generation):
        """
        Call magic method, assign fitness to genome.
        Does k-fold cross-validated GBLUP.
        :param genome: list, list of indexes into data matrix.
        :param generation: int, current generation. Not used for this override.
        :return: float, fitness of genome. Average Pearson's R over the folds.
        """
        sum = 0
        for k in range(self.n_folds):
            train_indices, validation_indices = self.train_validation_indices(k)
            sum += self.gblup(genome, train_indices, validation_indices)

        return sum / self.n_folds
