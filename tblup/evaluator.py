import os
import abc
import random
import numpy as np
import multiprocessing as mp
from math import sqrt
from tblup import make_grm
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def get_evaluator(args):
    """
    Gets the evaluator type corresponding to a string.
    :param args: object, argparse.Namespace.
    :return: tblup.Evaluator
    """
    splitter = None

    if args.splitter == "pca":
        # Decorate the splitter function so we only have to pass in the data later.
        splitter = lambda data: pca_splitter(data, outliers=args.pca_outliers)

    if args.removal_r is None:
        r = args.features
    else:
        r = args.removal_r

    constructor_args = [args.geno, args.pheno, args.heritability]
    constructor_kwargs = {'n_procs': args.processes, 'splitter': splitter, 'snp_remover': SNPRemovalHandler(
        r, args.h2_alpha, args.heritability, args.remove_snps
    )}

    regressor = None

    if args.regressor == args.REGRESSOR_TYPE_BLUP:
        regressor = BlupParallelEvaluator(*constructor_args, **constructor_kwargs)

    if args.regressor == args.REGRESSOR_TYPE_INTRACV_BLUP:
        constructor_kwargs['n_folds'] = args.cv_folds
        regressor = IntraGCVBlupParallelEvaluator(*constructor_args, **constructor_kwargs)

    if args.regressor == args.REGRESSOR_TYPE_INTERCV_BLUP:
        constructor_kwargs['n_folds'] = args.cv_folds
        regressor = InterGCVBlupParallelEvaluator(*constructor_args, **constructor_kwargs)

    if args.regressor == args.REGRESSOR_TYPE_MONTECV_BLUP:
        regressor = MonteCarloCVBlupParallelEvaluator(*constructor_args, **constructor_kwargs)

    if regressor is None:
        raise NotImplementedError("Regressor described by {} not implemented.".format(args.regressor))

    return regressor


#################################################
#                  Evaluators                   #
#################################################


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
        self.n_procs = n_procs
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.consumers = []

    def __enter__(self):
        """
        Create the multiprocessing pool.
        Only happens once to reduce overhead.
        Credit:
            https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/
        """
        for _ in range(self.n_procs):
            p = mp.Process(target=self.worker, args=(self.in_queue, self.out_queue, self.data_path, self.labels_path))
            p.daemon = True
            p.start()
            self.consumers.append(p)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for c in self.consumers:
            c.terminate()

    @staticmethod
    @abc.abstractmethod
    def worker(in_queue, out_queue, data_path, labels_path):
        """Worker static and abstract class."""
        pass

    def genomes_to_evaluate(self, population):
        raise NotImplementedError()

    def evaluate(self, population, generation):
        """
        :param population: tblup.Population
        :param generation: int
        """
        if len(self.consumers) == 0:
            raise AttributeError("Workers are not set up.")

        pass


class BlupParallelEvaluator(ParallelEvaluator):
    """
    Evaluates individuals using multiprocessing and genomic best linear unbiased predictor.

    Expects the individuals to have nonnegative integer valued genomes.
    """

    TRAIN_TEST_SPLIT = 0.8   # 80% of the data will be training, 20% will be testing.
    TRAIN_VALID_SPLIT = 0.8  # Of the training data, 20% will be validation.

    def __init__(self, data_path, labels_path, h2, n_procs=-1, splitter=None, snp_remover=None):
        """
        Constructor.
        :param data_path: string, path to the training data.
        :param labels_path: string, path to the labels for the training data.
        :param h2: float, trait heritability.
        :param n_procs: int, number of processes to use.
        :param splitter: callable | None, a special function to split the data into training and testing (optional).
            - If not provided, we just shuffle and use a specified split.
        :param snp_remover: tblup.SNPRemovalHandler, optional.
        """
        super(BlupParallelEvaluator, self).__init__(data_path, labels_path, n_procs=n_procs)

        # Store individuals we have evaluated already.
        # Indexing works as the hashed frozenset of the list => fitness.
        self.archive = {}
        self.snp_remover = snp_remover
        self.h2 = h2  # Used for the regularization parameter.

        # Build training and testing indices.
        data = np.load(data_path)
        shape = data.shape
        self.n_samples, self.n_columns = shape[0], shape[1]

        if splitter:
            self.training_indices, self.testing_indices = splitter(data)

        else:
            indices = random.sample(range(self.n_samples), self.n_samples)
            self.training_indices, self.testing_indices = train_test_split(indices,
                                                                           train_size=self.TRAIN_TEST_SPLIT,
                                                                           test_size=1 - self.TRAIN_TEST_SPLIT)

        self.training_indices, self.validation_indices = train_test_split(self.training_indices,
                                                                          train_size=self.TRAIN_VALID_SPLIT,
                                                                          test_size=1 - self.TRAIN_VALID_SPLIT)

    @staticmethod
    def worker(in_queue, out_queue, data_path, labels_path):
        """
        Deamon worker that only loads the data once.
        :param in_queue: mp.Queue, input arguments.
        :param out_queue: mp.Queue, pipe to outside world.
        :param data_path: string, path to data.
        :param labels_path: straing, path to labels.
        :return:
        """
        data = np.load(data_path)
        labels = np.load(labels_path)

        while True:
            index, blup_kwargs = in_queue.get()

            blup_kwargs['data'] = data
            blup_kwargs['labels'] = labels
            fitness = BlupParallelEvaluator.blup(**blup_kwargs)

            out_queue.put((index, fitness))

    def enqueue(self, index, indices, train_indices, validation_indices):
        """
        Put a job into the queue.
        Must be a new dictionary every time due to mutability issues.
        :param index: int, index of individual.
        :param indices: np.array, genome of individual.
        :param train_indices: np.array.
        :param validation_indices: np.array.
        """
        self.in_queue.put((index, {
            'h2': self.h2,
            'train_indices': train_indices,
            'validation_indices': validation_indices,
            'indices': indices
        }))


    @staticmethod
    def blup(indices, train_indices, validation_indices, data, labels, h2):
        """
        Do BLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.

        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :param validation_indices: list, list of ints corresponding to which samples to use for validation.
        :param data: np.array, data matrix.from tblup.sharearray import sharearray
        :param labels: np.array, label vector.
        :param h2: float, heritability.
        :return: float, prediction accuracy.
        """
        if len(indices) > data.shape[0]:
            # Do GBLUP. The GRM will be more efficient since we have more columns than samples.
            return BlupParallelEvaluator.gblup(indices, train_indices, validation_indices, data, labels, h2)

        else:
            # Do SNP-BLUP. Calculating the GRM is more costly in this case, so we just do normal ridge regression.
            return BlupParallelEvaluator.snp_blup(indices, train_indices, validation_indices, data, labels, h2)

    @staticmethod
    def gblup(indices, train_indices, validation_indices, data, labels, h2):
        """
        Do GBLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.

        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :param validation_indices: list, list of ints corresponding to which samples to use for validation.
        :return: float, prediction accuracy.
        """
        G = make_grm(data[:, indices])

        r = (1 - h2) / h2

        # Inverse the matrix using only the desired training samples.
        G_inv = G[train_indices, :][:, train_indices]
        G_inv.flat[:: G_inv.shape[0] + 1] += r  # Add regularization term to the diagonal of G.
        G_inv = np.linalg.inv(G_inv)

        prediction = np.matmul(np.matmul(G[:, train_indices], G_inv), labels[train_indices])

        return abs(pearsonr(labels[validation_indices], prediction[validation_indices])[0])

    @staticmethod
    def snp_blup(indices, train_indices, validation_indices, data, labels, h2):
        """
        Do SNP-BLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.

        :param indices: list, list of ints corresponding to the features indices to use.
        :param train_indices: list, list of ints corresponding to which samples to use for training.
        :param validation_indices: list, list of ints corresponding to which samples to use for validation.
        :return: float, prediction accuracy.
        """
        X = data[:, indices]
        y = labels

        X_train, X_valid = X[train_indices], X[validation_indices]
        y_train, y_valid = y[train_indices], y[validation_indices]

        p = np.mean(X_train, axis=0) / 2
        d = 2 * np.sum(p * (1 - p))
        r = (1 - h2) / (h2 / d)

        X_train -= (2 * p)
        X_valid -= (2 * p)

        clf = Ridge(alpha=r)
        clf.fit(X_train, y_train)

        return abs(pearsonr(clf.predict(X_valid), y_valid)[0])

    def train_validation_indices(self, generation):
        """
        Just splits the data into two splits.
        :param generation: int, current generation. Not used here, used in subclasses.
        :return: (list, list), tuple of list of ints, (training, validation) indices.
        """
        return self.training_indices, self.validation_indices

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

    def genomes_to_evaluate(self, population):
        """
        Get the genomes that we have not evaluated yet.
        :param population: tblup.Population.
        :return: list of lists
        """
        if self.snp_remover is not None and self.snp_remover.should_remove():
            return self.snp_remover.genomes_to_evaluate(population, self.archive)

        else:
            to_evaluate = []
            indices = []

            for i, indv in enumerate(population):
                if indv.uid not in self.archive:
                    indices.append(i)
                    to_evaluate.append(indv.genome)

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
        train_indices, validation_indices = self.train_validation_indices(generation)

        # Send the blup keyword arguments to the waiting worker process.
        for genome, index in zip(to_evaluate, indices):
            self.enqueue(index, genome, train_indices, validation_indices)

        # Get the results from the output queue.
        results = []
        while len(results) != len(to_evaluate):
            results.append(self.out_queue.get())

        # Assign recently calculated fitnesses.
        for index, fitness in results:
            population[index].set_fitness(fitness)
            self.archive[population[index].uid] = population[index].fitness

        return population

    def evaluate_testing(self, population):
        """
        Evaluate the whole population's testing accuracy.
        :param population: tblup.Population.
        :return: list, list of testing accuracies.
        """
        train = np.concatenate((self.training_indices, self.validation_indices))
        # Put all individuals to be evaluated for testing accuracy.
        for index, individual in enumerate(population):
            self.enqueue(index, self.snp_remover.combine_with_removed(individual.genome), train, self.testing_indices)

        # Get results.
        results = []
        while len(results) != len(population):
            results.append(self.out_queue.get())

        # Make sure things are in order.
        results.sort(key=lambda x: x[0])
        _, fitnesses = zip(*results)

        return list(fitnesses)


class InterGCVBlupParallelEvaluator(BlupParallelEvaluator):
    """
    Same as BlupParallelEvaluator, but with cross-validation between generations.
    This means that the validation set is changed each generation.

    InterGCV = intergenerational cross validation, or between generations.

    Only overrides the constructor and train_validation_indices.
    """
    def __init__(self, data_path, labels_path, h2, n_procs=-1, n_folds=5, splitter=None, snp_remover=None):
        """
        See parent doc.
        :param n_folds: int, number of cross validation folds.
        """
        super(InterGCVBlupParallelEvaluator, self).__init__(data_path, labels_path, h2, n_procs=n_procs,
                                                            splitter=splitter, snp_remover=snp_remover)

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

        # Prebuild the index pairs so we don't have to at run time.
        prebuilt = [[[], []] for _ in range(n_folds)]
        for i in range(n_folds):
            prebuilt[i][1] = fold_indices[i]

            for j in range(n_folds):
                if j != i:
                    prebuilt[i][0] += fold_indices[j]

        return prebuilt

    def train_validation_indices(self, generation):
        """
        Overriden method, changes the validation split every generation.
        :param generation: int, current generation.
        :return: (list, list), tuple of list of ints, (training, validation) indices.
        """
        return self.fold_indices[generation % self.n_folds]


class IntraGCVBlupParallelEvaluator(InterGCVBlupParallelEvaluator):
    """
    Same as BlupParallelEvaluator, but with k-fold cross-validation within a fitness evaluation.

    IntraGCV = intragenerational cross validation, or within generations.

    Only overrides the call method.
    """
    def __init__(self, data_path, labels_path, h2, n_procs=-1, n_folds=5, splitter=None, snp_remover=None):
        """
        See parent doc.
        """
        super(IntraGCVBlupParallelEvaluator, self).__init__(data_path, labels_path, h2, n_procs=n_procs,
                                                            n_folds=n_folds, splitter=splitter, snp_remover=snp_remover)

    def evaluate(self, population, generation):
        """
        Do BLUP, but n_folds times.
        :param population: tblup.Population
        :param generation: int, current generation.
        :return: tblup.Population.
        """
        to_evaluate, indices = self.genomes_to_evaluate(population)
        sums = {i: 0 for i in indices}

        for k in range(self.n_folds):
            train_indices, validation_indices = self.train_validation_indices(k)

            for genome, index in zip(to_evaluate, indices):
                self.enqueue(index, genome, train_indices, validation_indices)

            results = []
            while len(results) != len(to_evaluate):
                results.append(self.out_queue.get())

            for index, fitness in results:
                sums[index] += fitness

        for index, fitness_sum in sums.items():
            population[index].set_fitness(fitness_sum / self.n_folds)
            self.archive[population[index].uid] = population[index].fitness

        return population

    def __call__(self, genome, generation):
        """
        Call magic method, assign fitness to genome.
        Does k-fold cross-validated BLUP.
        :param genome: list, list of indexes into data matrix.
        :param generation: int, current generation. Not used for this override.
        :return: float, fitness of genome. Average Pearson's R over the folds.
        """
        fitness_sum = 0
        x, y = self.data, self.labels
        for k in range(self.n_folds):
            train_indices, validation_indices = self.train_validation_indices(k)
            fitness_sum += self.blup(genome, train_indices, validation_indices, x, y, self.h2)

        return np.asscalar(fitness_sum / self.n_folds)


class MonteCarloCVBlupParallelEvaluator(BlupParallelEvaluator):
    """
    Uses Monte Carlo cross-validation in the fitness function.

    Only overrides the train_validation_indices method.
    """
    def __init__(self, data_path, labels_path, h2, n_procs=-1, splitter=None, snp_remover=None):
        """
        See parent doc.
        """
        super(MonteCarloCVBlupParallelEvaluator, self).__init__(data_path, labels_path, h2, n_procs=n_procs,
                                                                splitter=splitter, snp_remover=snp_remover)

        self.indices = np.concatenate((self.training_indices, self.validation_indices))

    def train_validation_indices(self, generation):
        """
        Generates a random training and validation set.
        :param generation: int, current generation (not used).
        :return: (list, list), (training indices, validation indices)
        """
        return train_test_split(self.indices, test_size=0.2)


class SNPRemovalHandler:
    """
    Mixin to handle removing SNPs without reproducing too much code.
    """
    def __init__(self, r, alpha, h2, remove_snps):
        """
        Constructor.
        :param r: int, number of SNPs to remove.
        :param alpha: float, used to make threshold = h(1 + alpha)
        :param h2: float, heritability.
        :param remove_snps: bool, true if we should remove SNPs at the threshold.
        """
        self.r = r
        self.threshold = sqrt(h2) * (1 + alpha)
        self.removed = np.array([])
        self.remove_snps = remove_snps

    def should_remove(self):
        return self.remove_snps

    def genomes_to_evaluate(self, population, archive):
        """
        Same as BlupEvaluator.genomes_to_evaluate
        :param population: tblup.Population.
        :param archive: dict, archive from evaluator.
        :return:
        """
        to_evaluate = []
        indices = []

        best = max(population, key=lambda x: x.fitness)
        if best.fitness > self.threshold:
            remove = best.genome[-self.r:]  # Get the r "best" SNP indices to remove
            self.removed = np.union1d(self.removed, remove)

        for i, indv in enumerate(population):
            if best.fitness > self.threshold and indv.uid in archive:
                del archive[indv.uid]  # Archived fitness is now invalid.

            if indv.uid not in archive:
                diff = np.setdiff1d(indv.genome, self.removed)

                if len(diff) == 0:  # We removed all the indices, so the fitness is 0 (this will error if evaluated).
                    archive[indv.uid] = 0.0
                    indv.set_fitness(0.0)
                else:
                    indices.append(i)
                    to_evaluate.append(diff)

        return to_evaluate, indices

    def combine_with_removed(self, genome):
        """
        Combine the given genome with the indices that have been removed.
        :param genome: np.array
        :return: np.array
        """
        return [int(i) for i in np.union1d(genome, self.removed)]


#################################################
#                  Splitters                    #
#################################################


def pca_splitter(data, split=0.8, outliers=False):
    """
    PCA Splitter. Splits based on inliers or outliers when GRM projected into 2 dimensions.

    :param data: np.ndarray, data matrix.
    :param split: float, what percent of the data to use as training.
    :param outliers: bool, true for training set to be the outliers.
    """
    proj = PCA(n_components=2)
    x = proj.fit_transform(make_grm(data))

    mu = np.mean(x, axis=0)

    dists = (x - mu) ** 2
    dists = dists[:, 0] + dists[:, 1]

    idx_dist = [(i, dists[i]) for i in range(len(dists))]
    idx_dist.sort(key=lambda tup: tup[1], reverse=outliers)

    idxs = [x[0] for x in idx_dist]
    n = int(len(idxs) * split)

    return idxs[:n], idxs[n:]
