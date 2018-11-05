import math
import unittest
import numpy as np
from tblup import Evaluator, IndexIndividual, Monitor, DifferentialEvolutionSelector, Population
from tblup import get_evolver, get_scheduler
from tblup.config import parser


def ackley(genome):
    """Ackley function, https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html"""
    def _ackley(genome):
        genome = np.clip(genome, -32, 32)
        sum_1 = 0.0
        sum_2 = 0.0
        for c in genome:
            sum_1 += c ** 2.0
            sum_2 += math.cos(2.0 * math.pi * c)
        n = float(len(genome))
        return -20.0 * math.exp(-0.2 * math.sqrt(sum_1 / n)) - math.exp(sum_2 / n) + 20 + math.e
    return -1 * _ackley(genome)


def rastrigin(genome):
    """Rastrigin function."""
    def _rastrigin(genome):
        genome = np.clip(genome, -5.12, 5.12)
        s = 0.0
        for g in genome:
            s += g ** 2 - 10 * math.cos(2 * math.pi * g)
        return len(genome) * 10 + s

    return -1 * _rastrigin(genome)


class TestIndividual(IndexIndividual):
    def __init__(self, length, dimensionality, ipr=(-32, 32)):
        super(TestIndividual, self).__init__(length, dimensionality)
        self._genome = np.random.rand(length) * (ipr[1] - ipr[0]) + ipr[0]

    def set_ipr(self, ipr):
        """ Set initial parameter range. """
        self._genome = np.random.rand(self.length) * (ipr[1] - ipr[0]) + ipr[0]

    @property
    def genome(self):
        return self._genome


class TestFunctionEvaluator(Evaluator):
    def __init__(self, data_path, labels_path, func, eval_limit):
        super(TestFunctionEvaluator, self).__init__(data_path, labels_path)
        self.archive = {}
        self.evals = 0
        self.eval_limit = eval_limit
        self.func = func

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, exc_tb): pass

    def genomes_to_evaluate(self, population): pass

    def evaluate(self, previous_population, population, generation):
        for indv in population:
            if indv.uid in self.archive:
                continue
            else:
                self.archive[indv.uid] = True

            indv.fitness = self.func(indv.genome)

            self.evals += 1

        if self.evals >= self.eval_limit:
            print("Reached eval limit...")
            raise KeyboardInterrupt


class TestMonitor(Monitor):
    """Monitor that does nothing in order to prevent unnecessary outputs."""
    def make_subdir(self, args, function_name="test"): return function_name


class SystemTest(unittest.TestCase):
    def do_test_function(self, dimension, pop_size, cr, f, func, eval_limit, ipr, vtr):
        """
        :param dimension: int, dimension of problem.
        :param pop_size: int, population size.
        :param cr: float, [0, 1], crossover rate.
        :param f: float, 0 < f, mutation rate.
        :param func: callable, evaluation function,
        :param eval_limit: int, 0 < eval_limit, number of function evaluations allowed.
        :param ipr: tuple, initial parameter range.
        :param vtr: float, value to reach.
        """
        print("*** Starting {} Test ***".format(str.title(func.__name__)))
        args = parser.parse_args()

        dimension = dimension
        generations = 100000  # Not actually used, termination based on eval_limit in TestFunctionEvaluator.

        args.population_size = pop_size
        args.crossover_rate = cr
        args.mutation_intensity = f
        args.dimensionality = dimension

        d = {
            "evolver": get_evolver(args),
            "evaluator": TestFunctionEvaluator("./system.py", "./system.py", func, eval_limit),
            "selector": DifferentialEvolutionSelector(),
            "individual": TestIndividual,
            "scheduler": get_scheduler(args),
            "length": dimension,
            "dimensionality": dimension,
            "num_individuals": args.population_size,
            "monitor": TestMonitor(args),
        }

        population = Population(**d)

        for indv in population:
            indv.set_ipr(ipr)

        try:
            for g in range(1, generations):
                population.do_generation()

                if g % 1000 == 0:
                    print("Best at generation {}: {}".format(g, max(population, key=lambda x: x.fitness).fitness))

        except KeyboardInterrupt:
            print("Best at keyboard interrupt: {}".format(max(population, key=lambda x: x.fitness).fitness))

        best = max(population, key=lambda x: x.fitness).fitness

        print("Generations: {}".format(population.generation))
        print("Function evaluations: {}".format(population.evaluator.evals))

        print("Value to reach: {}".format(vtr))
        print("Beyond value to reach: {}".format(vtr <= best))

        self.assertGreaterEqual(best, vtr, "{}: DE Converges to a point better than the value to reach."
                                .format(func.__name__))

    #
    # Problem setups and definitions taken from the original Storn and Price paper.
    # Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over
    #   continuous spaces,”Journal of Global Optimization, vol. 11, no. 4, pp.341–359, Dec 1997.
    #

    def test_ackley(self):
        self.do_test_function(dimension=100,
                              pop_size=50,
                              cr=0.1,
                              f=0.5,
                              func=ackley,
                              eval_limit=37000,
                              ipr=(-32, 32),
                              vtr=-1 * math.e ** -3)

    def test_rastrigin(self):
        self.do_test_function(dimension=100,
                              pop_size=20,
                              cr=0,
                              f=0.5,
                              func=rastrigin,
                              eval_limit=75000,
                              ipr=(-5.12, 5.12),
                              vtr=-0.9)


if __name__ == '__main__':
    unittest.main()
