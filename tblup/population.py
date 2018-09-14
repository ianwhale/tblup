from math import sqrt
from tblup.utils import get_stop_condition


class Population:

    ARCHIVE_INTERVAL = 100

    def __init__(self, evolver, evaluator, selector, individual, scheduler, length,
                 dimensionality, num_individuals, monitor, seeded_initial=None, record_testing=False,
                 h2_stop_condition=None):
        """
        Constructor.
        :param evolver: tblup.Evolver.
        :param evaluator: tblup.Evaluator.
        :param selector: tblup.Selector.
        :param individual: callable, tblup.Individual.__init__.
        :param scheduler: tblup.Scheduler.
        :param length: int, length of an individual's genome.
        :param dimensionality: int, dimensionality of the problem.
        :param num_individuals: int, number of individuals in the population.
        :param monitor: tblup.Monitor, gathers population statistics.
        :param seeded_initial: iterable, tblup.Individuals with some desirable initial properties.
        :param record_testing: bool, True to record testing accuracy during search.
        :param h2_stop_condition: string, see tblup.utils.translate_stop_condition
        """
        self.evolver = evolver
        self.monitor = monitor
        self.selector = selector
        self.evaluator = evaluator
        self.scheduler = scheduler

        if seeded_initial is not None:
            self.population = []

            for _ in range(num_individuals):
                self.population.append(individual(length, dimensionality, genome=next(seeded_initial)))

        else:
            self.population = [individual(length, dimensionality) for _ in range(num_individuals)]

        self.record_testing = record_testing
        self.dimensionality = dimensionality
        self.h2_stop_condition = h2_stop_condition
        self.h2_threshold = self.evaluator.h2 / sqrt(self.evaluator.h2)
        self.generation = 0

        # Gather statistics on initial population.
        self.evaluator.evaluate(self, self.generation)
        self.monitor.report(self)
        self.monitor.save_archive(self)

        if self.record_testing:
            self.monitor.report_testing(self)

        self.generation += 1

    def __getitem__(self, index):
        return self.population[index]

    def __len__(self):
        return len(self.population)

    def do_generation(self):
        """
        Does a generational step.
        :return: bool, True if the search should continue, False if not.
        """
        next_pop = self.evolver.evolve(self)
        self.evaluator.evaluate(next_pop, self.generation)
        self.population = self.selector.select(self, next_pop)

        # If we need to increase the individual length, we also must reevaluate.
        if self.scheduler.should_step(self, self.generation):
            self.scheduler.step(self)
            self.evaluator.evaluate(self, self.generation)

        stats = self.monitor.report(self)

        if self.generation % self.ARCHIVE_INTERVAL == 0:
            self.monitor.save_archive(self)

        if self.record_testing:
            self.monitor.report_testing(self)

        self.generation += 1

        if self.h2_stop_condition is not None:
            if get_stop_condition(self.h2_stop_condition, stats, self.monitor) > self.h2_threshold:
                return False

        return True

