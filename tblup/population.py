import pickle


class Population:
    def __init__(self, evolver, evaluator, selector, individual, scheduler, length,
                 dimensionality, num_individuals, monitor, seeded_initial=None):
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
        :param seeded_initial: list, list of tblup.Individuals with some desirable initial properties.
        """
        self.evolver = evolver
        self.evaluator = evaluator
        self.selector = selector
        self.scheduler = scheduler

        if seeded_initial:
            self.population = seeded_initial
        else:
            self.population = [individual(length, dimensionality) for _ in range(num_individuals)]

        self.dimensionality = dimensionality
        self.monitor = monitor
        self.generation = 0

        # Gather statistics on initial population.
        self.evaluator.evaluate(self)
        self.monitor.report(self)

        self.generation += 1

    def __getitem__(self, index):
        return self.population[index]

    def __len__(self):
        return len(self.population)

    def do_generation(self):
        next_pop = self.evolver.evolve(self)
        self.evaluator.evaluate(next_pop)
        self.population = self.selector.select(self, next_pop)

        # If we need to increase the individual length, we also must reevaluate.
        if self.scheduler.do_step(self, self.generation):
            self.scheduler.step(self)
            self.evaluator.evaluate(self)

        self.monitor.report(self)
        self.generation += 1
