import abc


class Selector(abc.ABC):
    """
    Abstract base class for a selection scheme.
    """
    @abc.abstractmethod
    def select(self, population, offspring):
        raise NotImplementedError()


class DifferentialEvolutionSelector(Selector):
    """
    Standard selector for differential evolution.
    Rule: offspring replace parents if their fitness is higher.
    """
    def select(self, population, offspring):
        """
        Do selection, if the child's fitness is better they replace the parent.
        :param population: tblup.Population, the population.
        :param offspring: list, list of tblup.Individuals.
        :return: list
        """
        new_population = []

        for parent, child in zip(population, offspring):
            if child.fitness > parent.fitness:
                new_population.append(child)

            else:
                new_population.append(parent)

        return new_population
