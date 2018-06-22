import numpy
import random
from tblup.config import parser
from pprint import PrettyPrinter
from tblup.utils import build_kwargs
from tblup.population import Population


def main():
    """
    Main entry point.
    """
    args = parser.parse_args()

    PrettyPrinter(indent=4).pprint(args.__dict__)

    random.seed(args.seed)
    numpy.random.seed(args.seed)

    kwargs = build_kwargs(args)
    evaluator = kwargs['evaluator']

    with evaluator:
        population = Population(**kwargs)

        for gen in range(1, args.generations + 1):
            population.do_generation()

        results = evaluator.evaluate_testing(population)
        population.monitor.write(
            ["Testing"] + population.monitor.get_row_summary(results) + ["Final"]
        )

        if args.individual == "nullable":
            print(population[int(numpy.argmax(results))].genome)


if __name__ == '__main__':
    main()
