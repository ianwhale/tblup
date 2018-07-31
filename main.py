import numpy
import random
from tblup.config import parser
from pprint import PrettyPrinter
from tblup.utils import build_kwargs
from tblup.population import Population
from tblup.local import get_local_search


def main():
    """
    Main entry point.
    """
    args = parser.parse_args()

    PrettyPrinter(indent=4).pprint({k: v for k, v in args.__dict__.items() if not k.isupper()})

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

        population.monitor.save_archive(population)

    if args.local_search is not None:
        genome, fitness = get_local_search(args, population).search()

        population.monitor.report_local(genome, fitness)


if __name__ == '__main__':
    main()
