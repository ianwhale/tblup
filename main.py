import numpy
import random
from tblup.config import parser
from tblup.utils import build_kwargs
from tblup.population import Population


def main():
    """
    Main entry point.
    """
    args = parser.parse_args()

    random.seed(args.seed)
    numpy.random.seed(args.seed)

    kwargs = build_kwargs(args)
    evaluator = kwargs['evaluator']

    with evaluator:
        population = Population(**kwargs)

        for gen in range(1, args.generations + 1):
            population.do_generation()

        population.monitor.save_archive(population)

        results = evaluator.evaluate_testing(population)
        population.monitor.write(
            ["Testing"] + population.monitor.get_row_summary(results) + ["Final"]
        )


if __name__ == '__main__':
    main()
