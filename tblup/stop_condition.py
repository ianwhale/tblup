from math import sqrt


def get_stop_condition(args):
    """
    Get the stop condition class.
    :param args: argpase.Namespace
    :return: StopCondition
    """
    if args.stop_condition in HeritabilityStopCondition.H2_CONDITIONS:
        return HeritabilityStopCondition(args.heritability, args.h2_alpha, args.stop_condition)

    return StopCondition()


class StopCondition:
    def should_stop(self, population, stats):
        """Always returns false."""
        return False


class HeritabilityStopCondition(StopCondition):

    CONDITION_MAX = "h2_max"
    CONDITION_MIN = "h2_min"
    CONDITION_MEDIAN = "h2_median"
    CONDITION_MEAN = "h2_mean"

    H2_CONDITIONS = [CONDITION_MAX, CONDITION_MIN, CONDITION_MEDIAN, CONDITION_MEAN]

    def __init__(self, h2, alpha, condition):
        """
        :param h2: float, heritability.
        :param alpha: float, [0, 1) moves the threshold up by some value.
        :param condition: string, what condition to stop on.
        """
        self.threshold = sqrt(h2) * (1 + alpha)
        self.condition = condition

    def should_stop(self, population, stats):
        """
        If the condition (max, min, etc.) is above h2/sqrt(h2), tell the search to stop.
        :return: bool, True if the search should stop.
        """
        index = HeritabilityStopCondition.translate_condition_string(self.condition, population.monitor)
        return stats[index] > self.threshold

    @staticmethod
    def translate_condition_string(condition_string, monitor):
        if condition_string == HeritabilityStopCondition.CONDITION_MAX:
            return monitor.MAX_FITNESS_INDEX

        if condition_string == HeritabilityStopCondition.CONDITION_MIN:
            return monitor.MIN_FITNESS_INDEX

        if condition_string == HeritabilityStopCondition.CONDITION_MEAN:
            return monitor.MEAN_FITNESS_INDEX

        if condition_string == HeritabilityStopCondition.CONDITION_MEDIAN:
            return monitor.MEDIAN_FITNESS_INDEX

        raise NotImplementedError("Heritability stopping condition {} not implemented.".format(condition_string))
