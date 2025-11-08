"""Functions to generate targets for specific indicators"""

import numpy as np
import polars as pl

from . import indicator as ind
from .result import ProblemDescription, ResultSet


def log_targets(
    results: ResultSet, indicator: ind.Indicator | str, number_of_targets: int = 101
) -> dict[ProblemDescription, np.ndarray]:
    indicator = ind.resolve(indicator)

    targets = {}
    for desc, problem_results in results.by_problem():
        indicator_values = pl.concat([r._data[indicator.name] for r in problem_results])
        low = indicator_values.min()
        high = indicator_values.max()
        delta = high - low

        mul = np.logspace(-16, 0, number_of_targets)
        if low == high:
            targets[desc] = np.linspace(low, high, 1)
        elif indicator.larger_is_better:
            targets[desc] = low + delta * mul
        else:
            targets[desc] = np.flip(low + delta * mul)
    return targets


def linear_targets(
    results: ResultSet, indicator: ind.Indicator | str, number_of_targets: int = 101
) -> dict[ProblemDescription, np.ndarray]:
    indicator = ind.resolve(indicator)

    targets = {}
    for desc, problem_results in results.by_problem():
        indicator_values = pl.concat([r._data[indicator.name] for r in problem_results])
        low = indicator_values.min()
        high = indicator_values.max()

        # If the indicator is constant, only generate one target.
        if low == high:
            targets[desc] = np.linspace(low, high, 1)
        elif indicator.larger_is_better:
            targets[desc] = np.linspace(low, high, number_of_targets)
        else:
            targets[desc] = np.linspace(high, low, number_of_targets)

    return targets


def full_targets(results: ResultSet, indicator: ind.Indicator | str) -> dict[ProblemDescription, np.ndarray]:
    indicator = ind.resolve(indicator)
    targets = {}
    for desc, problem_results in results.by_problem():
        indicator_values = pl.concat([r._data[indicator.name] for r in problem_results])
        if indicator.larger_is_better:
            targets[desc] = indicator_values.unique().sort()
        else:
            targets[desc] = indicator_values.unique().sort(descending=True)
    return targets
