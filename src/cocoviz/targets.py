"""Functions to generate targets for specific indicators"""

import numpy as np
import polars as pl

from . import indicator as ind
from .result import ProblemDescription, ResultSet


def log_targets(
    results: ResultSet,
    indicator: ind.Indicator | str,
    number_of_targets: int = 101,
    min_target: float = -8.0,
) -> dict[ProblemDescription, np.ndarray]:
    """Generate log-spaced targets for `indicator` for each problem in `results`.

    Parameters
    ----------
    results : ResultSet
        Collection of results to derive per-problem target ranges from.
    indicator : Indicator or str
        Performance indicator to generate targets for.
    number_of_targets : int, optional
        Number of target values to generate for each problem.
    min_target : float, optional
        Exponent of the smallest (relative) target, i.e. targets are
        spaced logarithmically between `10 ** min_target` and `1` of the
        observed indicator range.

    Returns
    -------
    dict[ProblemDescription, np.ndarray]
        Target values for each problem in `results`.
    """
    indicator = ind.resolve(indicator)
    mul = np.logspace(min_target, 0, number_of_targets)

    targets = {}
    for desc, problem_results in results.by_problem():
        indicator_values = pl.concat([r._data[indicator.name] for r in problem_results])
        low = indicator_values.min()
        high = indicator_values.max()
        delta = high - low

        if low == high:
            targets[desc] = np.linspace(low, high, 1)
        elif indicator.larger_is_better:
            targets[desc] = low + delta * mul
        else:
            targets[desc] = high - delta * mul
    return targets


def linear_targets(
    results: ResultSet, indicator: ind.Indicator | str, number_of_targets: int = 101
) -> dict[ProblemDescription, np.ndarray]:
    """Generate linearly spaced targets for `indicator` for each problem in `results`.

    Parameters
    ----------
    results : ResultSet
        Collection of results to derive per-problem target ranges from.
    indicator : Indicator or str
        Performance indicator to generate targets for.
    number_of_targets : int, optional
        Number of target values to generate for each problem.

    Returns
    -------
    dict[ProblemDescription, np.ndarray]
        Target values for each problem in `results`.
    """
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
    """Use every unique observed value of `indicator` as a target, for each problem in `results`.

    Parameters
    ----------
    results : ResultSet
        Collection of results to derive per-problem targets from.
    indicator : Indicator or str
        Performance indicator to generate targets for.

    Returns
    -------
    dict[ProblemDescription, np.ndarray]
        Sorted, unique target values for each problem in `results`
        (ascending if `indicator.larger_is_better`, descending otherwise).
    """
    indicator = ind.resolve(indicator)
    targets = {}
    for desc, problem_results in results.by_problem():
        indicator_values = pl.concat([r._data[indicator.name] for r in problem_results])
        if indicator.larger_is_better:
            targets[desc] = indicator_values.unique().sort()
        else:
            targets[desc] = indicator_values.unique().sort(descending=True)
    return targets
