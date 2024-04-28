"""Functions to generate targets for specific indicators"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from .types import ProblemDescription, ResultSet


def linear_targets(
    results: ResultSet, indicator: str, number_of_targets: int = 101
) -> dict[ProblemDescription, ArrayLike]:
    targets = {}
    for desc, problem_results in results.by_problem():
        indicator_values = pl.concat([r._data for r in problem_results])[indicator]
        low = indicator_values.min()
        high = indicator_values.max()

        # If the indicator is constant, only generate one target.
        if low == high:
            targets[desc] = np.linspace(low, high, 1)
        else:
            targets[desc] = np.linspace(low, high, number_of_targets)

    return targets
