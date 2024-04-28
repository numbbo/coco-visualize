"""Expected Runtime calculations"""

import matplotlib.pyplot as plt
import polars as pl
import scipy.stats as stats

from .targets import linear_targets
from .types import ResultSet


def ertvalues(
    results: ResultSet,
    indicator: str,
    number_of_targets: int = 101,
    targets: dict = None,
):
    # If no targets are given, calculate `number_of_targets` linearly spaced targets
    if not targets:
        targets = linear_targets(results, indicator, number_of_targets)

    # Get (approximate) runtime to reach each target of indicator
    indicator_results = ResultSet()
    for r in results:
        indicator_results.append(r.at_indicator(indicator, targets[r.problem]))

    res = {}
    for algo, algo_results in indicator_results.by_algorithm():
        runtimes = []
        for algo_result in algo_results:
            runtimes.append(algo_result._data[["__fevals_dim", "__target_hit"]])
        runtimes = pl.concat(runtimes)
        ecdf = stats.ecdf(
            stats.CensoredData(
                runtimes["__fevals_dim"].filter(runtimes["__target_hit"] > 0).to_numpy(),
                right=runtimes["__fevals_dim"].filter(runtimes["__target_hit"] == 0).to_numpy(),
            )
        ).cdf
        # FIXME: No CIs for now...
        # res[algo] = (ecdf.quantiles, ecdf.probabilities, ecdf.confidence_interval())
        res[algo] = (ecdf.quantiles, ecdf.probabilities)
    return res


def ertplot(
    results: ResultSet,
    indicator: str,
    number_of_targets: int = 101,
    targets=None,
    ax=None,
):
    erts = ertvalues(results, indicator, number_of_targets=number_of_targets, targets=targets)
    if ax is None:
        fig, ax = plt.subplots()

    for algo, (fevals, prob) in erts.items():
        (line,) = ax.step(fevals, 100 * prob, label=algo)

    ax.set_xscale("log")
    ax.grid(True, which="both")
    ax.set_ylim(0, 100)
    ax.set_xlabel("$\log_{10}$(fevals / dimension)")
    ax.set_ylabel("Fraction of targets reached [%]")
    ax.legend()
    return ax
