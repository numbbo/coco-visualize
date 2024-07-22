"""Expected Runtime calculations"""

import matplotlib.pyplot as plt
import polars as pl
import scipy.stats as stats

from .targets import linear_targets
from .types import ResultSet
from .exceptions import BadResultSetException


def runtime_profiles(
    results: ResultSet,
    indicator: str,
    maximize_indicator: bool = True,
    number_of_targets: int = 101,
    targets: dict = None,
):
    """Compute a runtime profile for each algorithm in `results`.

    Parameters
    ----------
    results : ResultSet
        Collection of results of running any number of algorithms on any number of problems or problem instances.
    indicator : str
        Name of indicator to analyse.
    maximize_indicator : bool, optional
        Should the indicator be maximized or minimized?
    number_of_targets : int, optional
        Number of target values to generate for each problem it `targets` is missing.
    targets : dict, optional
        Dictionary of target values, indexed by problem.
        If missing, `number_of_targets` targets are automatically generated.

    Returns
    -------
    dict
        Quantiles and probabilities for each algorithm in `results`.
    """

    if len(results.number_of_variables) > 1:
        raise BadResultSetException("Cannot derive runtime profile for problems with different number of variables.")

    if len(results.number_of_objectives) > 1:
        raise BadResultSetException("Cannot derive runtime profile for problems with different number of objectives.")

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


def rtpplot(
    results: ResultSet,
    indicator: str,
    number_of_targets: int = 101,
    targets=None,
    ax=None,
):
    """Plot runtime profiles of the `results`.

    Parameters
    ----------
    results : ResultSet
        Collection of results of running any number of algorithms on any number of problems or problem instances.
    indicator : str
        Name of indicator to analyse.
    number_of_targets : int, optional
        Number of target values to generate for each problem it `targets` is missing.
    targets : dict, optional
        Dictionary of target values, index by problem.
        If missing, `number_of_targets` targets are automatically generated.

    ax : matplotlib.axes.Axes, optional
        Axes where the plot is drawn.
        If missing, a new figure is created and returned.

    Returns
    -------
    matplotlib.axes.Axes
        An Axes object containing the plot.
        If `ax` is provided, it is returned.
        Otherwise a new figure is created and the corresponding Axes object is returned.
    """
    profiles = runtime_profiles(results, indicator, number_of_targets=number_of_targets, targets=targets)
    if ax is None:
        fig, ax = plt.subplots()

    for algo, (fevals, prob) in profiles.items():
        (line,) = ax.step(fevals, 100 * prob, label=algo)

    ax.set_xscale("log")
    ax.grid(True, which="both")
    ax.set_ylim(0, 100)
    ax.set_xlabel("$\\log_{10}$(# fevals / dimension)")
    ax.set_ylabel("Fraction of targets reached [%]")
    ax.legend()
    return ax
