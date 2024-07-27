"""Runtime profiles"""

import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import matplotlib.transforms as mtransforms
import polars as pl
import scipy.stats as stats

from typing import Union

from . import indicator as ind
from .targets import linear_targets
from .result import ResultSet
from .exceptions import BadRuntimeProfileException


def runtime_profiles(
    results: ResultSet,
    indicator: Union[ind.Indicator, str],
    maximize_indicator: bool = True,
    number_of_targets: int = 101,
    targets: Union[dict, None] = None,
):
    """Compute a runtime profile for each algorithm in `results`.

    Parameters
    ----------
    results : ResultSet
        Collection of results of running any number of algorithms on any number of problems or problem instances.
    indicator : Indicator or str
        Performance indicator to analyse.
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
    indicator = ind.resolve(indicator)
    
    if len(results.number_of_variables) > 1:
        raise BadRuntimeProfileException("Cannot derive runtime profile for problems with different number of variables.")

    if len(results.number_of_objectives) > 1:
        raise BadRuntimeProfileException("Cannot derive runtime profile for problems with different number of objectives.")

    # If no targets are given, calculate `number_of_targets` linearly spaced targets
    if not targets:
        targets = linear_targets(results, indicator, number_of_targets)

    # Get (approximate) runtime to reach each target of indicator
    indicator_results = ResultSet()
    for r in results:
        indicator_results.append(r.at_indicator(indicator, targets[r.problem]))

    res = {}
    n_results = None
    for algo, algo_results in indicator_results.by_algorithm():        
        # Make sure each algorithm has the same number of repetitions/runs If
        # not, raise an exception for now. In the future we could try to down-
        # or upsample the offending algorithm.
        if n_results is None:
            n_results = len(algo_results)
        elif n_results != len(algo_results):
            raise BadRuntimeProfileException(f"Expected {n_results} results for algorithm {algo}, found {len(results)}.")

        rtlist = []

        for algo_result in algo_results:
            rtlist.append(algo_result._data[["__fevals_dim", "__target_hit"]])
        
        runtimes: pl.DataFrame = pl.concat(rtlist)
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
    indicator : Indicator or str
        Performance indicator to analyse. Name must match the name used in the results.
        If a string is passed in, the indicator must have been registered previously by a call to
        `cocoviz.indicator.register`.
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
    else:
        fig: mfigure.Figure = ax.figure

    for algo, (fevals, prob) in profiles.items():
        (line,) = ax.step(fevals, 100 * prob, label=algo)

    ax.set_xscale("log")
    ax.grid(True, which="both", color="lightgrey")
    ax.legend(title="Algorithms")
    
    problems = sorted(set(p.name for p in results.problems))
    if len(problems) > 1:
        offset = mtransforms.ScaledTranslation(10/72., -10/72., fig.dpi_scale_trans)
        ax.text(0, 1, "Problems\n" + "\n".join(problems), 
                transform=ax.transAxes + offset,
                bbox=dict(facecolor="white", edgecolor="lightgrey"),
                verticalalignment="top", 
                horizontalalignment="left")
    else:
        pass
    
    ax.set_ylim(0, 100)
    ax.set_xlabel("# fevals / dimension")
    ax.set_ylabel("Fraction of targets reached [%]")
    
    return ax
