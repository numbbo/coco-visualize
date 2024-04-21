"""Expected Runtime Plot"""
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from .types import ResultSet


def ertplot(results: ResultSet,
            indicator: str, 
            number_of_targets:int = 101,
            targets = None,
            ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    # If no targets are given, calculate `number_of_targets` linearly spaced targets
    if not targets:
        low = np.min([r[indicator].min() for r in results])
        high = np.max([r[indicator].max() for r in results])
        targets = np.linspace(low, high, number_of_targets)

    indicator_results = ResultSet()
        
    for r in results:
        indicator_results.append(r.at_indicator(indicator, targets))

    for algo, algo_results in indicator_results.by_algorithm():        
        parts = []
        for r in algo_results:
            df = r._data.with_columns((pl.col(r._name_fevals) / r.number_of_variables).alias("_fevals_div_dim"))
            parts.append(df)
        combined = pl.concat(parts)
        ax.ecdf(combined["_fevals_div_dim"], label=algo, compress=True)

    ax.set_xscale("log")
    ax.grid(True, which="both")
    ax.set_ylim(0, 1)    
    ax.set_xlabel("log10 fevals / dimension")
    ax.set_ylabel("Fraction of targets reached")
    ax.legend()
    return ax