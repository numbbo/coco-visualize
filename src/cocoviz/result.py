"""Data structures for collecting experimental results"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Generator, Iterable, Iterator, Sequence, Set
from typing import Any, Callable, final

import numpy as np
import numpy.typing as npt
import polars as pl
from polars.exceptions import ColumnNotFoundError, SchemaFieldNotFoundError

from . import indicator as ind
from ._typing import FilePath, FrameLike, NumericVector
from .exceptions import IndicatorMismatchException, NoSuchIndicatorException

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=True, order=True, frozen=True)
class ProblemDescription:
    """Description of a specific (benchmark) problem

    Attributes
    ----------
    name : str
    instance : str
    number_of_variables : int
    number_of_objectives : int

    Methods
    -------
    to_json()
        Return a JSON document describing the problem
    from_json(json)
        Create a ProblemDescription from a JSON document
    """

    name: str
    instance: str
    number_of_variables: int = 0
    number_of_objectives: int = 0

    def __str__(self) -> str:
        return f"Instance {self.instance} of problem {self.name} with {self.number_of_variables} variables and {self.number_of_objectives} objectives"

    def to_json(self) -> str:
        """Convert the problem description into JSON document.

        Returns
        -------
        str
            JSON document describing the problem.
        """
        raw = dataclasses.asdict(self)
        return json.dumps(raw)

    @classmethod
    def from_json(cls, str) -> ProblemDescription:
        """Create a ProblemDescription from a JSON document.

        Parameters
        ----------
        json : str
            JSON document describing the problem.

        Returns
        -------
        ProblemDescription
        """
        raw = json.loads(str)
        return cls(**raw)


@final
class Result:
    """Results of a single algorithms run on a single problem"""

    def __init__(
        self,
        algorithm: str,
        problem: ProblemDescription,
        data: FrameLike,
        fevals_column: str = "fevals",
    ):
        df = pl.DataFrame(data) if not isinstance(data, pl.DataFrame) else data.clone()

        self.algorithm = algorithm
        self.problem = problem

        if fevals_column != "__fevals":
            # Rename `fevals_column` to '__fevals', if not present guess and warn.
            try:
                df = df.rename({fevals_column: "__fevals"})
            except (ColumnNotFoundError, SchemaFieldNotFoundError):
                logger.warning(
                    f"Assuming first column ('{df.columns[0]}') contains the number of function evaluations."
                )
                df = df.rename({df.columns[0]: "__fevals"})

        # Sort data by time ('__fevals' column)
        df = df.sort("__fevals")

        # Pre-compute fevals / dim as '__fevals_dim'
        self._data = df.with_columns((pl.col("__fevals") / problem.number_of_variables).alias("__fevals_dim"))

        # All columns excluding '__fevals' and '__fevals_dim' are assumed to
        # contain indicator values.
        self.indicators: Set[str] = set(self._data.columns) - {"__fevals", "__fevals_dim"}

    def __getitem__(self, key: str) -> Any:
        """
        Return the indicator column named `key`.

        Parameters
        ----------
        key : str
            Name of indicator.

        Returns
        -------
        Any
            The polars `Series` of indicator values.

        Raises
        ------
        NoSuchIndicatorException
            If `key` is not known as an indicator for this result.
        """
        if key not in self.indicators:
            raise NoSuchIndicatorException(key)
        return self._data[key]

    def __setitem__(self, key: str, item):
        """
        Add / replace an indicator column, also registering `key` as an indicator.

        Parameters
        ----------
        key : str
            Name of the indicator.
        item : Any
            Column-like (broadcastable) containing the value of the indicator at the
            observed times.
        """
        self._data[key] = item
        self.indicators.add(key)

    def __str__(self) -> str:
        return f"Results for {self.algorithm} on instance {self.problem.instance} of {self.problem.name} in {self.problem.number_of_variables} dimensions with {self.problem.number_of_objectives} objectives"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        """
        Number of rows / observations in this result.

        Returns
        -------
        int
            number of observations
        """
        return self._data.height

    def at_indicator(self, indicator: ind.Indicator | str, targets: NumericVector):
        """
        Compute for each target value the *first* time (function evaluation)
        at which this run reaches (or surpasses) that target in the given indicator.

        Parameters
        ----------
        indicator : Indicator | str
            Indicator object or indicator name.
        targets : array-like
            Target values for the indicator.

        Returns
        -------
        Result
            A new `Result` instance resampled to `targets`.
        """
        indicator = ind.resolve(indicator)

        # FIXME: Assumes maximization of indicator...
        if indicator.name not in self.indicators:
            raise NoSuchIndicatorException(indicator)

        fvals = self._data["__fevals"]

        # Make sure indicator values are monotonic
        if indicator.larger_is_better:
            ivals = self._data[indicator.name].cum_max()
        else:
            ivals = self._data[indicator.name].cum_min()

        target_fvals: npt.NDArray[np.float64] = fvals.max() * np.ones(len(targets))
        target_hit = np.zeros(len(targets))

        indicator_idx = 0
        try:
            for i, target in enumerate(targets):
                ## FIXME: This is ugly duplication, but I don't know of
                ## a more elegant way to write it. :/
                if indicator.larger_is_better:
                    while ivals[indicator_idx] < target:
                        indicator_idx += 1
                else:
                    while ivals[indicator_idx] > target:
                        indicator_idx += 1
                target_fvals[i] = fvals[indicator_idx]
                target_hit[i] = True
        except IndexError:
            pass  # Break if indicator_idx goes out of bounds

        return self.__class__(
            self.algorithm,
            self.problem,
            pl.DataFrame(
                [
                    pl.Series("fevals", target_fvals),
                    pl.Series(indicator.name, targets),
                    pl.Series("__target_hit", target_hit),
                ]
            ),
        )

    def to_parquet(self, path: FilePath):
        """
        Serialise the result to parquet.

        Writes both the polars DataFrame *and* metadata
        (algorithm name + problem JSON) into parquet schema metadata.

        Parameters
        ----------
        path : path-like
            Output path. Existing file will be overwritten.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        tbl = self._data.to_arrow()
        metadata = {
            "algorithm": self.algorithm,
            "problem": self.problem.to_json(),
        }
        schema = tbl.schema.with_metadata(metadata)
        tbl = pa.Table.from_arrays(list(tbl.itercolumns()), schema=schema)
        pq.write_table(tbl, str(path))

    @classmethod
    def from_parquet(cls, path: FilePath):
        """
        Load a `Result` from parquet.

        Expects parquet schema metadata entries:
        - algorithm
        - problem (in ProblemDescription.to_json() form)

        Parameters
        ----------
        path : path-like
            Parquet file to load.

        Returns
        -------
        Result
        """
        import pyarrow.parquet as pq

        tbl = pq.read_table(path)
        algorithm = tbl.schema.metadata[b"algorithm"].decode("utf8")
        problem = ProblemDescription.from_json(tbl.schema.metadata[b"problem"].decode("utf8"))
        data = pl.from_arrow(tbl)

        return cls(algorithm, problem, data, "__fevals")


class ResultSet:
    "Collection of `Result` objects"

    def __init__(self, results: Iterable[Result] = ()):
        self.algorithms: Set[str] = set()
        self.problems: Set[ProblemDescription] = set()
        self.number_of_variables: Set[int] = set()
        self.number_of_objectives: Set[int] = set()
        self._results: Sequence[Result] = []

        _ = self.extend(results)

    def __getitem__(self, key: int) -> Result:
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[Result]:
        return iter(self._results)

    def append(self, result: Result) -> ResultSet:
        # Make sure results have matching indicators and raise an exception if
        # they don't.
        if len(self._results) > 0 and self._results[0].indicators != result.indicators:
            raise IndicatorMismatchException(
                "Indicators in results don't match: {self._results[0].indicators} vs {result.indicators}"
            )
        self.algorithms.add(result.algorithm)
        self.problems.add(result.problem)
        self.number_of_variables.add(result.problem.number_of_variables)
        self.number_of_objectives.add(result.problem.number_of_objectives)
        self._results.append(result)
        return self

    def extend(self, results: Iterable[Result]) -> ResultSet:
        """Extend a ResultSet with a sequence of `Result`s or another `ResultSetǹ.

        Parameters
        ----------
        results: Sequence[Result] or Resultset:
            Results to add to this result set.
        """
        for result in results:
            _ = self.append(result)
        return self

    def filter(self, function: Callable[[Result], bool]) -> ResultSet:
        """Return a ResultSet containing the results for which `function` returns `True`

        Parameters
        ----------
        function : Function
            Function to select results to return

        Returns
        -------
        ResultSet
            Results matched by `function`.
        """
        subset = ResultSet()
        for result in self._results:
            if function(result):
                _ = subset.append(result)
        return subset

    def by_algorithm(self) -> Generator[tuple[str, ResultSet], None]:
        for algorithm in sorted(self.algorithms):
            subset = ResultSet()
            for result in self._results:
                if result.algorithm == algorithm:
                    _ = subset.append(result)
            if len(subset) > 0:
                yield algorithm, subset

    def by_problem(self) -> Generator[tuple[ProblemDescription, ResultSet], None]:
        for problem in sorted(self.problems):
            subset = ResultSet()
            for result in self._results:
                if result.problem == problem:
                    _ = subset.append(result)
            if len(subset) > 0:
                yield problem, subset

    def _by_int_problem_property(self, property: str) -> Generator[tuple[int, ResultSet], None]:
        values: Set[int] = set()
        for problem in self.problems:
            values.add(getattr(problem, property))

        for value in sorted(values):
            subset = ResultSet()
            for result in self._results:
                if getattr(result.problem, property) == value:
                    _ = subset.append(result)
            if len(subset) > 0:
                yield value, subset

    def _by_str_problem_property(self, property: str) -> Generator[tuple[str, ResultSet], None]:
        values: Set[str] = set()
        for problem in self.problems:
            values.add(getattr(problem, property))

        for value in sorted(values):
            subset = ResultSet()
            for result in self._results:
                if getattr(result.problem, property) == value:
                    _ = subset.append(result)
            if len(subset) > 0:
                yield value, subset

    def by_problem_name(self) -> Generator[tuple[str, ResultSet], None]:
        return self._by_str_problem_property("name")

    def by_problem_instance(self) -> Generator[tuple[str, ResultSet], None]:
        return self._by_str_problem_property("instance")

    def by_number_of_variables(self) -> Generator[tuple[int, ResultSet], None]:
        return self._by_int_problem_property("number_of_variables")

    def by_number_of_objectives(self) -> Generator[tuple[int, ResultSet], None]:
        return self._by_int_problem_property("number_of_objectives")
