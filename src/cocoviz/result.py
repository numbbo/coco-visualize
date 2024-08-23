"""Data structures for collecting experimental results"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Generator, Union, Callable, Iterator

import numpy as np
import numpy.typing as npt

import polars as pl
from polars.exceptions import ColumnNotFoundError, SchemaFieldNotFoundError

from . import indicator as ind
from ._typing import FilePath
from .exceptions import NoSuchIndicatorException, IndicatorMismatchException

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


class Result:
    """Results of a single algorithms run on a single problem"""
    
    def __init__(
        self,
        algorithm: str,
        problem: ProblemDescription,
        data,
        fevals_column: str = "fevals",
    ):
        if not isinstance(data, pl.DataFrame):
            data = pl.DataFrame(data)
        self.algorithm = algorithm
        self.problem = problem

        if fevals_column != "__fevals":
            # Rename `fevals_column` to '__fevals', if not present guess and warn.
            try:
                data = data.rename({fevals_column: "__fevals"})
            except (ColumnNotFoundError, SchemaFieldNotFoundError):
                logger.warning(f"Assuming first column ('{data.columns[0]}') contains the number of function evaluations.")
                data = data.rename({data.columns[0]: "__fevals"})

        # Sort data by '__fevals' column
        data = data.sort("__fevals")

        # Pre-compute fevals / dim as '__fevals_dim'
        self._data = data.with_columns((pl.col("__fevals") / problem.number_of_variables).alias("__fevals_dim"))

        # All columns excluding '__fevals' and '__fevals_dim' are assumed to
        # contain indicator values.
        self.indicators = set(self._data.columns) - {"__fevals", "__fevals_dim"}

    def __getitem__(self, key: str) -> Any:
        if key not in self.indicators:
            raise NoSuchIndicatorException(key)
        return self._data[key]

    def __setitem__(self, key: str, item):
        self._data[key] = item
        self.indicators.add(key)

    def __str__(self) -> str:
        return f"Results for {self.algorithm} on instance {self.problem.instance} of {self.problem.name} in {self.problem.number_of_variables} dimensions with {self.problem.number_of_objectives} objectives"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self._data.height

    def at_indicator(self, indicator: Union[ind.Indicator, str], targets):
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

        target_fvals : npt.NDArray[np.float64] = fvals.max() * np.ones(len(targets))
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
        """Write results to a parquet file"""
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
        """Read results from a parquet file."""
        import pyarrow.parquet as pq

        tbl = pq.read_table(path)
        algorithm = tbl.schema.metadata[b"algorithm"].decode("utf8")
        problem = ProblemDescription.from_json(tbl.schema.metadata[b"problem"].decode("utf8"))
        data = pl.from_arrow(tbl)

        return cls(algorithm, problem, data, "__fevals")


class ResultSet:
    def __init__(self, results=[]):
        self.algorithms = set()
        self.problems = set()
        self.problem_classes = set()
        self.problem_instances = set()
        self.number_of_variables = set()
        self.number_of_objectives = set()
        self._results = []

        for r in results:
            self.append(r)

    def __getitem__(self, key: int) -> Result:
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) ->  Iterator[Result]:
        return iter(self._results)

    def append(self, result: Result) -> ResultSet:
        # Make sure results have matching indicators and raise an exception if
        # they don't.        
        if len(self._results) > 0 and self._results[0].indicators != result.indicators:
            raise IndicatorMismatchException("Indicators in results don't match: {self._results[0].indicators} vs {result.indicators}")
        self.algorithms.add(result.algorithm)
        self.problems.add(result.problem)
        self.number_of_variables.add(result.problem.number_of_variables)
        self.number_of_objectives.add(result.problem.number_of_objectives)
        self._results.append(result)
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

        res = ResultSet()
        for result in self._results:
            if function(result):
                res.append(result)
        return res
                
    def by_algorithm(self) -> Generator[tuple[str, ResultSet], Any, None]:
        ## FIXME: Caution, this is potentially quadratic !
        for algorithm in sorted(self.algorithms):
            ss = ResultSet()
            for result in self._results:
                if result.algorithm == algorithm:
                    ss.append(result)
            if len(ss) > 0:
                yield algorithm, ss

    def by_problem(self) -> Generator[tuple[ProblemDescription, ResultSet], Any, None]:
        ## FIXME: Caution, this is potentially quadratic!
        for problem in sorted(self.problems):
            ss = ResultSet()
            for result in self._results:
                if result.problem == problem:
                    ss.append(result)
            if len(ss) > 0:
                yield problem, ss

    def _by_int_problem_property(self, property: str) -> Generator[tuple[int, ResultSet], Any, None]:
        values = set()
        for problem in self.problems:
            values.add(getattr(problem, property))

        for value in sorted(values):
            ss = ResultSet()
            for result in self._results:
                if getattr(result.problem, property) == value:
                    ss.append(result)
            if len(ss) > 0:
                yield value, ss

    def _by_problem_property(self, property: str) -> Generator[tuple[Union[int, str], ResultSet], Any, None]:
        values = set()
        for problem in self.problems:
            values.add(getattr(problem, property))

        for value in sorted(values):
            ss = ResultSet()
            for result in self._results:
                if getattr(result.problem, property) == value:
                    ss.append(result)
            if len(ss) > 0:
                yield value, ss


    def by_problem_name(self) -> Generator[tuple[Union[int, str], ResultSet], Any, None]:
        return self._by_problem_property("name")

    def by_problem_instance(self) -> Generator[tuple[Union[int, str], ResultSet], Any, None]:
        return self._by_problem_property("instance")

    def by_number_of_variables(self) -> Generator[tuple[int, ResultSet], Any, None]:
        return self._by_int_problem_property("number_of_variables")        

    def by_number_of_objectives(self) -> Generator[tuple[int, ResultSet], Any, None]:
        return self._by_int_problem_property("number_of_objectives")        
