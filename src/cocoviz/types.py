"""Data structures collect experimental results"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Generator, Union

import numpy as np
import polars as pl

from ._typing import FilePath
from .exceptions import NoSuchIndicatorException

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
    def from_json(cls, json) -> ProblemDescription:
        """Create a ProblemDescription from a JSON document.

        Parameters
        ----------
        json : str
            JSON document describing the problem.

        Returns
        -------
        ProblemDescription            
        """
        raw = json.loads(json)
        return cls(**raw)


class Result:
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

        # Rename `fevals_column` to '__fevals', if not present guess and warn.
        try:
            data = data.rename({fevals_column: "__fevals"})
        except pl.SchemaFieldNotFoundError:
            logger.warning(f"Assuming first column ('{data.columns[0]}') contains the number of function evaluations.")
            data = data.rename({data.columns[0]: "__fevals"})

        # Pre-compute fevals / dim as '__fevals_dim' and sort data by '__fevals'
        self._data = data.with_columns((pl.col("__fevals") / problem.number_of_variables).alias("__fevals_dim")).sort(
            "__fevals"
        )

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

    def at_indicator(self, indicator: str, targets):
        if indicator not in self.indicators:
            raise NoSuchIndicatorException(indicator)

        fvals = self._data["__fevals"]
        # Make sure indicator values are monotonically increasing
        ivals = self._data[indicator].cum_max()

        target_fvals = fvals.max() * np.ones(len(targets))
        target_hit = np.zeros(len(targets))

        indicator_idx = 0
        try:
            for i, target in enumerate(targets):
                while ivals[indicator_idx] < target:
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
                    pl.Series(indicator, targets),
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

        return cls(algorithm, problem, data)


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

    def __getitem__(self, key: int) -> Any:
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)

    def append(self, result: Result) -> ResultSet:
        self.algorithms.add(result.algorithm)
        self.problems.add(result.problem)
        self._results.append(result)
        return self

    def by_algorithm(self) -> Generator[tuple[str, ResultSet], Any, None]:
        ## FIXME: Caution, this potentially is quadratic :/
        for algorithm in sorted(self.algorithms):
            ss = ResultSet()
            for result in self._results:
                if result.algorithm == algorithm:
                    ss.append(result)
            if len(ss) > 0:
                yield algorithm, ss

    def by_problem(self) -> Generator[tuple[ProblemDescription, ResultSet], Any, None]:
        ## FIXME: Caution, this potentially is quadratic :/
        for problem in sorted(self.problems):
            ss = ResultSet()
            for result in self._results:
                if result.problem == problem:
                    ss.append(result)
            if len(ss) > 0:
                yield problem, ss

    def _by_problem_property(self, property: str) -> Generator[tuple[ProblemDescription, ResultSet], Any, None]:
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

    def by_problem_name(
        self,
    ) -> Generator[tuple[Union[int, str], ResultSet], Any, None]:
        return self._by_problem_property("name")

    def by_problem_instance(
        self,
    ) -> Generator[tuple[Union[int, str], ResultSet], Any, None]:
        return self._by_problem_property("instance")

    def by_number_of_variables(self) -> Generator[tuple[int, ResultSet], Any, None]:
        return self._by_problem_property("number_of_variables")

    def by_number_of_objectives(self) -> Generator[tuple[int, ResultSet], Any, None]:
        return self._by_problem_property("number_of_objectives")
