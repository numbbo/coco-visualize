"""Data structures collect experimental results"""
from __future__ import annotations 

from dataclasses import dataclass

from typing import Any, Generator
from ._typing import FilePath

import numpy as np
import polars as pl

class Result:
    def __init__(self,
                 algorithm: str, 
                 problem_class: str,
                 problem_instance: str,
                 number_of_variables: int,
                 number_of_objectives: int,
                 data
                 ):
        if not isinstance(data, pl.DataFrame):
            data = pl.DataFrame(data) 
        self.algorithm = algorithm
        self.problem_class = problem_class
        self.problem_instance = problem_instance
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self._name_fevals = data.columns[0]
        self.indicators = set(data.columns[1:])
        self._data = data.sort(self._name_fevals)

    def clone(self) -> Result:
        return self.__class__(self.algorithm, 
                              self.problem_class, 
                              self.problem_instance,
                              self.number_of_variables,
                              self.number_of_objectives,
                              self._data.clone())
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, item):
        self._data[key] = item
    
    def __str__(self) -> str:
        return f"Results for {self.algorithm} on instance {self.problem_instance} of {self.problem_class} in {self.number_of_variables} dimensions with {self.number_of_objectives} objectives"

    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return self._data.height
    
    def at_indicator(self, indicator: str, targets):                
        target_fvals = np.inf * np.ones(len(targets))

        fvals = self[self._name_fevals]
        ivals = self[indicator].cum_max()

        indicator_idx = 0
        try:
            for target_idx, target in enumerate(targets):
                while ivals[indicator_idx] < target:
                    indicator_idx += 1                
                target_fvals[target_idx] = fvals[indicator_idx]
        except IndexError:
            pass # Break if indicator_idx goes out of bounds
        res = self.clone()
        res._data = pl.DataFrame([pl.Series(self._name_fevals, target_fvals), 
                            pl.Series(indicator, targets)])
        return res

    def to_parquet(self, path: FilePath):
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = self._data.to_arrow()
        metadata = {
          "algorithm": self.algorithm,
          "problem_class": self.problem_class,
          "problem_instance":  self.problem_instance,
          "number_of_variables": str(self.number_of_variables),
          "number_of_objectives": str(self.number_of_objectives),
        }
        schema = tbl.schema.with_metadata(metadata)
        tbl = pa.Table.from_arrays(list(tbl.itercolumns()), schema=schema)
        pq.write_table(tbl, str(path))

    @classmethod
    def from_parquet(cls, path: FilePath):
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        data = pl.from_arrow(tbl)
        print(tbl.schema.metadata)
        return cls(tbl.schema.metadata[b"algorithm"].decode("utf8"),
                   tbl.schema.metadata[b"problem_class"].decode("utf8"),
                   tbl.schema.metadata[b"problem_instance"].decode("utf8"),
                   int(tbl.schema.metadata[b"number_of_variables"]),
                   int(tbl.schema.metadata[b"number_of_objectives"]),
                   data)

@dataclass(eq=True, frozen=True)
class ProblemDescription:
    """Class to bundle the description of a problem instance."""
    problem_class: str
    problem_instance: str
    number_of_variables: int = 0
    number_of_objectives: int = 0

    def __str__(self) -> str:
        return f"Instance {self.problem_instance} of problem {self.problem_class} with {self.number_of_variables} variables and {self.number_of_objectives} objectives"
    
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
        self.problem_classes.add(result.problem_class)
        self.problem_instances.add(result.problem_instance)
        self.number_of_variables.add(result.number_of_variables)
        self.number_of_objectives.add(result.number_of_objectives)
        self.problems.add(ProblemDescription(result.problem_class, 
                                             result.problem_instance,
                                             result.number_of_variables,
                                             result.number_of_objectives))
        self._results.append(result)
        return self
    
    def by_algorithm(self) -> Generator[tuple[str, ResultSet], Any, None]:
        for g in sorted(self.algorithms):
            yield g, ResultSet(filter(lambda x: x.algorithm == g, self._results))            

    def by_problem(self) -> Generator[tuple[ProblemDescription, ResultSet], Any, None]:
        for pc in self.problem_classes:
            for pi in self.problem_instances:
                for n in self.number_of_variables:
                    for d in self.number_of_objectives:
                        ss = ResultSet(filter(lambda x: x.problem_class == pc and 
                                                x.problem_instance == pi and
                                                x.number_of_variables == n and
                                                x.number_of_objectives == d,
                                              self._results))
                        if len(ss) > 0:
                            yield ProblemDescription(pc, pi, n, d), ss            

    def by_problem_class(self) -> Generator[tuple[str, ResultSet], Any, None]:
        for g in self.problem_classes:
            yield g, ResultSet(filter(lambda x: x.algorithm == g, self._results))            

    def by_problem_instance(self) -> Generator[tuple[str, ResultSet], Any, None]:
        for g in self.problem_instances:
            yield g, ResultSet(filter(lambda x: x.problem_instance == g, self._results))            

    def by_number_of_variables(self) -> Generator[tuple[int, ResultSet], Any, None]:
        for g in self.number_of_variables:
            yield g, ResultSet(filter(lambda x: x.number_of_variables == g, self._results))            

    def by_number_of_objectives(self) -> Generator[tuple[int, ResultSet], Any, None]:
        for g in self.number_of_objectives:
            yield g, ResultSet(filter(lambda x: x.number_of_objectives == g, self._results))            
