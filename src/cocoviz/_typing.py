from os import PathLike
from typing import TypeAlias, Union, Any
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

FilePath: TypeAlias = str | PathLike[str]

Numeric: TypeAlias = int | float | np.number
NumericVector: TypeAlias = npt.ArrayLike | Sequence[Numeric]

FrameLike: TypeAlias = Union[
    pl.DataFrame,
    Mapping[str, Union[Sequence[object], Mapping[str, Sequence[object]], pl.Series]],
    Sequence[Any],
    npt.ArrayLike,
    "pyarrow.Table",
    "pandas.DataFrame",
    "torch.Tensor",
]
