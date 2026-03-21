
import polars as pl
from typing import List

from .result import Result, ResultSet, ProblemDescription


def read_coco_dataset(name: str | List[str]) -> ResultSet:
    try:
        import cocopp as pp
    except ImportError as e:
        raise ImportError("read_coco_dataset requires the cocopp package to be installed. Install it with: `pip install coco-postprocessing`.") from e

    resultset = ResultSet()

    # FIXME: Need to reset settings, otherwise you cannot load datasets from
    #   different suites.
    data = pp.load2(name, flat_list=True)
    print(type(data))
    for result in data:
        evals = result.evals
        for i in range(1, evals.shape[1]):
            problem = ProblemDescription(
                    f"bbob-f_{result.funcId}",
                    result.instancenumbers[i - 1],
                    result.dim,
                    1
            )

            df =  pl.DataFrame({
                "value": evals[:, 0],
                "fevals": evals[:, i]
            }).drop_nans()

            r = Result(result.algId, problem,df)
            resultset.append(r)
    return resultset
