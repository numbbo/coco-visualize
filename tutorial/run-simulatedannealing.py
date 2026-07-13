from cocoviz import ProblemDescription, Result
from pathlib import Path
import numpy as np
from scipy.optimize import dual_annealing
from problem import PROBLEMS

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

for problem in PROBLEMS:
    for local_search in [True, False]:
        algorithm_name = f"sa{'_ls' if local_search else ''}"
        for r in range(10):  # replications
            print(f"[{algorithm_name}] {problem.name} d={problem.number_of_variables} i={problem.instance} rep={r:02d}")
            problem.reset()
            x0 = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.number_of_variables)

            dual_annealing(
                problem,
                bounds=list(zip(problem.lower_bounds, problem.upper_bounds)),
                x0=x0,
                maxfun=problem.number_of_variables * 1000,
                no_local_search=not local_search,
            )

            pd = ProblemDescription(
                problem.name,
                str(problem.instance),
                problem.number_of_variables,
                problem.number_of_objectives,
            )


            result = Result(algorithm_name, pd, {"fevals": problem._fevals, "y": problem._y})
            filename = DATA_DIR / f"a={algorithm_name}_p={problem.name}_d={problem.number_of_variables}_i={problem.instance}_rep={r:02d}.parquet"
            result.to_parquet(filename)
            print(f"[{algorithm_name}] done, best y={problem._ymin:.6g}, nevals={problem._nevals}")
