import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from problem import PROBLEMS

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


for problem in PROBLEMS:
    for r in range(10):  # replications
        print(f"[powell] {problem.name} d={problem.number_of_variables} i={problem.instance} rep={r:02d}")
        problem.reset()
        x0 = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.number_of_variables)
        assert len(x0) == problem.number_of_variables

        result = minimize(
            problem,
            x0,
            method="Powell",
            bounds=zip(problem.lower_bounds, problem.upper_bounds),
            options={"maxiter": problem.number_of_variables * 1000},
        )
        log = problem.log()

        filename = DATA_DIR / f"a=powell_p={problem.name}_d={problem.number_of_variables}_i={problem.instance}_rep={r:02d}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fevals", "y"])
            writer.writerows(log)
        print(f"[powell] done, best y={problem._ymin:.6g}, nevals={problem._nevals}")
