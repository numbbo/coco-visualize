import json
from pathlib import Path

import numpy as np
from cmaes import CMA
from problem import PROBLEMS

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

for problem in PROBLEMS:
    BUDGET = problem.number_of_variables * 1000
    for r in range(10):  # replications
        print(f"[cmaes] {problem.name} d={problem.number_of_variables} i={problem.instance} rep={r:02d}")
        problem.reset()
        lower_bounds = np.array(problem.lower_bounds)
        upper_bounds = np.array(problem.upper_bounds)
        bounds = np.stack([lower_bounds, upper_bounds], axis=1)

        x0 = np.random.uniform(lower_bounds, upper_bounds, problem.number_of_variables)
        sigma0 = 0.25 * float(np.mean(upper_bounds - lower_bounds))

        optimizer = CMA(mean=x0, sigma=sigma0, bounds=bounds)

        while problem._nevals < BUDGET and not optimizer.should_stop():
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = float(problem(x))
                solutions.append((x, value))
            optimizer.tell(solutions)

        log = [{"fevals": fevals, "y": float(y)} for fevals, y in problem.log()]
        filename = DATA_DIR / f"a=cmaes_p={problem.name}_d={problem.number_of_variables}_i={problem.instance}_rep={r:02d}.json"
        with open(filename, "w") as f:
            json.dump(log, f)
        print(f"[cmaes] done, best y={problem._ymin:.6g}, nevals={problem._nevals}")
