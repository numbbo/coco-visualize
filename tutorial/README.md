<h1 align="center">
<img src="https://raw.githubusercontent.com/numbbo/coco-visualize/main/cocoviz.svg" width="400"><br>
</h1>

# Tutorial

[![Open Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/numbbo/coco-visualize/blob/main/tutorial/tutorial.ipynb)

## Recording

This is a prerecorded version of the hands on portion of our tutorial "More than Tables: Visualizing Anytime Performance in Single- and Multiobjective Optimization" at GECCO 2026.

[![Watch the video on YouTube](https://img.youtube.com/vi/xqZ9cZn5R8Y/0.jpg)](https://www.youtube.com/watch?v=xqZ9cZn5R8Y)

## About

Assessing the performance of optimization algorithms is important for their design, selection and recommendation, in both single- and multiobjective settings.
Despite recent progress in how performance is analyzed in single-objective optimization, studies in multiobjective optimization often still follow a practice established two decades ago: presenting long tables of quality indicator values at a (single) fixed evaluation budget. 
This format places the burden of interpretation on the reader and makes it difficult to understand how algorithms behave over time.

In contrast, single-objective studies nowadays often rely on aggregation and visualization of performance data, highlighting *anytime* algorithm behavior using runtime profile plots (also known as empirical runtime distribution functions or data profiles). 
These visualizations provide an immediate overview of algorithm performance over time and across problems.

This tutorial shows how similar ideas can be applied seamlessly across both single- and multiobjective scenarios, shifting from budget-based to target-based and therefore anytime performance assessment. 
We introduce [**cocoviz**](https://github.com/numbbo/coco-visualize), a lightweight python package that produces runtime profile plots directly from simple input data (pairs of quality indicator values and the corresponding number of evaluations), with only minimal required data preparation.

Participants are encouraged to bring their own algorithm results for an interactive hands-on session with cocoviz, where they can generate visualizations of their data and, if they wish, compare it with that of other participants.

## Repository contents

| Path | Description |
| --- | --- |
| `tutorial.ipynb` | The tutorial notebook — start here. |
| `problem.py` | Toy single-objective benchmark problems (`SphereProblem`, `ColorProblem`) used to generate demo data. |
| `run-cmaes.py` | Runs CMA-ES on all problems and writes results as JSON. |
| `run-powell.py` | Runs Powell's method on all problems and writes results as CSV. |
| `run-simulatedannealing.py` | Runs (dual) simulated annealing, with and without local search, and writes results natively as `cocoviz` Parquet files. |

The three `run-*.py` scripts intentionally each save their results in a different format (CSV, JSON, native Parquet) and with a slightly different layout, as if each had been written independently by a different student who never saw the others' code. 
This is on purpose: it reflects the reality of algorithm result data you'll encounter in the real world.

## Bring your own data

If you'd like to follow along with your own algorithm results during the hands-on session, all cocoviz needs is, for each run, a sequence of `(evaluations, indicator value)` pairs plus a bit of metadata (problem name, dimension, algorithm name).
See the "Reading results" section of the notebook for examples of importing from CSV, JSON and Parquet.
The same general approach should work for whatever format your own data happens to be in.

## Links

- [cocoviz on GitHub](https://github.com/numbbo/coco-visualize)
- [COCO / numbbo](https://github.com/numbbo/coco)
- [IOHProfiler](https://iohprofiler.github.io/)
