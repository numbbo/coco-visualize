import jax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)


def _ciede2000_pairwise(L, a, b):
    _EPS = 1e-10  # For numerical stability of the gradient
    L_arr = np.asarray(L, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = L_arr.shape[0]

    # Upper-triangle indices (i < j), excluding diagonal
    rows, cols = np.triu_indices(n, k=1)

    # Per-color intermediates (1-D)
    C_star = np.sqrt(a_arr**2 + b_arr**2 + _EPS)

    # Gather pairs
    C_star_i, C_star_j = C_star[rows], C_star[cols]
    a_i, a_j = a_arr[rows], a_arr[cols]
    b_i, b_j = b_arr[rows], b_arr[cols]
    L_i, L_j = L_arr[rows], L_arr[cols]

    C_bar = (C_star_i + C_star_j) / 2.0
    C_bar_7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))

    a_prime_i = a_i * (1.0 + G)
    a_prime_j = a_j * (1.0 + G)

    C_prime_i = np.sqrt(a_prime_i**2 + b_i**2 + _EPS)
    C_prime_j = np.sqrt(a_prime_j**2 + b_j**2 + _EPS)

    h_prime_i = np.degrees(np.arctan2(b_i, a_prime_i)) % 360.0
    h_prime_j = np.degrees(np.arctan2(b_j, a_prime_j)) % 360.0

    delta_L_prime = L_j - L_i
    delta_C_prime = C_prime_j - C_prime_i

    delta_h_prime = _compute_delta_h(h_prime_i, h_prime_j, C_prime_i, C_prime_j)
    delta_H_prime = (
        2.0
        * np.sqrt(C_prime_i * C_prime_j + _EPS)
        * np.sin(np.radians(delta_h_prime / 2.0))
    )

    L_bar_prime = (L_i + L_j) / 2.0
    C_bar_prime = (C_prime_i + C_prime_j) / 2.0
    h_bar_prime = _compute_mean_hue(h_prime_i, h_prime_j, C_prime_i, C_prime_j)

    L_bar_prime_minus_50_sq = (L_bar_prime - 50.0) ** 2
    S_L = 1.0 + (0.015 * L_bar_prime_minus_50_sq) / np.sqrt(
        20.0 + L_bar_prime_minus_50_sq
    )
    S_C = 1.0 + 0.045 * C_bar_prime

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )
    S_H = 1.0 + 0.015 * C_bar_prime * T

    delta_theta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    C_bar_prime_7 = C_bar_prime**7
    R_C = 2.0 * np.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25.0**7))
    R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

    term_L = delta_L_prime / S_L
    term_C = delta_C_prime / S_C
    term_H = delta_H_prime / S_H

    vals = np.sqrt(term_L**2 + term_C**2 + term_H**2 + R_T * term_C * term_H + _EPS)
    return vals


def _compute_delta_h(h1, h2, C1_prime, C2_prime):
    diff = h2 - h1
    delta_h = np.where(
        np.abs(diff) <= 180.0,
        diff,
        np.where(diff > 180.0, diff - 360.0, diff + 360.0),
    )
    delta_h = np.where((C1_prime == 0) | (C2_prime == 0), 0.0, delta_h)

    return delta_h


def _compute_mean_hue(h1, h2, C1_prime, C2_prime):
    both_achromatic = (C1_prime == 0) & (C2_prime == 0)
    one_achromatic = (C1_prime == 0) ^ (C2_prime == 0)

    diff = np.abs(h1 - h2)
    mean_std = (h1 + h2) / 2.0
    mean_wrap = (h1 + h2 + 360.0) / 2.0
    mean_wrap = np.where(mean_wrap >= 360.0, mean_wrap - 360.0, mean_wrap)
    h_bar = np.where(diff <= 180.0, mean_std, mean_wrap)
    h_bar = np.where(one_achromatic, h1 + h2, h_bar)
    h_bar = np.where(both_achromatic, 0.0, h_bar)

    return h_bar


@jax.jit
def _color_objective(x):
    L = np.clip(x[0::3], 50, 75)
    a = np.clip(x[1::3], -86, 98)
    b = np.clip(x[2::3], -108, 95)
    return -_ciede2000_pairwise(L, a, b).min()

@jax.jit
def _sphere_objective(center, x):
    return np.linalg.norm(x - center)


class Problem:
    def __init__(self):
        self.number_of_objectives = 1
        self.instance = 1
        self.reset()

    def reset(self):
        self._nevals = 0
        self._ymin = float("inf")
        self._fevals = []
        self._y = []

    def log(self):
        return zip(self._fevals, self._y)

    def _evaluate(self, x):
        raise NotImplementedError

    def __call__(self, x):
        y = self._evaluate(x)
        self._nevals += 1
        if y < self._ymin:
            self._ymin = y
            self._fevals.append(self._nevals)
            self._y.append(y)
        return y


class SphereProblem(Problem):
    def __init__(self, dimension: int, instance: int):
        super().__init__()
        self._center = jax.random.uniform(
            jax.random.PRNGKey(instance), (dimension,), minval=-4.0, maxval=4.0
        )
        self.lower_bounds = np.array([-5] * dimension)
        self.upper_bounds = np.array([5] * dimension)
        self.number_of_variables = dimension
        self.name = "sphere"
        self.instance = instance

    def _evaluate(self, x):
        return _sphere_objective(self._center, x)


class ColorProblem(Problem):
    def __init__(self, ncolors: int):
        super().__init__()
        self._ncolors = ncolors
        self.lower_bounds = np.array([50.0, -86.0, -108.0] * ncolors)
        self.upper_bounds = np.array([75.0, 98.0, 95.0] * ncolors)
        self.number_of_variables = ncolors * 3
        self.name = "equicolors"

    def _evaluate(self, x):
        return _color_objective(x)


PROBLEMS = \
    [ColorProblem(ncolors) for ncolors in [8, 16, 32]] + \
    [SphereProblem(dim, inst) for dim in [24, 48, 96] for inst in [1, 2, 3, 4, 5]]
