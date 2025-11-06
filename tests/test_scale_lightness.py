# test_scale_lightness.py
from typing import Sequence
import pytest
from math import isclose

# Adjust this import to your project/module layout
# e.g., from mypackage.colors import scale_lightness
from cocoviz.utilities import scale_lightness


def rgb_approx(a: Sequence[float], b: Sequence[float], tol: float = 1e-6):
    return all(isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "rgb",
    [
        [0.0, 0.0, 0.0],  # black
        [1.0, 1.0, 1.0],  # white
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [0.2, 0.4, 0.6],  # arbitrary
    ],
)
def test_identity_scale_returns_same(rgb: list[float]):
    out = scale_lightness(rgb, 1.0)
    assert rgb_approx(out, rgb)


@pytest.mark.parametrize(
    "rgb, scale, expected",
    [
        ((1.0, 0.0, 0.0), 0.5, (0.5, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), 0.5, (0.0, 0.5, 0.0)),
        ((0.0, 0.0, 1.0), 0.5, (0.0, 0.0, 0.5)),
        ((0.2, 0.2, 0.2), 2.0, (0.4, 0.4, 0.4)),
        ((0.3, 0.3, 0.3), 0.5, (0.15, 0.15, 0.15)),
    ],
)
def test_known_values(rgb, scale, expected):
    out = scale_lightness(rgb, scale)
    assert rgb_approx(out, expected, tol=1e-12)


@pytest.mark.parametrize(
    "rgb, scale",
    [
        ((0.1, 0.2, 0.3), 0.3),
        ((0.8, 0.1, 0.5), 0.7),
        ((0.9, 0.9, 0.1), 0.9),
    ],
)
def test_outputs_stay_in_unit_interval(rgb, scale):
    out = scale_lightness(rgb, scale)
    assert all(0.0 <= c <= 1.0 for c in out)


def test_clamps_to_white_on_large_scale():
    out = scale_lightness([1.0, 0.0, 0.0], 3.0)
    assert rgb_approx(out, [1.0, 1.0, 1.0])


def test_clamps_to_black_on_small_scale():
    out = scale_lightness([1.0, 0.0, 0.0], -3.0)
    assert rgb_approx(out, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("scale", [0.25, 0.5, 0.75, 1.5, 2.0])
def test_grays_remain_gray(scale: float):
    out = scale_lightness((0.4, 0.4, 0.4), scale)
    # stays gray: all channels equal
    assert isclose(out[0], 0.4 * scale, abs_tol=1e-12)
    assert isclose(out[1], 0.4 * scale, abs_tol=1e-12)
    assert isclose(out[2], 0.4 * scale, abs_tol=1e-12)


@pytest.mark.parametrize("scale", [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0])
def test_pure_color_darkening_preserves_zero_channels(scale: float):
    out = scale_lightness([0.0, 1.0, 0.0], scale)
    assert isclose(out[0], 0.0, abs_tol=1e-12) and isclose(out[2], 0.0, abs_tol=1e-12)


def test_lightening_pure_color_until_white():
    out = scale_lightness([0.0, 0.0, 1.0], 10.0)
    assert rgb_approx(out, [1.0, 1.0, 1.0])
