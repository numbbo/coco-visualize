import logging
import polars as pl

from polars.testing import assert_frame_equal

from cocoviz.result import Result, ProblemDescription


HV_a1 = {
    "fevals":      [  1,   2,   3,  10,  20,  50, 100],
    "hypervolume": [ 10,  20,  30,  40,  50,  60,  70],
    "r2":          [ 20,  40,  60,  80, 100, 120, 140]
}

HV_a1_nofevals = {
    "nofevals":      [  1,   2,   3,  10,  20,  50, 100],
    "hypervolume": [ 10,  20,  30,  40,  50,  60,  70],
    "r2":          [ 20,  40,  60,  80, 100, 120, 140]
}

pd_f1 = ProblemDescription("f1", "i1", 10, 2)


def test_fevals_column(caplog):
    with caplog.at_level(logging.WARNING):    
        res = Result("a1", pd_f1, HV_a1_nofevals) # noqa: F841
    assert "Assuming first column" in caplog.text


def test_at_indicator():
    targets = [15, 25, 35, 120]
    

    res = Result("a1", pd_f1, HV_a1)
    
    ind = res.at_indicator("hypervolume", targets)

    true_ind = pl.DataFrame([
        pl.Series("__fevals", [2.0, 3.0, 10.0, 100.0], dtype=pl.Float64),
        pl.Series("hypervolume", [15, 25, 35, 120], dtype=pl.Int64),
        pl.Series("__target_hit", [1.0, 1.0, 1.0, 0.0], dtype=pl.Float64),
        pl.Series("__fevals_dim", [0.2, 0.30, 1.0, 10.0], dtype=pl.Float64),
    ])
    
    assert_frame_equal(ind._data, true_ind)
