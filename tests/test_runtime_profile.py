import pytest

from cocoviz.exceptions import BadRuntimeProfileException
from cocoviz import runtime_profiles, Result, ResultSet, ProblemDescription


HV_a1 = {
    "fevals":      [  1,   2,   3,  10,  20,  50, 100],
    "hypervolume": [ 10,  20,  30,  40,  50,  60,  70],
    "r2":          [ 20,  40,  60,  80, 100, 120, 140]
}

HV_a2 = {
    "fevals":      [  1,   2,   3,  10,  20,  50, 100],
    "hypervolume": [ 12,  22,  32,  42,  52,  62,  72],
    "r2":          [ 22,  42,  62,  82, 102, 122, 142]
}

def test_aggregate_over_objectives():
    pd_f1_d2 = ProblemDescription("f1", "i1", 10, 2)
    pd_f1_d3 = ProblemDescription("f1", "i1", 10, 3)
    rs = ResultSet([
        Result("a1", pd_f1_d2, HV_a1),
        Result("a1", pd_f1_d3, HV_a1)
    ])

    with pytest.raises(BadRuntimeProfileException):
        runtime_profiles(rs, indicator="hypervolume")

def test_aggregate_over_variables():
    pd_f1_d2 = ProblemDescription("f1", "i1", 10, 2)
    pd_f1_d3 = ProblemDescription("f1", "i1", 20, 2)
    rs = ResultSet([
        Result("a1", pd_f1_d2, HV_a1),
        Result("a1", pd_f1_d3, HV_a1)
    ])

    with pytest.raises(BadRuntimeProfileException):
        runtime_profiles(rs, indicator="hypervolume")