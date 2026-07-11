import pytest

from cocoviz.exceptions import IndicatorMismatchException
from cocoviz.result import Result, ResultSet, ProblemDescription


HV_a1 = {
    "fevals": [1, 2, 3, 10, 20, 50, 100],
    "hypervolume": [10, 20, 30, 40, 50, 60, 70],
    "r2": [20, 40, 60, 80, 100, 120, 140],
}

HV_a2 = {
    "fevals": [1, 2, 3, 10, 20, 50, 100],
    "hypervolume": [12, 22, 32, 42, 52, 62, 72],
    "r2": [22, 42, 62, 82, 102, 122, 142],
}

HV_a2_nor2 = HV_a2.copy()
del HV_a2_nor2["r2"]


def test_indicator_mismatch():
    pd_f1 = ProblemDescription("f1", "i1", 10, 2)
    r1 = Result("a1", pd_f1, HV_a1)
    r2 = Result("a2", pd_f1, HV_a2_nor2)

    with pytest.raises(IndicatorMismatchException):
        rs = ResultSet([r1, r2])

    with pytest.raises(IndicatorMismatchException):
        rs = ResultSet([r1, r2])

    with pytest.raises(IndicatorMismatchException):
        rs = ResultSet()
        rs.append(r1)
        rs.append(r2)

    with pytest.raises(IndicatorMismatchException):
        rs = ResultSet()
        rs.append(r2)
        rs.append(r1)


def test_str():
    pd_f1 = ProblemDescription("f1", "i1", 10, 2)
    pd_f2 = ProblemDescription("f2", "i1", 10, 2)
    r1 = Result("a1", pd_f1, HV_a1)
    r2 = Result("a2", pd_f2, HV_a2)
    r3 = Result("a1", pd_f2, HV_a2)

    rs = ResultSet([r1, r2, r3])

    assert str(rs) == "ResultSet with 3 result(s) from 2 algorithm(s) on 2 function(s)"
    assert repr(rs) == str(rs)
 
