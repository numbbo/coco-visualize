from unittest.mock import Mock, patch

from polars.testing import assert_frame_equal

from cocoviz.result import Result, ProblemDescription


HV_a1 = {
    "fevals": [1, 2, 3, 10, 20, 50, 100],
    "hypervolume": [10, 20, 30, 40, 50, 60, 70],
    "r2": [20, 40, 60, 80, 100, 120, 140],
}

pd_f1 = ProblemDescription("f1", "i1", 10, 2)


def test_parquet_roundtrip(tmp_path):
    res = Result("a1", pd_f1, HV_a1)

    path = tmp_path / "result.parquet"
    res.to_parquet(path)
    loaded = Result.from_parquet(path)

    assert loaded.algorithm == res.algorithm
    assert loaded.problem == res.problem
    assert_frame_equal(loaded._data, res._data)


def test_from_parquet_url(tmp_path):
    res = Result("a1", pd_f1, HV_a1)

    path = tmp_path / "result.parquet"
    res.to_parquet(path)
    content = path.read_bytes()

    response = Mock()
    response.content = content
    response.raise_for_status = Mock()

    with patch("requests.get", return_value=response) as get:
        loaded = Result.from_parquet("https://example.com/result.parquet")

    get.assert_called_once_with("https://example.com/result.parquet")
    response.raise_for_status.assert_called_once()
    assert loaded.algorithm == res.algorithm
    assert loaded.problem == res.problem
    assert_frame_equal(loaded._data, res._data)
