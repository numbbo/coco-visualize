from cocoviz import read_coco_dataset

def test_read_single_alg():
    res = read_coco_dataset("22/cma-es-pycma")
    assert len(res) == 24 * 15 * 6 # 24 functions, 15 instances, 6 dimensions

def test_read_list():
    res = read_coco_dataset(["22/cma-es-pycma", "09/nelderdoerr"])
    assert "NELDERDOERR_doerr" in res.algorithms
    assert "CMA-ES-pycma_Gharafi" in res.algorithms
    assert len(res) == 3960
