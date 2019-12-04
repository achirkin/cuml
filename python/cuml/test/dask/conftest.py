import pytest

from dask_cuda import LocalCUDACluster


@pytest.fixture(scope="module")
def cluster():
    cluster = LocalCUDACluster(protocol="ucx", threads_per_worker=1)
    yield cluster
    cluster.close()
