import numpy as np
from einsum_pipe import einsum_pipe, ops


def test_trace():
    A = np.random.rand(10, 10, 10, 10)
    assert np.allclose(einsum_pipe(ops.trace(-3, 3), A),
                       np.trace(A, axis1=-3, axis2=3))


def test_diagonal():
    A = np.random.rand(10, 20, 10, 30)
    assert np.allclose(einsum_pipe(ops.diagonal(0, 2), A),
                       np.diagonal(A, axis1=0, axis2=2))
    assert np.allclose(einsum_pipe(ops.diagonal(-4, 2), A),
                       np.diagonal(A, axis1=-4, axis2=2))


def test_sum():
    A = np.random.rand(10, 20, 10, 30)
    assert np.allclose(einsum_pipe(ops.sum((3, -3, 0)), A),
                       np.sum(A, (3, -3, 0)))


def test_transpose():
    A = np.random.rand(10, 20, 10, 30)
    assert np.allclose(einsum_pipe(ops.transpose((3, -3, 0, -2)), A),
                       np.transpose(A, (3, -3, 0, -2)))


def test_inner():
    A = np.random.rand(2, 3, 30)
    B = np.random.rand(30)
    assert np.allclose(einsum_pipe(ops.inner(), A, B),
                       np.inner(A, B))


def test_outer():
    A = np.random.rand(2, 3, 30)
    B = np.random.rand(2, 3, 30)
    assert np.allclose(einsum_pipe(ops.outer(), A, B),
                       np.outer(A, B))
