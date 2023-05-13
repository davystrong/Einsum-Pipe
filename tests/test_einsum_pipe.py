import pytest
import numpy as np
from einsum_pipe import einsum_pipe


def einsum_pipe_simple(*args):
    subs = [arg for arg in args if isinstance(arg, (str, list, tuple))]
    ops = [arg for arg in args if not isinstance(arg, (str, list, tuple))]

    state: np.ndarray = ops.pop(0)
    for sub in subs:
        if isinstance(sub, str):
            extra_state = [ops.pop(0) for _ in range(sub.count(','))]
            state = np.einsum(sub, state, *extra_state)
        else:
            state = np.reshape(state, sub)

    return state


def test_single_arg_no_reshape():
    A = np.random.rand(10, 20, 30)
    args = [
        'abc->bca',
        'abc->ab',
        'ab->a',
        A
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_single_arg_with_reshape():
    A = np.random.rand(10, 20, 30)
    args = [
        'abc->bca',
        (10, 2, 5, 3, 2, 10),
        'abdfbg->afg',
        'abc->a',
        A
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_multi_args():
    A = np.random.rand(10, 20, 30)
    B = np.random.rand(10, 20, 30)
    args = [
        'abc,dec->bcad',
        (10, 2, 5, 3, 2, 10, 5, 2),
        'abcdbace->eacd',
        'abcd->acd',
        A, B
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_implicit_mode():
    A = np.random.rand(10, 20, 30)
    B = np.random.rand(10, 20, 30)
    args = [
        'abc,dec',
        'acbd',
        'cbdd',
        A, B
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_broadcasting():
    A = np.random.rand(10, 20, 30)
    B = np.random.rand(10, 20, 30)
    args = [
        'abc,...c->bca...',
        (10, 2, 5, 3, 2, 10, 5, 2, 20),
        'abcdbace...->eacd...',
        'abcd...->acd...',
        'acb...',
        'c...ba',
        A, B
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_implicit_array_creation():
    A = np.random.rand(10, 20, 30)
    args = [
        'abc,->bca',
        'abc->a',
        A, 10
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_multistage_args():
    A = np.random.rand(10, 20, 30)
    B = np.random.rand(10, 20, 30)
    args = [
        'bac',
        'abc,dec->bcad',
        (10, 2, 5, 3, 2, 10, 5, 2),
        'abcdbace,->eacd',
        'abcd->acd',
        A, B, 10
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))
