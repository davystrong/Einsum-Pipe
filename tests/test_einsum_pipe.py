import numpy as np
from einsum_pipe import einsum_pipe, compile_einsum_args


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


def test_unequal_broadcasting():
    A = np.random.rand(10, 20, 20)
    B = np.random.rand(10, 20, 30)
    args = [
        'ab...,...c->bca...',
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


def test_implicit_broadcast():
    A = np.random.rand(10)
    B = np.random.rand(10)
    args = [
        '...,...',
        A, B
    ]
    assert np.allclose(einsum_pipe(*args), einsum_pipe_simple(*args))


def test_simplify():
    # Make a discontiguous array
    A = np.random.rand(3, 3, 3, 9).transpose((0, 2, 1, 3))
    assert not A.flags['CONTIGUOUS']

    script, output_shape = compile_einsum_args(
        [(27, 3, 3), 'abc->bc'], [A.shape], True)
    script_max, output_shape_max = compile_einsum_args(
        [(27, 3, 3), 'abc->bc'], [A.shape], 'max')

    X = A.reshape(script.input_shapes[0])
    Y = A.reshape(script_max.input_shapes[0])
    assert all(x == y for x, y in zip(output_shape, output_shape_max))
    assert np.shares_memory(A, X) and not np.shares_memory(A, Y)
    assert np.allclose(einsum_pipe(A, script=script, output_shape=output_shape), einsum_pipe(
        A, script=script_max, output_shape=output_shape_max))
