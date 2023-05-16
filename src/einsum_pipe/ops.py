from typing import Any, Callable, List, Tuple, Union
import math
from .einsum_script import _get_char


Subs = Union[str, Callable[[List[List[int]]], Union[str, List, Tuple]]]


def _build_base_subs(axis1: int, axis2: int) -> List[str]:
    # Unfortunately this function isn't universal: if the negative indices overlap with the
    # positive indices, i.e. an array with 5 dims and a reference to axis 4 and -5 produces
    # an invalid output. As such, it's only safe to use this if both axes are positive or both is negative.
    axis1, axis2 = sorted([axis1, axis2])
    subs = [_get_char(i) for i in range(axis2 + 1)] + ['...'] + \
        [_get_char(max(axis2 + 1, 0) + i - axis1) for i in range(axis1, 0)]
    return subs


def _same_sign(a: int, b: int) -> bool:
    return (a >= 0 and b >= 0) or (a < 0 and b < 0)


def _pop_many(data: List[Any], *axes: int) -> None:
    axes = tuple(sorted(ax if ax >= 0 else (len(data) + ax) for ax in axes))
    for ax in axes[::-1]:
        data.pop(ax)


def trace(axis1=0, axis2=1) -> Subs:
    if _same_sign(axis1, axis2):
        subs = _build_base_subs(axis1, axis2)
        subs[axis2] = subs[axis1]
        return ''.join(subs)
    else:
        def inner(input_shapes: List[List[int]]) -> str:
            subs = [_get_char(i) for i in range(len(input_shapes[0]))]
            subs[axis2] = subs[axis1]
            return ''.join(subs)
        return inner


def diagonal(axis1=0, axis2=1) -> Subs:
    axis1, axis2 = sorted([axis1, axis2])

    def from_subs(subs):
        subs[axis2] = subs[axis1]
        in_subs = ''.join(subs)
        sub = subs[axis1]
        _pop_many(subs, axis1, axis2)
        out_subs = ''.join(subs) + sub
        return in_subs + '->' + out_subs

    if _same_sign(axis1, axis2):
        subs = _build_base_subs(axis1, axis2)
        return from_subs(subs)
    else:
        def inner(input_shapes: List[List[int]]) -> str:
            subs = [_get_char(i) for i in range(len(input_shapes[0]))]
            return from_subs(subs)
        return inner


def diag() -> Subs:
    return diagonal()


def sum(axis: Union[None, int, List[int], Tuple[int, ...]] = None) -> Subs:
    if axis is None:
        return '...->'
    else:
        if isinstance(axis, int):
            axis = [axis]
        axis = sorted(axis)

        def from_subs(subs):
            in_subs = ''.join(subs)
            for ax in axis:
                subs[ax] = None  # type: ignore
            subs = [sub for sub in subs if sub is not None]
            out_subs = ''.join(subs)
            return in_subs + '->' + out_subs

        if _same_sign(axis[0], axis[-1]):
            subs = _build_base_subs(axis[0], axis[-1])
            return from_subs(subs)
        else:
            def inner(input_shapes: List[List[int]]) -> str:
                subs = [_get_char(i) for i in range(len(input_shapes[0]))]
                return from_subs(subs)
            return inner


def transpose(axes: Union[None, List[int], Tuple[int, ...]]) -> Subs:
    if axes is None:
        def inner(input_shapes: List[List[int]]) -> str:
            in_subs = ''.join([_get_char(i)
                              for i in range(len(input_shapes[0]))])
            return in_subs[::-1]
        return inner
    else:
        in_subs = [_get_char(i) for i in range(len(axes))]
        out_subs = [in_subs[a] for a in axes]
        return ''.join(in_subs) + '->' + ''.join(out_subs)


def inner() -> Subs:
    return '...a,...a'


def outer() -> Subs:
    def inner(input_shapes: List[List[int]]):
        out_shape = [math.prod(input_shapes[0]), math.prod(input_shapes[1])]
        in_subs = [_get_char(i)
                   for i in range(len(input_shapes[0]) + len(input_shapes[1]))]
        in_subs.insert(len(input_shapes[0]), ',')
        in_subs = ''.join(in_subs)
        return in_subs, out_shape
    return inner
