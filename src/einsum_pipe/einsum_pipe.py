import copy
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union, cast, overload

import numpy as np
from .einsum_script import EinsumScript

Shape = Tuple[int, ...]
Subscript = Union[Shape, str, Callable[[List[Shape]],
                                       'Subscript'], List['Subscript'], Tuple['Subscript', ...]]


def compile_einsum_args(subscripts: List[Subscript], input_shapes: List[Tuple[int, ...]],
                        simplify: Union[str, bool] = True) -> Tuple[EinsumScript, Tuple[int, ...]]:
    unused_shapes = copy.copy(input_shapes)
    scripts: List[EinsumScript] = []

    while len(subscripts) > 0:
        sub = subscripts.pop(0)
        if isinstance(sub, str):
            # Normal subscript
            nargs = sub.count(',') + 1
            next_input_shapes = [list(unused_shapes.pop(0))
                                 for _ in range(nargs)]
            script = EinsumScript.parse(next_input_shapes, sub)
            unused_shapes.insert(0, script.output_shape)
            scripts.append(script)
        elif callable(sub):
            # Lazy argument
            subscripts.insert(0, sub(unused_shapes))
        elif isinstance(sub, (list, tuple)):
            if isinstance(sub[0], int):
                # Reshape
                unused_shapes[0] = cast(Tuple[int], tuple(sub))
            else:
                # Inner list which needs to be flattened
                for val in sub[::-1]:
                    subscripts.insert(0, cast(Subscript, val))

    output_shape = unused_shapes[0]
    output_script = reduce(lambda x, y: x+y, scripts)
    if simplify == 'max':
        output_script.simplify()
    elif simplify:
        output_script.simplify(input_shapes)
    return output_script, output_shape


@overload
def einsum_pipe(*args, simplify=True, **kwargs) -> np.ndarray: ...


@overload
def einsum_pipe(*args, simplify=True,
                script: EinsumScript, output_shape: Tuple[int, ...], **kwargs) -> np.ndarray: ...


def einsum_pipe(*args, simplify=True,
                script: Optional[EinsumScript] = None, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> np.ndarray:
    assert (script is None and output_shape is None) or (
        script is not None and output_shape is not None)
    subs = []
    ops: List[np.ndarray] = []
    for arg in args:
        if isinstance(arg, (str, list, tuple)) or callable(arg):
            if isinstance(arg, list) and not isinstance(arg[0], int):
                subs.extend(arg)
            else:
                subs.append(arg)
        else:
            try:
                assert arg.shape is not None
                ops.append(arg)
            except AttributeError:
                ops.append(np.array(arg))

    if script is None:
        output_script, output_shape = compile_einsum_args(
            subs, [op.shape for op in ops], simplify=simplify)
    else:
        output_script = script

    reshaped_ops = [np.reshape(op, [comp.size for comp in inp])
                    for op, inp in zip(ops, output_script.inputs)]
    raw_output: np.ndarray = np.einsum(
        str(output_script), *reshaped_ops, **kwargs)
    return raw_output.reshape(cast(Shape, output_shape))
