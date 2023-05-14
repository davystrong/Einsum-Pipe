from functools import reduce
from typing import List, cast

import numpy as np
from .einsum_script import EinsumScript
from .ops import LazySubs


def einsum_pipe(*args):
    subs = [arg for arg in args if isinstance(
        arg, (str, list, tuple)) or callable(arg)]
    ops = []
    for arg in args:
        if not (isinstance(arg, (str, list, tuple)) or callable(arg)):
            try:
                assert arg.shape is not None
                ops.append(arg)
            except AttributeError:
                ops.append(np.array(arg))
    ops_index = 0
    scripts: List[EinsumScript] = []

    while len(subs) > 0:
        input_shapes = []
        sub = subs.pop(0)
        if not (isinstance(sub, str) or callable(sub)):
            input_shapes.append(sub)
            sub = subs.pop(0)
        elif len(scripts) != 0:
            input_shapes.append(scripts[-1].output_shape)
        assert isinstance(sub, str) or callable(sub)

        try:
            nargs = sub.count(',') + 1
        except AttributeError:
            try:
                nargs = cast(LazySubs, sub).nargs
            except AttributeError:
                nargs = 1
        n_missing_args = nargs - len(input_shapes)

        input_shapes.extend(tuple(x.shape)
                            for x in ops[ops_index:ops_index+n_missing_args])
        ops_index += n_missing_args

        if callable(sub):
            sub = sub(input_shapes)
        scripts.append(EinsumScript.parse(input_shapes, sub))

    output_script = reduce(lambda x, y: x+y, scripts)
    output_shape = scripts[-1].output_shape
    output_script.simplify()
    reshaped_ops = [np.reshape(op, [comp.size for comp in inp])
                    for op, inp in zip(ops, output_script.inputs)]
    raw_output: np.ndarray = np.einsum(str(output_script), *reshaped_ops)
    return raw_output.reshape(output_shape)
