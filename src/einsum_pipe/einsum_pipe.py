from functools import reduce
from typing import List

import numpy as np
from .einsum_script import EinsumScript


def einsum_pipe(*args):
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
    unused_shapes = [op.shape for op in ops]
    scripts: List[EinsumScript] = []

    while len(subs) > 0:
        sub = subs.pop(0)
        if isinstance(sub, str):
            # Normal subscript
            nargs = sub.count(',') + 1
            input_shapes = [list(unused_shapes.pop(0)) for _ in range(nargs)]
            script = EinsumScript.parse(input_shapes, sub)
            unused_shapes.insert(0, script.output_shape)
            scripts.append(script)
        elif callable(sub):
            # Lazy argument
            subs.insert(0, sub(unused_shapes))
        elif isinstance(sub, (list, tuple)):
            if isinstance(sub[0], int):
                # Reshape
                unused_shapes[0] = tuple(sub)
            else:
                # Inner list which needs to be flattened
                for val in sub[::-1]:
                    subs.insert(0, val)

    output_shape = unused_shapes[0]
    output_script = reduce(lambda x, y: x+y, scripts)
    output_script.simplify()
    reshaped_ops = [np.reshape(op, [comp.size for comp in inp])
                    for op, inp in zip(ops, output_script.inputs)]
    raw_output: np.ndarray = np.einsum(str(output_script), *reshaped_ops)
    return raw_output.reshape(output_shape)
