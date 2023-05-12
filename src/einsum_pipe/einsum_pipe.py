from functools import reduce
from typing import List

import numpy as np
from .einsum_script import EinsumScript


def einsum_pipe(*args):
    subs = [arg for arg in args if isinstance(arg, (str, list, tuple))]
    ops = [arg for arg in args if not isinstance(arg, (str, list, tuple))]
    ops_index = 0
    scripts: List[EinsumScript] = []

    while len(subs) > 0:
        input_shapes = []
        sub = subs.pop(0)
        if not isinstance(sub, str):
            input_shapes.append(sub)
            sub = subs.pop(0)
        assert isinstance(sub, str)

        args = sub.count(',') + 1 - len(input_shapes)

        input_shapes.extend(tuple(x.shape)
                            for x in ops[ops_index:ops_index+args])
        ops_index += args

        x = EinsumScript.parse(input_shapes, sub)
        scripts.append(x)

    output_script = reduce(lambda x, y: x+y, scripts)
    output_script.simplify()
    reshaped_ops = [np.reshape(op, [comp.size for comp in inp])
                    for op, inp in zip(ops, output_script.inputs)]
    raw_output: np.ndarray = np.einsum(str(output_script), *reshaped_ops)
    return raw_output.reshape([comp.size for comp in scripts[-1].outputs])
