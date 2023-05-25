import copy
from functools import reduce
import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple, Union, cast, overload

import numpy as np
from .einsum_script import EinsumScript, IncompatibleShapeError
import opt_einsum

Shape = Tuple[int, ...]
Subscript = Union[Shape, str, Callable[[List[Shape]],
                                       'Subscript'], List['Subscript'], Tuple['Subscript', ...]]


class _ScriptAdder:
    def __init__(self, scripts: List[EinsumScript]) -> None:
        self.scripts = scripts
        self.cache: Dict[Tuple[int, int], EinsumScript] = {}

    def __call__(self, start: int, stop: int) -> EinsumScript:
        try:
            (cache_start, cache_stop), output = max(((key, val) for key, val in self.cache.items(
            ) if key[0] >= start and key[1] <= stop), key=lambda item: item[0][1] - item[0][0])
            if cache_start != start:
                output = self(start, cache_start) + output
            if cache_stop != stop:
                output += self(cache_stop, stop)
            return output
        except ValueError:
            return reduce(lambda x, y: x+y, self.scripts[start:stop])


def _improve_splits(scripts: List[EinsumScript], output_sizes: List[int], split_indices: List[int], script_adder: _ScriptAdder) -> List[int]:
    # Find the combinations of indices which would result in the smallest total memory
    combinations = itertools.product(
        *[list(range(x+1, y+1)) for x, y in zip([0, *split_indices], split_indices)])
    sorted_groups = sorted(combinations, key=lambda comb: sum(
        output_sizes[x-1] for x in comb))

    for indices in sorted_groups:
        try:
            for x, y in zip([0, *indices], [*indices, len(scripts)]):
                script_adder(x, y)
            return list(indices)
        except IncompatibleShapeError:
            pass
    return split_indices


def _collapse(scripts: List[EinsumScript]) -> List[EinsumScript]:
    if len(scripts) == 0:
        return []
    split_indices: List[int] = []
    splits: List[EinsumScript] = [scripts[0]]
    output_sizes: List[int] = []
    for i, script in enumerate(scripts[1:], 1):
        script.simplify()
        try:
            output_sizes.append(math.prod(splits[-1].output_shape))
            splits[-1] += script
        except IncompatibleShapeError:
            splits.append(script)
            split_indices.append(i)

    split_min_sizes = [min(output_sizes[x:y])
                       for x, y in zip([None, *split_indices], split_indices)]
    if len(splits) == 1 or all(output_sizes[i-1] == split_min_size for i, split_min_size in zip(split_indices, split_min_sizes)):
        return splits
    else:
        script_adder = _ScriptAdder(scripts)
        split_indices = _improve_splits(
            scripts, output_sizes, split_indices, script_adder)
        return [script_adder(x, y) for x, y in zip([0, *split_indices], [*split_indices, len(scripts)])]


def compile_einsum_args(subscripts: List[Subscript], input_shapes: List[Tuple[int, ...]],
                        simplify: Union[str, bool] = True) -> Tuple[List[EinsumScript], Tuple[int, ...]]:
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
    output_scripts = _collapse(scripts)

    if simplify == 'max':
        for script in output_scripts:
            script.simplify()
    elif simplify:
        input_shape_iter = iter(input_shapes)
        first_val = next(input_shape_iter)
        for script in output_scripts:
            script.simplify()
            script.match_splits([first_val] + [next(input_shape_iter)
                                for _ in script.inputs[1:]])
            first_val = None
    return output_scripts, output_shape


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
        output_scripts, output_shape = compile_einsum_args(
            subs, [op.shape for op in ops], simplify=simplify)
    else:
        output_scripts = [script]

    ops_iter = iter(ops)
    state = next(ops_iter)

    for script in output_scripts:
        reshaped_ops = [np.reshape(op, shape)
                        for shape, op in zip(script.input_shapes, itertools.chain([state], ops_iter))]
        state = cast(np.ndarray, opt_einsum.contract(
            str(script), *reshaped_ops, **kwargs))

    return state.reshape(cast(Shape, output_shape))
