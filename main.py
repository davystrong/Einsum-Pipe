from __future__ import annotations
import copy
from functools import reduce
from typing import Dict, Generator, Generic, Iterator, List, Self, Tuple, TypeVar, Union, cast
import numpy as np
import math
from collections.abc import MutableMapping


def einsum_simple(*args):
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


class EinsumComp:
    def __init__(self, size: int) -> None:
        self.size = size


class NullTag:
    pass


K = TypeVar('K')
V = TypeVar('V')


class BiDict(MutableMapping, Generic[K, V]):
    """A custom dictionary that keeps track of the inverse mapping from values to keys. Values must be unique too
    """

    def __init__(self):
        self.store: Dict[K, V] = {}
        self.inverse: Dict[V, K] = {}

    def __getitem__(self, key: K) -> V:
        return self.store[key]

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            del self.inverse[self[key]]
        self.inverse[value] = key
        self.store[key] = value

    def __delitem__(self, key: K) -> None:
        del self.inverse[self[key]]
        del self.store[key]

    def values(self):
        return self.inverse.keys()

    def __iter__(self) -> Iterator[K]:
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class EinsumScript:
    def __init__(self, inputs: List[List[EinsumComp]], outputs: List[EinsumComp]) -> None:
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def parse(cls, input_shapes: List[List[int]], subscripts: str) -> Self:
        subscripts = subscripts.replace(' ', '')
        # Easier to deal with broadcasting as a single character
        subscripts = subscripts.replace('...', '?')
        # The broadcasting character is automatically sorted to the start
        letters = sorted(subscripts.replace(',', '').replace('->', ''))
        if '->' not in subscripts:
            output_letters = [l for l in letters if l ==
                              '?' or letters.count(l) == 1]
            subscripts += '->' + ''.join(output_letters)
        letter_dict = {v: EinsumComp(0) for v in set(letters) if v != '?'}

        inputs_subs, output_subs = subscripts.split('->')
        inputs: List[List[EinsumComp]] = []
        broadcast_comps: List[EinsumComp] = []
        for sub, shape in zip(inputs_subs.split(','), input_shapes):
            inputs.append([])
            for c in sub:
                if c == '?':
                    # Broadcasting works from the last axis to the first and shares these axes with other broadcasts
                    undefined_axes = len(shape) - (len(sub) - 1)
                    for _ in range(undefined_axes - len(broadcast_comps)):
                        broadcast_comps.insert(0, EinsumComp(0))
                    inputs[-1].extend(broadcast_comps[-undefined_axes:])
                else:
                    inputs[-1].append(letter_dict[c])

        outputs: List[EinsumComp] = []
        for c in output_subs:
            if c == '?':
                # All broadcasted axes are added in order
                outputs.extend(broadcast_comps)
            else:
                outputs.append(letter_dict[c])

        script = EinsumScript(inputs, outputs)
        for inp, shape in zip(inputs, input_shapes):
            assert len(inp) == len(shape)
            for comp, dim in zip(inp, shape):
                comp.size = dim

        return script

    def split_comp(self, comp: EinsumComp, part_sizes: List[int]) -> None:
        repeats = [EinsumComp(size) for size in part_sizes[1:]]
        comp.size = part_sizes[0]
        for inp in [*self.inputs, self.outputs]:
            for i in range(len(inp)-1, -1, -1):
                if inp[i] == comp:
                    for rep in repeats[::-1]:
                        inp.insert(i+1, rep)

    def remove_ones(self):
        for inp in [*self.inputs, self.outputs]:
            for i in range(len(inp)-1, -1, -1):
                if inp[i].size == 1:
                    inp.pop(i)

    def transform_shapes(self, input_shapes: List[List[int]]) -> List[int]:
        assert len(input_shapes) == len(self.inputs)
        shape_dict = {sub: comp for subs, shape in zip(
            self.inputs, input_shapes) for sub, comp in zip(subs, shape)}
        return [shape_dict[out_sub] for out_sub in self.outputs]

    def simplify(self):
        next_map: BiDict[Union[NullTag, EinsumComp],
                         Union[NullTag, EinsumComp]] = BiDict()

        for comps in [*self.inputs, self.outputs]:
            prev = NullTag()
            for comp in comps:
                if prev in next_map:
                    if next_map[prev] != comp:
                        next_map[NullTag()] = next_map[prev]
                        next_map[NullTag()] = comp
                        next_map[prev] = NullTag()
                elif comp in next_map.values():
                    # Don't need to check if key is already the same as this will be caught by the previous condition
                    key = next_map.inverse[comp]
                    next_map[key] = NullTag()
                    next_map[prev] = NullTag()
                    next_map[NullTag()] = comp
                else:
                    next_map[prev] = comp
                prev = comp
            next_map[prev] = NullTag()

        null_tags = [key for key in next_map if isinstance(key, NullTag)]
        group_pairs: List[Tuple[List[EinsumComp], EinsumComp]] = []
        for tag in null_tags:
            seq: List[EinsumComp] = []
            while not isinstance(next_map[tag], NullTag):
                seq.append(cast(EinsumComp, next_map[tag]))
                tag = next_map[tag]
            if len(seq) > 1:
                group_pairs.append(
                    (seq, EinsumComp(math.prod(comp.size for comp in seq))))

        for comps in [*self.inputs, self.outputs]:
            for group, new_comp in group_pairs:
                while group[0] in comps:
                    i = comps.index(group[0])
                    comps[i] = new_comp
                    for _ in range(len(group) - 1):
                        comps.pop(i + 1)

    def simplified(self) -> Self:
        val = copy.deepcopy(self)
        val.simplify()
        return val

    @staticmethod
    def _get_char(index: int) -> str:
        return chr((ord('a') if index < 26 else (ord('A') - 26)) + index)

    def __str__(self) -> str:
        comps = list(set(comp for inp in self.inputs for comp in inp))

        subs = []
        for inp in self.inputs:
            subs.append(''.join(self._get_char(comps.index(comp))
                        for comp in inp))

        output_str = ''.join(self._get_char(comps.index(comp))
                             for comp in self.outputs)

        return ','.join(subs) + '->' + output_str

    def __add__(self, rhs: Self) -> Self:
        lhs = copy.deepcopy(self)
        rhs = copy.deepcopy(rhs)
        lhs_out_iter = rev_mut_iter(lhs.outputs)
        rhs_in_iter = rev_mut_iter(rhs.inputs[0])

        lhs_out_val = next(lhs_out_iter)
        rhs_in_val = next(rhs_in_iter)

        try:
            while True:
                if lhs_out_val.size == rhs_in_val.size:
                    lhs_out_val = next(lhs_out_iter)
                    rhs_in_val = next(rhs_in_iter)
                elif lhs_out_val.size > rhs_in_val.size:
                    lhs.split_comp(lhs_out_val, [
                        lhs_out_val.size // rhs_in_val.size, rhs_in_val.size])
                    rhs_in_val = next(rhs_in_iter)
                else:
                    rhs.split_comp(rhs_in_val, [
                        rhs_in_val.size // lhs_out_val.size, lhs_out_val.size])
                    lhs_out_val = next(lhs_out_iter)
        except StopIteration:
            pass

        assert len(lhs.outputs) == len(rhs.inputs[0])
        assert all(x.size == y.size for x, y in zip(
            lhs.outputs, rhs.inputs[0]))

        for i, x in enumerate(rhs.inputs[0]):
            val = lhs.outputs[i]
            lhs.outputs[i] = x
            for inp in lhs.inputs:
                if val in inp:
                    for j, y in enumerate(inp):
                        if y == val:
                            inp[j] = x

        return EinsumScript(lhs.inputs + rhs.inputs[1:], rhs.outputs)


T = TypeVar('T')


def rev_mut_iter(data: List[T]) -> Generator[T, None, None]:
    for i in range(len(data)-1, -1, -1):
        yield data[i]


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
        print(str(x))
        scripts.append(x)

    output_script = reduce(lambda x, y: x+y, scripts)
    output_script.simplify()
    reshaped_ops = [np.reshape(op, [comp.size for comp in inp]) for op, inp in zip(ops, output_script.inputs)]
    raw_output: np.ndarray = np.einsum(str(output_script), *reshaped_ops)
    return raw_output.reshape([comp.size for comp in scripts[-1].outputs])


def get_subscripts(keep_indices: List[int], length: int, start_at='a') -> str:
    start = ord(start_at)
    assert all(k < 2*length for k in keep_indices)
    in_subs_1 = list(range(start, start + length))*2
    in_subs_2 = list(range(start + length, start + 2*length))*2
    out_subs = [start + k for k in keep_indices]
    for i, k in enumerate(keep_indices):
        next_sub = start + 2*length + i
        if k < length:
            in_subs_1[length + k] = next_sub
        else:
            in_subs_2[k] = next_sub
        out_subs.append(next_sub)
    return ''.join(chr(x) for x in in_subs_1) + ',' + ''.join(chr(x) for x in in_subs_2) + '->' + ''.join(chr(x) for x in out_subs)


if __name__ == '__main__':
    A = np.random.rand(32, 32)
    B = np.random.rand(32, 32)

    print(get_subscripts([0, 1], 5))

    X_obj = np.einsum('ikok,ll->io',
                      A.reshape([4, 8, 4, 8]), B).reshape((4, 4))

    C = np.einsum('ij,kl->ikjl', A, B)
    X = C.reshape([2, ]*20)
    Y = X.transpose([2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14,
                    15, 16, 17, 18, 19, 0, 1, 10, 11])
    Z = Y.reshape([256, 256, 4, 4])
    X_base = np.trace(Z)

    assert np.allclose(X_base, X_obj)

    X_simple = einsum_simple(
        'ik,jl',
        [2, ]*20,
        'abcde fghij klmno pqrst->cde fghij mno pqrst ab kl',
        [256, 256, 4, 4],
        'ii...',
        A, B
    )

    assert np.allclose(X_simple, X_obj)

    assert np.allclose(np.einsum('a...,...->a...', A, B),
                       np.einsum('ab,cb->acb', A, B))

    assert np.allclose(np.einsum('...,...c->...c', A, B),
                       np.einsum('ab,bc->abc', A, B))

    assert np.allclose(np.einsum('...,...c', A, B),
                       np.einsum('ab,bc->abc', A, B))

    X_advanced = einsum_pipe(
        'ik,jl',
        [2, ]*20,
        'abcde fghij klmno pqrst->cde fghij mno pqrst ab kl',
        [256, 256, 4, 4],
        'ii...->...',
        A, B
    )

    assert np.allclose(X_advanced, X_simple)