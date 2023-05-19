from __future__ import annotations
import copy
import math
from typing import Generator, List, Optional, Tuple, TypeVar, Union, cast
from .bidict import _BiDict


class EinsumComp:
    def __init__(self, size: int) -> None:
        self.size = size


class NullTag:
    pass


def _get_char(index: int) -> str:
    assert index < 26*2
    return chr((ord('a') if index < 26 else (ord('A') - 26)) + index)


class EinsumScript:
    def __init__(self, inputs: List[List[EinsumComp]], outputs: List[EinsumComp]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self._parsed_script: Optional[str] = None

    @classmethod
    def parse(cls, input_shapes: List[List[int]], subscripts: str) -> EinsumScript:
        parsed_subscripts = subscripts
        subscripts = subscripts.replace(' ', '')
        # Easier to deal with broadcasting as a single character
        subscripts = subscripts.replace('...', '?')
        # The broadcasting character is automatically sorted to the start
        letters = sorted(subscripts.replace(',', '').replace('->', ''))
        if '->' not in subscripts:
            output_letters = [l for l in letters if l ==
                              '?' or letters.count(l) == 1]
            if (bc_count := output_letters.count('?')) > 1:
                output_letters = output_letters[bc_count - 1:]
            subscripts += '->' + ''.join(output_letters)
        letter_dict = {v: EinsumComp(0) for v in set(letters) if v != '?'}

        inputs_subs, output_subs = subscripts.split('->')
        inputs: List[List[EinsumComp]] = []
        broadcast_comps: List[EinsumComp] = []
        assert len(inputs_subs.split(',')) == len(
            input_shapes), f'''Error while parsing "{subscripts}" with input shapes {input_shapes}: insuficient input shapes found for number of references in subscripts!'''
        for sub, shape in zip(inputs_subs.split(','), input_shapes):
            inputs.append([])
            for c in sub:
                if c == '?':
                    # Broadcasting works from the last axis to the first and shares these axes with other broadcasts
                    undefined_axes = len(shape) - (len(sub) - 1)
                    for _ in range(undefined_axes - len(broadcast_comps)):
                        broadcast_comps.insert(0, EinsumComp(0))
                    if undefined_axes > 0:
                        inputs[-1].extend(broadcast_comps[-undefined_axes:])
                else:
                    inputs[-1].append(letter_dict[c])

        assert len(inputs) > 0

        outputs: List[EinsumComp] = []
        for c in output_subs:
            if c == '?':
                # All broadcasted axes are added in order
                outputs.extend(broadcast_comps)
            else:
                outputs.append(letter_dict[c])

        script = EinsumScript(inputs, outputs)
        script._parsed_script = parsed_subscripts
        for inp, shape in zip(inputs, input_shapes):
            assert len(inp) == len(
                shape), f'Error while parsing "{subscripts}" with input shapes {input_shapes}: {len(inp)} != {len(shape)}'
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

    @property
    def input_shapes(self) -> List[Tuple[int]]:
        return [tuple(comp.size for comp in inp) for inp in self.inputs]

    @property
    def output_shape(self) -> Tuple[int]:
        return tuple(comp.size for comp in self.outputs)

    def simplify(self, keep_shapes: Optional[List[Tuple[int, ...]]] = None):
        # Get sequences (repeated or not) in which each element is unique to the sequence
        # Basically, run through it as something like a linked list
        next_map: _BiDict[Union[NullTag, EinsumComp],
                          Union[NullTag, EinsumComp]] = _BiDict()

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

        # Replace sequences of comps with their respective group comp
        for comps in [*self.inputs, self.outputs]:
            for group, new_comp in group_pairs:
                while group[0] in comps:
                    i = comps.index(group[0])
                    comps[i] = new_comp
                    for _ in range(len(group) - 1):
                        comps.pop(i + 1)

        if keep_shapes is not None:
            for inp, input_shape in zip(self.inputs, keep_shapes):
                input_shape_iter = iter(input_shape[::-1])
                inp_in_iter = rev_mut_iter(inp)

                try:
                    input_shape_val = next(input_shape_iter)
                    inp_in_val = next(inp_in_iter)

                    while True:
                        if input_shape_val == inp_in_val.size:
                            input_shape_val = next(input_shape_iter)
                            inp_in_val = next(inp_in_iter)
                        elif input_shape_val > inp_in_val.size:
                            input_shape_val //= inp_in_val.size
                            inp_in_val = next(inp_in_iter)
                        else:
                            self.split_comp(inp_in_val, [
                                inp_in_val.size // input_shape_val, input_shape_val])
                            input_shape_val = next(input_shape_iter)
                except StopIteration:
                    pass

    def simplified(self, keep_shape: Optional[List[Tuple[int, ...]]] = None) -> EinsumScript:
        val = copy.deepcopy(self)
        val.simplify(keep_shape)
        return val

    def __repr__(self) -> str:
        if self._parsed_script is None:
            return f'"{self}"'
        else:
            return f'"{self}" (parsed from "{self._parsed_script}")'

    def __str__(self) -> str:
        comps = list(set(comp for inp in self.inputs for comp in inp))

        subs = []
        for inp in self.inputs:
            subs.append(''.join(_get_char(comps.index(comp)) for comp in inp))

        output_str = ''.join(_get_char(comps.index(comp))
                             for comp in self.outputs)

        return ','.join(subs) + '->' + output_str

    def __add__(self, rhs: EinsumScript) -> EinsumScript:
        lhs = copy.deepcopy(self)
        rhs = copy.deepcopy(rhs)
        lhs_out_iter = rev_mut_iter(lhs.outputs)
        rhs_in_iter = rev_mut_iter(rhs.inputs[0])

        try:
            lhs_out_val = next(lhs_out_iter)
            rhs_in_val = next(rhs_in_iter)

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

        rhs.remove_ones()
        lhs.remove_ones()

        assert len(lhs.outputs) == len(
            rhs.inputs[0]), f'Incompatible shapes between {repr(lhs)} and {repr(rhs)}: {lhs.output_shape} and {rhs.input_shapes[0]}'
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
