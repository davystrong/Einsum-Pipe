import copy
import math
from typing import Generator, List, Self, Tuple, TypeVar, Union, cast
from .bidict import BiDict


class EinsumComp:
    def __init__(self, size: int) -> None:
        self.size = size


class NullTag:
    pass


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