# Einsum Pipe

A Python package to compile multiple Numpy [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) operations into one.

## Installation

The package is available on PyPi:

```
pip install einsum_pipe
```

## Example

Given two arrays:
```python
A = np.random.rand(32, 32, 10, 5)
B = np.random.rand(32, 32, 10, 5)
```

We frequently need to run multiple reshape/transpose/products/trace/etc., such as:
```python
C = np.einsum('ij...,kl...->ikjl...', A, B)
D = C.reshape([2, ]*20 + [10, 5])
E = D.transpose([2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14,
                15, 16, 17, 18, 19, 0, 1, 10, 11, 20, 21])
F = E.reshape([256, 256, 4, 4, 10, 5])
X = np.trace(F)
```

This obviously results in multiple intermediate arrays, some of which can be large. Instead of doing this, it is possible to combine multiple `np.einsum` operations into one. By carefully modifying the input shape, it is even possible to do this in cases in which the intermediate data is reshaped during the process, provided the shapes are all [compatible](#shape-compatibility). The previous example can instead be performed in a single `np.einsum` step:
```python
X = einsum_pipe(
    'ik...,jl...->ijkl...',
    [2, ]*20 + [10, 5],
    'abcde fghij klmno pqrst...->cde fghij mno pqrst ab kl...',
    [256, 256, 4, 4, 10, 5],
    'ii...',
    A, B
)
```

Internally, this calculates a compatible input shape, `(4, 8, 4, 8, 50)` and `(32, 32, 50)`, and a combined `np.einsum` set of subscripts, `"ebdbc,aac->edc"`. `A` and `B` are reshaped (which is frequently free), the single `np.einsum` operation is run, and the output is reshaped back to the expected output shape.

You can find further examples in the "tests" folder.

## Syntax

The syntax is based on Numpy's `einsum`, with the addition of allowing multiple subscripts and defining the shapes of the intermediate arrays. The input arrays can be put at the end, as shown, or next to the subscript definitions. In this example, only two arrays are used at start of the pipe, however you can add more arrays at later stages. The output of the previous step is always considered the first input of the subsequent step.

## Shape Compatibility

Shapes are compatible if each dimension is the product of some subsequence of a matching shape (of the previous output). For example, `(32, 32)` and `(4, 256)` are compatible, since both can be built from the shape `(4, 8, 4, 8)`: `(4*8, 4*8)` and `(4, 8*4*8)`. On the other hand, `(2, 3)` and `(3, 2)` aren't directly compatible since they don't share divisors.

Note that transposition of axes also causes the transposition of the compatible shape, so while `[(3, 2), 'ij->ij', (2, 3)]` isn't valid, `[(3, 2), 'ij->ji', (2, 3)]` is.

I plan to implement a "best effort" fallback which would reduce a sequence of operations to as few operations as possible, depending on incompatible shapes.

## Subscript Simplification

In order to merge multiple subscript steps with different intermediate shapes, the input arrays must be reshaped to be compatible with all steps. However, after merging multiple subscripts, certain complex shapes may be eliminated. While it makes no difference to the performance of the operations, the actual subscript string passed to `np.einsum` can be unnecessarily long. This may even be an issue if there are more axes than available letters.

`einsum_pipe` includes the `simplify` argument to deal with such cases. This can be set to `False` to disable simplification or `"max"` to reduce the length of the subscripts as much as possible. However, this isn't always advisable as merging smaller axes into a larger axis can force an array copy during the initial `reshape` if the input array has been transposed (more on that [here](https://stackoverflow.com/a/60389152/11057932)). Splitting an axis should never cause a problem. The default argument (`True`) simplifies the subscript as much as possible while maintaining the splits from the original input arrays. If your inputs are contiguous, you can safely use `"max"`.

## Numpy Operations

Numpy's documentation on [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) lists some operations that can be implemented using `np.einsum`. Some of these have been implemented here in the `ops` submodule. These are just convenience functions to generate the correct subscripts for `np.einsum`, they generally produce a string. They can be used as part of `einsum_pipe` operations:
```python
from einsum_pipe import ops

X = einsum_pipe(
    ops.inner(),
    ops.transpose((1, 0))
    ops.diag(),
    'a->'
    A, B
)
```

More operations may be added in future. As part of this implementation, `einsum_pipe` also supports "lazy" arguments: functions passed as arguments which will be called during parsing with the list of available input shapes, to then produce the subscript string or a reshape operation. Note this is still run during "compilation", not when running with `np.einsum`.
