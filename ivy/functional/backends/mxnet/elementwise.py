# global
import mxnet as mx
import math
from typing import Union

# local
from ivy.functional.backends.mxnet import (
    _handle_flat_arrays_in_out,
    _scalar_or_flat_array_to_scalar,
)


@_handle_flat_arrays_in_out
def add(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.add(x1, x2)


@_handle_flat_arrays_in_out
def bitwise_and(
    x1: Union[int, mx.nd.NDArray],
    x2: Union[int, mx.nd.NDArray],
) -> mx.nd.ndarray.NDArray:
    return mx.numpy.bitwise_and(x1, x2)


@_handle_flat_arrays_in_out
def ceil(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.ceil(x)


@_handle_flat_arrays_in_out
def floor(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.floor(x)


@_handle_flat_arrays_in_out
def divide(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.divide(x1, x2)


@_handle_flat_arrays_in_out
def greater(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.greater(x1, x2)


@_handle_flat_arrays_in_out
def greater_equal(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.greater_equal(x1, x2)


@_handle_flat_arrays_in_out
def isfinite(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.contrib.isfinite(x.astype("float32")).astype("bool")


@_handle_flat_arrays_in_out
def isinf(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.contrib.isinf(x.astype("float32")).astype("bool")


def sqrt(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.sqrt(x) if isinstance(x, float) else mx.nd.sqrt(x)


@_handle_flat_arrays_in_out
def isnan(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.contrib.isnan(x).astype("bool")


@_handle_flat_arrays_in_out
def less(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.lesser(x1, x2).astype("bool")


@_handle_flat_arrays_in_out
def logical_xor(x1: mx.nd.NDArray, x2: mx.nd.NDArray, dtype: ["bool"]) -> mx.nd.NDArray:
    return mx.nd.logical_xor(x1, x2, dtype).astype("bool")


@_handle_flat_arrays_in_out
def logical_not(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.logical_not(x)


@_handle_flat_arrays_in_out
def acos(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.acos(x) if isinstance(x, float) else mx.nd.arccos(x)


@_handle_flat_arrays_in_out
def logical_and(x1: mx.nd.NDArray, x2: mx.nd.NDArray, dtype: ["bool"]) -> mx.nd.NDArray:
    return mx.nd.logical_and(x1, x2, dtype).astype("bool")


@_handle_flat_arrays_in_out
def logical_or(x1: mx.nd.NDArray, x2: mx.nd.NDArray, dtype: ["bool"]) -> mx.nd.NDArray:
    return mx.nd.logical_or(x1, x2, dtype).astype("bool")


@_handle_flat_arrays_in_out
def multiply(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.multiply(x1, x2)


@_handle_flat_arrays_in_out
def acosh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.acosh(x) if isinstance(x, float) else mx.nd.arccosh(x)


@_handle_flat_arrays_in_out
def sin(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.sin(x) if isinstance(x, float) else mx.nd.sin(x)


@_handle_flat_arrays_in_out
def negative(x: Union[float, mx.nd.NDArray]) -> mx.nd.NDArray:
    return mx.np.negative(x)


@_handle_flat_arrays_in_out
def tanh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.tanh(x) if isinstance(x, float) else mx.nd.tanh(x)


@_handle_flat_arrays_in_out
def bitwise_or(
    x1: Union[int, mx.nd.NDArray],
    x2: Union[int, mx.nd.NDArray],
) -> mx.nd.ndarray.NDArray:
    return mx.numpy.bitwise_or(x1, x2)


@_handle_flat_arrays_in_out
def sinh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.sinh(x) if isinstance(x, float) else mx.nd.sinh(x)


@_handle_flat_arrays_in_out
def square(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.square(x)


@_handle_flat_arrays_in_out
def round(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.round(x)


@_handle_flat_arrays_in_out
def trunc(x: mx.nd.NDArray) -> mx.nd.ndarray.NDArray:
    return mx.np.trunc(x)


@_handle_flat_arrays_in_out
def subtract(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.subtract(x1, x2)


@_handle_flat_arrays_in_out
def abs(x: Union[float, mx.nd.NDArray]) -> mx.nd.ndarray.NDArray:
    return mx.nd.abs(x)


def cos(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.cos(x) if isinstance(x, float) else mx.nd.cos(x)


def exp(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return math.exp(x) if isinstance(x, float) else mx.nd.exp(x)


tan = lambda x: math.tan(x) if isinstance(x, float) else mx.nd.tan(x)
asin = lambda x: math.asin(x) if isinstance(x, float) else mx.nd.arcsin(x)
atan = lambda x: math.atan(x) if isinstance(x, float) else mx.nd.arctan(x)
atan2 = (
    lambda x, y: math.atan2(x, y)
    if isinstance(x, float)
    else mx.np.arctan2(x.as_np_ndarray(), y.as_np_ndarray()).as_nd_ndarray()
)
cosh = lambda x: math.cosh(x) if isinstance(x, float) else mx.nd.cosh(x)
asinh = lambda x: math.asinh(x) if isinstance(x, float) else mx.nd.arcsinh(x)
atanh = lambda x: math.atanh(x) if isinstance(x, float) else mx.nd.arctanh(x)
log = lambda x: math.log(x) if isinstance(x, float) else mx.nd.log(x)
equal = lambda x1, x2: x1 == x2
equal.__name__ = "equal"

# Extra #
# ------#


minimum = lambda x, y: mx.nd.array(
    mx.nd.minimum(
        _scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)
    )
)
maximum = lambda x, y: mx.nd.array(
    mx.nd.maximum(
        _scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)
    )
)
erf = lambda x: math.erf(x) if isinstance(x, float) else mx.nd.erf(x)
