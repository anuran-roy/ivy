# global
import jax.numpy as jnp
from typing import Union, Optional, Tuple, Literal, List, NamedTuple
from collections import namedtuple

# local
import ivy
from ivy import inf
from ivy.functional.backends.jax import JaxArray


# Array API Standard #
# -------------------#


def cholesky(x: JaxArray, upper: bool = False) -> JaxArray:
    if not upper:
        return jnp.linalg.cholesky(x)
    axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
    return jnp.transpose(
        jnp.linalg.cholesky(jnp.transpose(x, axes=axes)), axes=axes
    )


def cross(x1: JaxArray, x2: JaxArray, axis: int = -1) -> JaxArray:
    return jnp.cross(a=x1, b=x2, axis=axis)


def det(x: JaxArray) -> JaxArray:
    return jnp.linalg.det(x)


def diagonal(
    x: JaxArray,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
) -> JaxArray:
    if x.dtype != bool and not jnp.issubdtype(x.dtype, jnp.integer):
        ret = jnp.diagonal(x, offset, axis1, axis2)
        ret_edited = jnp.diagonal(
            x.at[1 / x == -jnp.inf].set(-jnp.inf), offset, axis1, axis2
        )
        ret_edited = ret_edited.at[ret_edited == -jnp.inf].set(-0.0)
        ret = ret.at[ret == ret_edited].set(ret_edited[ret == ret_edited])
    else:
        ret = jnp.diagonal(x, offset, axis1, axis2)
    return ret


def eigh(x: JaxArray) -> JaxArray:
    return jnp.linalg.eigh(x)


def eigvalsh(x: JaxArray) -> JaxArray:
    return jnp.linalg.eigvalsh(x)


def inv(x: JaxArray) -> JaxArray:
    return (
        x
        if jnp.any(jnp.linalg.det(x.astype("float64")) == 0)
        else jnp.linalg.inv(x)
    )


def matmul(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.matmul(x1, x2)


def matrix_norm(
    x: JaxArray,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
) -> JaxArray:
    if x.size == 0:
        return (
            x.reshape(x.shape[:-2] + (1, 1))
            if keepdims
            else x.reshape(x.shape[:-2])
        )

    else:
        return jnp.linalg.norm(x, ord, (-2, -1), keepdims)


def matrix_power(x: JaxArray, n: int) -> JaxArray:
    return jnp.linalg.matrix_power(x, n)


def matrix_rank(
    x: JaxArray,
    rtol: Optional[Union[float, Tuple[float]]] = None,
) -> JaxArray:
    if x.size == 0:
        return 0
    elif x.size == 1:
        return jnp.count_nonzero(x)
    else:
        if x.ndim > 2:
            x = x.reshape([-1])
        return jnp.linalg.matrix_rank(x, rtol)


def matrix_transpose(x: JaxArray) -> JaxArray:
    return jnp.swapaxes(x, -1, -2)


def outer(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.outer(x1, x2)


def pinv(x: JaxArray, rtol: Optional[Union[float, Tuple[float]]] = None) -> JaxArray:
    return jnp.linalg.pinv(x) if rtol is None else jnp.linalg.pinv(x, rtol)


def qr(x: JaxArray, mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = jnp.linalg.qr(x, mode=mode)
    return res(q, r)


def slogdet(
    x: Union[ivy.Array, ivy.NativeArray]
) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = jnp.linalg.slogdet(x)
    return results(sign, logabsdet)


def solve(x1: JaxArray, x2: JaxArray) -> JaxArray:
    expanded_last = False
    if len(x2.shape) <= 1 and x2.shape[-1] == x1.shape[-1]:
        expanded_last = True
        x2 = jnp.expand_dims(x2, axis=1)

    # if any of the arrays are empty
    is_empty_x1 = x1.size == 0
    is_empty_x2 = x2.size == 0
    if is_empty_x1 or is_empty_x2:
        for _ in range(len(x1.shape) - 2):
            x2 = jnp.expand_dims(x2, axis=0)
        output_shape = list(jnp.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        output_shape.append(x2.shape[-2])
        output_shape.append(x2.shape[-1])
        ret = jnp.array([]).reshape(output_shape)
    else:
        output_shape = tuple(jnp.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        x1 = jnp.broadcast_to(x1, output_shape + x1.shape[-2:])
        x2 = jnp.broadcast_to(x2, output_shape + x2.shape[-2:])
        ret = jnp.linalg.solve(x1, x2)

    if expanded_last:
        ret = jnp.squeeze(ret, axis=-1)
    return ret


def svd(
    x: JaxArray, full_matrices: bool = True
) -> Union[JaxArray, Tuple[JaxArray, ...]]:
    results = namedtuple("svd", "U S Vh")
    U, D, VT = jnp.linalg.svd(x, full_matrices=full_matrices)
    return results(U, D, VT)


def svdvals(x: JaxArray) -> JaxArray:
    return jnp.linalg.svd(x, compute_uv=False)


def tensordot(
    x1: JaxArray,
    x2: JaxArray,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
) -> JaxArray:
    return jnp.tensordot(x1, x2, axes)


def trace(x: JaxArray, offset: int = 0) -> JaxArray:
    return jnp.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=x.dtype)


def vecdot(x1: JaxArray, x2: JaxArray, axis: int = -1) -> JaxArray:
    return jnp.tensordot(x1, x2, (axis, axis))


def vector_norm(
    x: JaxArray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
) -> JaxArray:
    if axis is None:
        jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), ord, axis, keepdims)
    else:
        jnp_normalized_vector = jnp.linalg.norm(x, ord, axis, keepdims)

    return (
        jnp.expand_dims(jnp_normalized_vector, 0)
        if jnp_normalized_vector.shape == ()
        else jnp_normalized_vector
    )


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(vector: JaxArray) -> JaxArray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = jnp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = jnp.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = jnp.concatenate((zs, -a3s, a2s), -1)
    row2 = jnp.concatenate((a3s, zs, -a1s), -1)
    row3 = jnp.concatenate((-a2s, a1s, zs), -1)
    return jnp.concatenate((row1, row2, row3), -2)
