# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def argsort(
    x: JaxArray, axis: int = -1, descending: bool = False, stable: bool = True
) -> JaxArray:
    return (
        jnp.asarray(
            jnp.argsort(
                -1 * jnp.searchsorted(jnp.unique(x), x), axis, kind="stable"
            )
        )
        if descending
        else jnp.asarray(jnp.argsort(x, axis, kind="stable"))
    )


def sort(
    x: JaxArray, axis: int = -1, descending: bool = False, stable: bool = True
) -> JaxArray:
    kind = "stable" if stable else "quicksort"
    res = jnp.asarray(jnp.sort(x, axis=axis, kind=kind))
    return jnp.asarray(jnp.flip(res, axis=axis)) if descending else res
