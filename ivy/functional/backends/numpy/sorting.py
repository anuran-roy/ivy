# global
import numpy as np


def argsort(
    x: np.ndarray, axis: int = -1, descending: bool = False, stable: bool = True
) -> np.ndarray:
    return (
        np.asarray(
            np.argsort(
                -1 * np.searchsorted(np.unique(x), x), axis, kind="stable"
            )
        )
        if descending
        else np.asarray(np.argsort(x, axis, kind="stable"))
    )


def sort(
    x: np.ndarray, axis: int = -1, descending: bool = False, stable: bool = True
) -> np.ndarray:
    kind = "stable" if stable else "quicksort"
    ret = np.asarray(np.sort(x, axis=axis, kind=kind))
    if descending:
        ret = np.asarray((np.flip(ret, axis)))
    return ret
