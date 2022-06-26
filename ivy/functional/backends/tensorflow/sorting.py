# global
import tensorflow as tf
from typing import Union


def argsort(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Union[tf.Tensor, tf.Variable]:
    if tf.convert_to_tensor(x).dtype.is_bool:
        return (
            tf.argsort(
                tf.cast(x, dtype=tf.int32),
                axis=axis,
                direction="DESCENDING",
                stable=stable,
            )
            if descending
            else tf.argsort(
                tf.cast(x, dtype=tf.int32),
                axis=axis,
                direction="ASCENDING",
                stable=stable,
            )
        )

    elif descending:
        return tf.argsort(
            tf.convert_to_tensor(x),
            axis=axis,
            direction="DESCENDING",
            stable=stable,
        )

    else:
        return tf.argsort(
            tf.convert_to_tensor(x),
            axis=axis,
            direction="ASCENDING",
            stable=stable,
        )


def sort(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Union[tf.Tensor, tf.Variable]:
    if tf.convert_to_tensor(x).dtype.is_bool:
        res = (
            tf.sort(
                tf.cast(x, dtype=tf.int32), axis=axis, direction="DESCENDING"
            )
            if descending
            else tf.sort(
                tf.cast(x, dtype=tf.int32), axis=axis, direction="ASCENDING"
            )
        )

        return tf.cast(res, tf.bool)
    elif descending:
        return tf.sort(tf.convert_to_tensor(x), axis=axis, direction="DESCENDING")
    else:
        return tf.sort(tf.convert_to_tensor(x), axis=axis, direction="ASCENDING")
