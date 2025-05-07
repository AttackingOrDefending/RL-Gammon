"""File with various utility functions used in the project."""
import pickle
from typing import Any


def copy(item: Any) -> Any:  # noqa: ANN401
    """
    Return a deepcopy of the provided environment.

    :param item: item to be copied
    :return: the created copy of the item
    """
    return pickle.loads(pickle.dumps(item, -1))

