"""File with various utility functions used in the project."""
import pickle
from typing import Any

from rlgammon.environment.backgammon_env import BackgammonEnv


def copy(item: Any) -> Any:  # noqa: ANN401
    """
    Return a deepcopy of the provided environment.

    :param item: item to be copied
    :return: the created copy of the item
    """
    return pickle.loads(pickle.dumps(item, -1))


def interleave_lists(list1: list[Any], list2: list[Any]) -> list[Any]:
    """
    TODO.

    :param list1:
    :param list2:
    :return:
    """
    new_list: list[Any] = []
    for i in range(len(list1)):
        new_list.append(list1[i])
        if i < len(list2):
            new_list.append(list2[i])
    new_list.extend(list1[len(list1):])
    return new_list
