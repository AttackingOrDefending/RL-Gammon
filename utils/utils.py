"""TODO"""
import pickle

from rlgammon.environment.backgammon_env import BackgammonEnv


def copy(env: BackgammonEnv) -> BackgammonEnv:
    """
    Return a deepcopy of the provided environment.

    :param env: the environment to be copied
    :return: the created copy of the environment
    """
    return pickle.loads(pickle.dumps(env, -1))
