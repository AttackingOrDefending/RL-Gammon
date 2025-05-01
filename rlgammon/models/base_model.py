"""TODO."""
from abc import abstractmethod
from datetime import datetime
import random

import torch as th
from torch import nn

from rlgammon.rlgammon_types import State


class BaseModel(nn.Module):
    """TODO."""

    def __init__(self, lr: float, lamda: float, seed: int=123) -> None:
        """
        TODO.

        :param lr:
        :param lamda:
        :param seed:
        """
        super(BaseModel).__init__()
        self.lr = lr
        self.lamda = lamda  # trace-decay parameter
        self.start_episode = 0

        self.eligibility_traces = None
        self.optimizer = None

        th.manual_seed(seed)
        random.seed(seed)

    def forward(self, x: State) -> th.Tensor:
        """
        TODO.

        :param x:
        :return:
        """
        pass

    @abstractmethod
    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """
        TODO>.

        :param p:
        :param p_next:
        :return:
        """
        raise NotImplementedError

    def checkpoint(self, checkpoint_path: str, step: int, name_experiment: str):
        """
        TODO.

        :param checkpoint_path:
        :param step:
        :param name_experiment:
        :return:
        """
        path = checkpoint_path + "/{}_{}_{}.tar".format(name_experiment, datetime.now().strftime("%Y%m%d_%H%M_%S_%f"), step + 1)
        th.save({"step": step + 1, "model_state_dict": self.state_dict(), "eligibility": self.eligibility_traces if self.eligibility_traces else []}, path)
        print(f"\nCheckpoint saved: {path}")

    def load(self, checkpoint_path: str, optimizer=None, eligibility_traces=None):
        """
        TODO.

        :param checkpoint_path:
        :param optimizer:
        :param eligibility_traces:
        :return:
        """
        checkpoint = th.load(checkpoint_path)
        self.start_episode = checkpoint["step"]

        self.load_state_dict(checkpoint["model_state_dict"])

        if eligibility_traces is not None:
            self.eligibility_traces = checkpoint["eligibility"]

        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
