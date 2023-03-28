from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(ABC):
    def __init__(self, lmbda: float):
        """
        Base class of regularizers

        Args:
            lmbda (float): Penalty coefficient of a regularizer
        """
        self.lmbda = lmbda

    @abstractmethod
    def penalty(self, factors: Tuple[torch.Tensor]):
        pass

    def checkpoint(self, regularizer_cache_path: str, epoch_id: int):
        if regularizer_cache_path is not None:
            print(f"Save the regularizer at epoch {epoch_id}")
            path = regularizer_cache_path + f"{epoch_id}.reg"
            torch.save(self.state_dict(), path)
            print(f"Regularizer Checkpoint: {path}")


class F2(Regularizer):
    def __init__(self, lmbda: float):
        """
        F2/L2 regularizer

        Args:
            lmbda (float): Penalty coefficient
        """
        super().__init__(lmbda)

    def penalty(self, factors: Tuple[torch.Tensor]) -> float:
        """
        Args:
            factors (Tuple[torch.Tensor]): Model factors

        Returns:
            float: regularization loss
        """
        norm = 0
        for factor in factors:
            norm += self.lmbda * torch.sum(factor**2)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, lmbda: float):
        """
        N3 regularizer http://arxiv.org/abs/1806.07297

        Args:
            lmbda (float): Penalty coefficient
        """
        super().__init__(lmbda)

    def penalty(self, factors: Tuple[torch.Tensor]):
        """
        Args:
            factors (Tuple[torch.Tensor]): Model factors

        Returns:
            float: regularization loss
        """
        norm = 0
        for factor in factors:
            norm += self.lmbda * torch.sum(torch.abs(factor) ** 3)
        return norm / factors[0].shape[0]
