import torch
from abc import ABC, abstractmethod

class IntrinsicMotivationModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def surprisal(self, *args, **kwargs):
        return torch.zeros(10)