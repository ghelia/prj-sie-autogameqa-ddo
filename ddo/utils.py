from abc import ABC, abstractmethod
from typing import List, Tuple, NamedTuple, MutableMapping, Optional

import torch

from .config import Config


class Step(NamedTuple):
    current_obs: torch.Tensor
    current_action: Optional[torch.Tensor]


class Option(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module, termination: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.termination = termination


class Agent(torch.nn.Module):
    def __init__(self, meta: torch.nn.Module, options: List[Option]) -> None:
        super().__init__()
        self.meta = meta
        self.options = torch.nn.Sequential(*options)

class EvalMetric(torch.nn.Module):
    @abstractmethod
    def eval_agent(self, agent: Agent) -> float:
        raise NotImplementedError
