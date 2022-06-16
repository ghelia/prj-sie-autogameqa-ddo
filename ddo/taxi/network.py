from typing import List, NamedTuple, MutableMapping, Optional

import torch
import numpy as np

from ..config import Config
from .config import TaxiConfig
from ..utils import Option, Agent
from ..network import Dense


class TaxiNetworkBase(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=TaxiConfig.init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=TaxiConfig.init_std)


class TaxiMetaNetwork(TaxiNetworkBase):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(
            [TaxiConfig.ninputs] + TaxiConfig.hidden_layer + [Config.noptions],
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        return outputs


class TaxiPolicyNetwork(TaxiNetworkBase):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(
            [TaxiConfig.ninputs] + TaxiConfig.hidden_layer + [TaxiConfig.nactions],
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        return outputs


class TaxiTerminationNetwork(TaxiNetworkBase):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(
            [TaxiConfig.ninputs] + TaxiConfig.hidden_layer + [1],
            torch.nn.Tanh(),
            torch.nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs).reshape([-1])
        return outputs


class TaxiAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            TaxiMetaNetwork(),
            [Option(TaxiPolicyNetwork(), TaxiTerminationNetwork()) for _ in range(Config.noptions)]
        )

class DebugPolicyNetwork(TaxiNetworkBase):
    def __init__(self, index: int) -> None:
        self.index = index
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros([Config.batch_size, TaxiConfig.nactions])
        outputs[:,self.index] = 10.
        return outputs.softmax(1)


class DebugAgent(TaxiAgent):
    def __init__(self) -> None:
        super().__init__()
        assert Config.noptions == TaxiConfig.nactions
        self.options = torch.nn.Sequential(
            *[Option(DebugPolicyNetwork(idx), TaxiTerminationNetwork()) for idx in range(TaxiConfig.nactions)]
        )
