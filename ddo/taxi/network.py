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
        self.previous_option = -1
        self.current_option = -1
        self.option_tracker = [0 for _ in range(Config.noptions)]
        self.option_change_tracker = [0 for _ in range(Config.noptions)]
        self.action_prob = 0.
        self.termination_prob = 0.
        self.meta_prob = 0.

    def reset(self) -> None:
        self.current_option = -1
        self.option_tracker = [0 for _ in range(Config.noptions)]
        self.option_change_tracker = [0 for _ in range(Config.noptions)]

    def select_option(self, obs: torch.Tensor, greedy: bool = False, force_option: Optional[int] = None) -> None:
        previous = self.current_option

        if greedy:
            self.current_option = self.meta(obs)[0].argmax().int().item()
        else:
            selection_distribution = torch.distributions.categorical.Categorical(self.meta(obs))
            self.current_option = selection_distribution.sample()[0].int().item()
        if force_option is not None:
            self.current_option = 1

        self.meta_prob = self.meta(obs)[0][self.current_option]
        self.option_tracker[self.current_option] += 1
        if self.previous_option != self.current_option:
            self.option_change_tracker[self.current_option] += 1
        self.previous_option = self.current_option

    def action(self, obs: torch.Tensor, greedy: bool = False, only_option: Optional[int] = None) -> int:
        assert obs.shape[0] == 1
        if self.current_option < 0:
            self.select_option(obs, greedy, only_option)
        option = self.options[self.current_option]
        self.termination_prob = option.termination(obs)[0].item()
        if greedy:
            if self.termination_prob > 0.5:
                self.select_option(obs, greedy, only_option)
        else:
            if np.random.random() < self.termination_prob:
                self.select_option(obs, greedy, only_option)

        if greedy:
            action = option.policy(obs)[0].argmax().int().item()
        else:
            selection_distribution = torch.distributions.categorical.Categorical(option.policy(obs))
            action = selection_distribution.sample()[0].int().item()

        self.action_prob = option.policy(obs)[0][action]
        return action

    def all_probs(self, obs: torch.Tensor) -> torch.Tensor:
        probs = []
        for option in self.options:
            prob = option.policy(obs)[0]
            probs.append(prob)
        return torch.stack(probs)


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
