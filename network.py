from abc import ABC, abstractmethod
from typing import List, Tuple, NamedTuple, MutableMapping, Optional

import torch

from config import Config
from forward_backward import ForwardBackward
from utils import Step, Option, Agent


def add_logprob(logprobs: List[torch.Tensor], prob: torch.Tensor) -> None:
    # logprob = (prob + Config.epsilon).log()
    logprob = prob.log()
    assert torch.isnan(logprob).sum().item() == 0.
    logprobs.append(logprob)


class DDOLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, trajectory: List[Step], agent: Agent) -> torch.Tensor:
        fb = ForwardBackward(agent, trajectory)
        logprobs = []
        for step_idx, step in enumerate(trajectory):
            meta = agent.meta(step.current_obs)
            for opt_idx, option in enumerate(agent.options):
                action = option.policy(step.current_obs)[torch.arange(Config.batch_size), step.current_action]
                termination = option.termination(step.current_obs)
                is_option_factor = fb.is_option_factor(opt_idx, step_idx)
                has_switch_to_option_factor = fb.has_switch_to_option_factor(opt_idx, step_idx)
                add_logprob(logprobs, has_switch_to_option_factor * meta[:, opt_idx])
                add_logprob(logprobs, is_option_factor * action)
                if step_idx < len(trajectory) - 1:
                    next_step = trajectory[step_idx + 1]
                    termination = option.termination(next_step.current_obs)
                    option_will_continue_factor = fb.option_will_continue_factor(opt_idx, step_idx)
                    option_will_terminate_factor = fb.option_will_terminate_factor(opt_idx, step_idx)
                    add_logprob(logprobs, option_will_terminate_factor * termination)
                    add_logprob(logprobs, option_will_continue_factor * (1. - termination))
        return -torch.concat(logprobs).mean()


class TaxiMetaNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(Config.taxi_ninputs, Config.taxi_hidden_layer),
            torch.nn.Tanh(),
            torch.nn.Linear(Config.taxi_hidden_layer, Config.noptions),
            torch.nn.Softmax(dim=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=Config.taxi_init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=Config.taxi_init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        return outputs


class TaxiPolicyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(Config.taxi_ninputs, Config.taxi_nactions),
            torch.nn.Softmax(dim=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=Config.taxi_init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=Config.taxi_init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        return outputs


class TaxiTerminationNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(Config.taxi_ninputs, 1),
            torch.nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=Config.taxi_init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=Config.taxi_init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs).reshape([-1])
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

    def select_option(self, obs: torch.Tensor) -> None:
        previous = self.current_option
        self.current_option = self.meta(obs)[0].argmax().int().item()
        self.meta_prob = self.meta(obs)[0][self.current_option]
        self.option_tracker[self.current_option] += 1
        if self.previous_option != self.current_option:
            self.option_change_tracker[self.current_option] += 1
        self.previous_option = self.current_option

    def action(self, obs: torch.Tensor) -> int:
        assert obs.shape[0] == 1
        if self.current_option < 0:
            self.select_option(obs)
        option = self.options[self.current_option]
        action = option.policy(obs)[0].argmax().int().item()
        self.action_prob = option.policy(obs)[0][action]
        self.termination_prob = option.termination(obs)[0].item()
        if self.termination_prob > 0.5:
            self.current_option = -1
        return action
