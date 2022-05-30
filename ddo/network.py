from abc import ABC, abstractmethod
from typing import List, Tuple, NamedTuple, MutableMapping, Optional

import torch
import numpy as np

from .config import Config
from .forward_backward import ForwardBackward
from .utils import Step, Option, Agent


def add_logprob(logprobs: List[torch.Tensor],
                prob: torch.Tensor,
                weight: torch.Tensor
               ) -> None:
    logprob = (prob + Config.epsilon).log() * weight
    # logprob = prob.log() * weight
    assert torch.isnan(logprob).sum().item() == 0.
    logprobs.append(logprob)


class DDOLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.KL = torch.nn.KLDivLoss(reduction="batchmean")

    def kl_divergence(self, all_actions_probs: List[torch.Tensor]) -> torch.Tensor:
        divs = []
        noptions = len(all_actions_probs)
        assert noptions == Config.noptions
        for i in range(noptions):
            for j in range(noptions):
                prob1 = all_actions_probs[i]
                prob2 = all_actions_probs[j]
                if i != j:
                    divs.append(self.KL((prob1 + Config.epsilon).log(), prob2))
        return torch.stack(divs).mean()

    def forward(self, trajectory: List[Step], agent: Agent) -> torch.Tensor:
        fb = ForwardBackward(agent, trajectory)
        logprobs = []
        all_kldivs = []
        for step_idx, step in enumerate(trajectory):
            meta = agent.meta(step.current_obs)
            all_actions_probs = []
            for opt_idx, option in enumerate(agent.options):
                actions_probs = option.policy(step.current_obs)
                all_actions_probs.append(actions_probs)
                action = actions_probs[torch.arange(Config.batch_size), step.current_action]
                is_option_factor = fb.is_option_factor(opt_idx, step_idx)
                has_switch_to_option_factor = fb.has_switch_to_option_factor(opt_idx, step_idx)
                add_logprob(logprobs, meta[:, opt_idx], has_switch_to_option_factor)
                add_logprob(logprobs, action, is_option_factor)
                if step_idx < len(trajectory) - 1:
                    useless_next_switch = fb.useless_switch(opt_idx, step_idx + 1) * Config.useless_switch_factor
                    next_step = trajectory[step_idx + 1]
                    next_termination = option.termination(next_step.current_obs)
                    option_will_continue_factor = fb.option_will_continue_factor(opt_idx, step_idx)
                    option_will_terminate_factor = fb.option_will_terminate_factor(opt_idx, step_idx)
                    add_logprob(logprobs, next_termination, option_will_terminate_factor - useless_next_switch)
                    add_logprob(logprobs, (1. - next_termination), option_will_continue_factor + useless_next_switch)
            kldiv = self.kl_divergence(all_actions_probs)
            all_kldivs.append(kldiv)
        return -torch.cat(logprobs).mean() + Config.kl_divergence_factor * torch.stack(all_kldivs).mean()


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
            torch.nn.Linear(Config.taxi_ninputs, Config.taxi_hidden_layer),
            torch.nn.Tanh(),
            torch.nn.Linear(Config.taxi_hidden_layer, Config.taxi_nactions),
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
            torch.nn.Linear(Config.taxi_ninputs, Config.taxi_hidden_layer),
            torch.nn.Tanh(),
            torch.nn.Linear(Config.taxi_hidden_layer, 1),
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

class DebugPolicyNetwork(torch.nn.Module):
    def __init__(self, index: int) -> None:
        self.index = index
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros([Config.batch_size, Config.taxi_nactions])
        outputs[:,self.index] = 10.
        return outputs.softmax(1)

class DebugAgent(TaxiAgent):
    def __init__(self) -> None:
        super().__init__()
        assert Config.noptions == Config.taxi_nactions
        self.options = torch.nn.Sequential(
            *[Option(DebugPolicyNetwork(idx), TaxiTerminationNetwork()) for idx in range(Config.taxi_nactions)]
        )
