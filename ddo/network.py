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
        return (-torch.cat(logprobs).mean(), -torch.stack(all_kldivs).mean())


class Dense(torch.nn.Module):
    def __init__(self, layers_dims: List[int], activation: torch.nn.Module, output_activation: torch.nn.Module) -> None:
        super().__init__()
        layers = []
        previous = layers_dims[0]
        for idx, dim in enumerate(layers_dims[1:]):
            if idx > 0:
                layers.append(activation)
            layers.append(torch.nn.Linear(previous, dim))
            previous = dim
        layers.append(output_activation)
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        return outputs

