from typing import List, Tuple, NamedTuple, MutableMapping, Optional

import torch

from .config import Config
from .utils import Step, Option, Agent


class ForwardBackward:

    def __init__(self, agent: Agent, trajectory: List[Step]) -> None:
        self.agent = agent
        self.trajectory = trajectory
        self.forward_option_prob_tracker: MutableMapping[Tuple[int, int], torch.Tensor] = {}
        self.backward_option_prob_tracker: MutableMapping[Tuple[int, int], torch.Tensor] = {}

    def forward_option_prob(self, option_idx: int, step: int) -> torch.Tensor:
        key = (option_idx, step)
        if key in self.forward_option_prob_tracker:
            return self.forward_option_prob_tracker[key]
        prob = self._forward_option_prob(option_idx, step)
        prob = prob.detach()
        self.forward_option_prob_tracker[key] = prob
        return prob

    def _forward_option_prob(self, option_idx: int, step: int) -> torch.Tensor:
        current_obs = self.trajectory[step].current_obs
        if step == 0:
            return self.agent.meta(current_obs)[:, option_idx]
        previous_obs = self.trajectory[step - 1].current_obs
        previous_action = self.trajectory[step - 1].current_action

        switch_to_option_prob = self.agent.meta(current_obs)[:, option_idx]
        any_switch_prob = torch.zeros([Config.batch_size], device=Config.device)
        stay_to_option_prob = torch.zeros([Config.batch_size], device=Config.device)
        for idx, option in enumerate(self.agent.options):
            previous_option_prob = self.forward_option_prob(idx, step - 1)
            action_prob = option.policy(previous_obs)[torch.arange(Config.batch_size), previous_action]
            termination_prob = option.termination(current_obs)
            if idx == option_idx:
                stay_to_option_prob += action_prob * previous_option_prob * (1 - termination_prob)
            any_switch_prob += action_prob * previous_option_prob * termination_prob
        return any_switch_prob*switch_to_option_prob + stay_to_option_prob + Config.epsilon

    def backward_option_prob(self, option_idx: int, step: int) -> torch.Tensor:
        key = (option_idx, step)
        if key in self.backward_option_prob_tracker:
            return self.backward_option_prob_tracker[key]
        prob = self._backward_option_prob(option_idx, step)
        prob = prob.detach()
        self.backward_option_prob_tracker[key] = prob
        return prob

    def _backward_option_prob(self, option_idx: int, step: int) -> torch.Tensor:
        current_obs = self.trajectory[step].current_obs
        current_action = self.trajectory[step].current_action
        option = self.agent.options[option_idx]
        action_prob = option.policy(current_obs)[torch.arange(Config.batch_size), current_action]
        if step == Config.nsteps - 1:
            return action_prob
        next_obs = self.trajectory[step + 1].current_obs
        next_termination_prob = option.termination(next_obs)
        continue_same_option_prob = self.backward_option_prob(option_idx, step + 1)
        change_option_prob = torch.zeros([Config.batch_size], device=Config.device)
        for idx, next_option in enumerate(self.agent.options):
            next_option_prob = self.agent.meta(next_obs)[:, idx]
            backward_prob = self.backward_option_prob(idx, step + 1)
            change_option_prob += next_option_prob * backward_prob
        return action_prob * (next_termination_prob * change_option_prob + (1 - next_termination_prob) * continue_same_option_prob) + Config.epsilon

    def option_prob(self, option_idx: int, step: int) -> torch.Tensor:
        return self.forward_option_prob(option_idx, step) * self.backward_option_prob(option_idx, step)

    def step_prob(self, step: int) -> torch.Tensor:
        all_prob = [self.option_prob(idx, step) for idx in range(Config.noptions)]
        return torch.sum(torch.stack(all_prob), dim=0)

    def is_option_factor(self, option_idx: int, step: int) -> torch.Tensor:
        with torch.no_grad():
            return (self.option_prob(option_idx, step) / self.step_prob(step))

    def has_switch_to_option_factor(self, option_idx: int, step: int) -> torch.Tensor:
        with torch.no_grad():
            if step == 0:
                return self.is_option_factor(option_idx, step)
            previous_obs = self.trajectory[step - 1].current_obs
            current_obs = self.trajectory[step].current_obs
            previous_action = self.trajectory[step - 1].current_action
            def option_end_prob(idx: int) -> torch.Tensor:
                option = self.agent.options[idx]
                previous_action_prob = option.policy(previous_obs)[torch.arange(Config.batch_size), previous_action]
                return self.forward_option_prob(idx, step - 1) * previous_action_prob * option.termination(current_obs)
            all_end_prob = [option_end_prob(idx) for idx in range(Config.noptions)]
            any_option_end_prob = torch.sum(torch.stack(all_end_prob), dim=0)
            return (
                (1 / self.step_prob(step)) *
                any_option_end_prob *
                self.agent.meta(current_obs)[:, option_idx] *
                self.backward_option_prob(option_idx, step)
            )

    def option_will_continue_factor(self, option_idx: int, step: int) -> torch.Tensor:
        with torch.no_grad():
            option = self.agent.options[option_idx]
            current_obs = self.trajectory[step].current_obs
            next_obs = self.trajectory[step + 1].current_obs
            current_action = self.trajectory[step].current_action
            action_prob = option.policy(current_obs)[torch.arange(Config.batch_size), current_action]
            option_end_prob = option.termination(next_obs)
            return (
                (1. / self.step_prob(step)) *
                self.forward_option_prob(option_idx, step) *
                action_prob *
                (1. - option_end_prob) *
                self.backward_option_prob(option_idx, step + 1)
            )

    def option_will_terminate_factor(self, option_idx: int, step: int) -> torch.Tensor:
        with torch.no_grad():
            return (self.is_option_factor(option_idx, step) - self.option_will_continue_factor(option_idx, step))

    def useless_switch(self, option_idx: int, step: int) -> torch.Tensor:
        with torch.no_grad():
            is_option_factor = self.is_option_factor(option_idx, step - 1)
            useless_switch = is_option_factor * self.has_switch_to_option_factor(option_idx, step)
            return useless_switch
