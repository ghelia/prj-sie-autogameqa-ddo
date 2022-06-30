from abc import ABC, abstractmethod
from typing import List, Tuple, NamedTuple, MutableMapping, Optional, Any

import torch
import numpy as np
from tqdm import tqdm

from .config import Config
from .recorder import Recorder


class Step(NamedTuple):
    current_obs: torch.Tensor
    current_action: Optional[torch.Tensor]
    weight: torch.Tensor


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
        self.previous_option = -1
        self.current_option = -1
        self.option_tracker = [0 for _ in range(Config.noptions)]
        self.option_change_tracker = [0 for _ in range(Config.noptions)]
        self.action_tracker = {}
        self.action_prob = 0.
        self.termination_prob = 0.
        self.meta_prob = 0.

    def reset(self, reset_tracker: bool = True) -> None:
        self.current_option = -1
        if reset_tracker:
            self.action_tracker = {}
            self.option_tracker = [0 for _ in range(Config.noptions)]
            self.option_change_tracker = [0 for _ in range(Config.noptions)]

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

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
        obs = self.preprocess(obs)
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
        if action not in self.action_tracker:
            self.action_tracker[action] = 0
        self.action_tracker[action] += 1
        return action

    def all_probs(self, obs: torch.Tensor) -> torch.Tensor:
        probs = []
        for option in self.options:
            prob = option.policy(obs)[0]
            probs.append(prob)
        return torch.stack(probs)


class Env(torch.nn.Module):

    def record(self, trajectory: List[Step], recorder: Recorder, agent: Agent) -> None:
        pass

    def eval_agent(self, agent: Agent, neval: int, nsteps: int) -> float:
        success = 0
        print("eval")
        agent.reset()
        with torch.no_grad():
            for _ in tqdm(range(neval)):
                agent.reset(reset_tracker=False)
                trajectory = self.eval_batch(1, nsteps)
                for step in trajectory:
                    if step.current_action is None:
                        continue
                    expert_action = step.current_action.item()
                    obs = step.current_obs
                    agent_action = agent.action(obs, greedy=True)
                    if expert_action == agent_action:
                        success += 1
        success_rate = success/(nsteps*neval)
        print("success : ", success_rate)
        print("total steps : ", (nsteps*neval))
        print("action selections : ", agent.action_tracker)
        print("option selections : ", agent.option_tracker)
        print("option changements : ", agent.option_change_tracker)
        agent.reset()
        return success_rate

    @abstractmethod
    def batch(self, batch_size: int, nsteps: int) -> List[Step]:
        raise NotImplementedError

    @abstractmethod
    def eval_batch(self, batch_size: int, nsteps: int) -> List[Step]:
        raise NotImplementedError


class DDOData(torch.utils.data.Dataset):

    def __init__(self, env: Env) -> None:
        self.env = env

    def __len__(self) -> int:
        return Config.nsubepoch

    def __getitem__(self, idx: int) -> Any:
        return self.env.batch(Config.batch_size)
