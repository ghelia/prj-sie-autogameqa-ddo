import os
import time
from typing import Tuple, List

import torch
import gym
import numpy as np

from ..config import Config
from .config import TaxiConfig
from ..utils import Step
from ..recorder import Recorder
from ..utils import Env, Agent
from .network import TaxiAgent

class Action:
    DOWN = 0       # 0: move south
    UP = 1     # 1: move north
    RIGHT = 2    # 2: move east
    LEFT = 3     # 3: move west
    PICKUP = 4   # 4: pickup passenger
    DROPOFF = 5  # 5: drop off passenger

    labels = [
        "Down",
        "Up",
        "Right",
        "Left",
        "Pickup",
        "Dropoff"
    ]

red = (0, 0)
green = (0, 4)
yellow = (4, 0)
blue = (4, 3)

GoToRed = [
    [Action.UP, Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN],
    [Action.UP, Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN],
    [Action.UP, Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT],
    [Action.UP, Action.UP, Action.LEFT, Action.UP, Action.LEFT],
    [Action.UP, Action.UP, Action.LEFT, Action.UP, Action.LEFT],
]

GoToYellow = [
    [Action.DOWN, Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN],
    [Action.DOWN, Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN],
    [Action.DOWN, Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT],
    [Action.DOWN, Action.UP, Action.LEFT, Action.UP, Action.LEFT],
    [Action.DOWN, Action.UP, Action.LEFT, Action.UP, Action.LEFT],
]

GoToGreen = [
    [Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.UP],
    [Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.UP],
    [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.UP],
    [Action.UP, Action.UP, Action.UP, Action.RIGHT, Action.UP],
    [Action.UP, Action.UP, Action.UP, Action.RIGHT, Action.UP],
]

GoToBlue = [
    [Action.DOWN, Action.DOWN, Action.RIGHT, Action.DOWN, Action.LEFT],
    [Action.DOWN, Action.DOWN, Action.RIGHT, Action.DOWN, Action.LEFT],
    [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.LEFT],
    [Action.UP, Action.UP, Action.UP, Action.DOWN, Action.LEFT],
    [Action.UP, Action.UP, Action.UP, Action.DOWN, Action.LEFT],
]

GoTo = [GoToRed, GoToGreen, GoToYellow, GoToBlue]
Goal = [red, green, yellow, blue]


class TaxiEnv(Env):

    def __init__(self) -> None:
        env = gym.make("Taxi-v3")
        self.expert = Expert()
        self.env = env
        self.taxi_row: int
        self.taxi_col: int
        self.passenger_index: int
        self.destination_index: int
        self.position: Tuple[int, int]
        self.reset()
        self.rewards = 0.
        self.done = False

    def decode(self, state: int) -> None:
        (self.taxi_row,
         self.taxi_col,
         self.passenger_index,
         self.destination_index) = self.env.decode(state)
        self.position = (self.taxi_row, self.taxi_col)

    def reset(self) -> None:
        state = self.env.reset()
        assert isinstance(state, int)
        self.decode(state)

    def tensor(self) -> torch.Tensor:
        return self._tensor(self.taxi_row, self.taxi_col, self.passenger_index, self.destination_index)

    def _tensor(self, row: int, col: int, passenger: int, destination: int) -> torch.Tensor:
        inputs = torch.zeros([TaxiConfig.ninputs])
        inputs[TaxiConfig.row_offset + row] = 1.
        inputs[TaxiConfig.col_offset + col] = 1.
        inputs[TaxiConfig.passenger_offset + passenger] = 1.
        inputs[TaxiConfig.destination_offset + destination] = 1.
        return inputs

    def render(self) -> None:
        print("taxi : ", self.taxi_row, self.taxi_col)
        print("passenger : ", self.passenger_index, " -> ", self.destination_index)
        self.env.render()

    def step(self, action: int) -> None:
        state, reward, done, _ = self.env.step(action)
        self.done = done
        self.rewards += reward
        assert isinstance(state, int)
        self.decode(state)
        if done:
            self.reset()

    def eval_agent(self, agent: Agent) -> float:
        assert isinstance(agent, TaxiAgent)
        self.reset()
        agent.reset()
        success = 0
        L = 1000
        for s in range(L):
            expert_action = self.expert.action(self)
            obs = self.tensor().reshape([1, -1])
            agent_action = agent.action(obs, greedy=True)
            self.step(expert_action)
            if expert_action == agent_action:
                success += 1
        success_rate = success/L
        print("success : ", success_rate)
        print("option selections : ", agent.option_tracker)
        print("option changements : ", agent.option_change_tracker)
        self.reset()
        agent.reset()
        return success_rate


    def batch(self, batch_size: int) -> List[Step]:
        all_obs = []
        all_actions = []
        for bi in range(batch_size):
            obs = []
            actions = []
            for s in range(Config.nsteps):
                obs.append(self.tensor())
                action = self.expert.action(self)
                self.step(action)
                actions.append(action)
            all_obs.append(obs)
            all_actions.append(actions)
        trajectory = []
        for s in range(Config.nsteps):
            step = Step(
                torch.stack([obs[s] for obs in all_obs]),
                torch.tensor([actions[s] for actions in all_actions], device=Config.device).long(),
            )
            trajectory.append(step)
        return trajectory


class Expert:
    def __init__(self) -> None:
        self.picked = False

    def action(self, env: TaxiEnv) -> int:
        if np.random.random() < TaxiConfig.expert_epsilon:
            return np.random.randint(4)
        if self.picked:
            goal = env.destination_index
            if Goal[goal] == env.position:
                self.picked = False
                return Action.DROPOFF
            return GoTo[goal][env.taxi_row][env.taxi_col]
        else:
            goal = env.passenger_index
            if Goal[goal] == env.position:
                self.picked = True
                return Action.PICKUP
            return GoTo[goal][env.taxi_row][env.taxi_col]
