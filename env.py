import os
import time
from typing import Tuple, List

import torch
import gym

from config import Config
from utils import Step

class Action:
    DOWN = 0       # 0: move south
    UP = 1     # 1: move north
    RIGHT = 2    # 2: move east
    LEFT = 3     # 3: move west
    PICKUP = 4   # 4: pickup passenger
    DROPOFF = 5  # 5: drop off passenger

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


class Env:
    def __init__(self) -> None:
        env = gym.make("Taxi-v3")
        self.env = env
        self.taxi_row: int
        self.taxi_col: int
        self.passenger_index: int
        self.destination_index: int
        self.position: Tuple[int, int]
        self.reset()
        self.rewards = 0
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
        inputs = torch.zeros([Config.taxi_ninputs])
        inputs[Config.taxi_row_offset + self.taxi_row] = 1.
        inputs[Config.taxi_col_offset + self.taxi_col] = 1.
        inputs[Config.taxi_passenger_offset + self.passenger_index] = 1.
        inputs[Config.taxi_destination_offset + self.destination_index] = 1.
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


class Expert:
    def __init__(self) -> None:
        self.picked = False

    def action(self, env: Env) -> int:
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


class TaxiBatch:
    def __init__(self) -> None:
        self.env =  Env()
        self.expert = Expert()

    def __call__(self, batch_size) -> List[Step]:
        all_obs = []
        all_actions = []
        for bi in range(batch_size):
            obs = []
            actions = []
            for s in range(Config.nsteps):
                obs.append(self.env.tensor())
                action = self.expert.action(self.env)
                self.env.step(action)
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

if __name__ == "__main__":
    env = Env()
    expert = Expert()

    try:
        while True:
            os.system("clear")
            action = expert.action(env)
            env.step(action)
            env.render()
            print("Rewards : ", env.rewards)
            time.sleep(0.1)
    except KeyboardInterrupt:
        exit()
