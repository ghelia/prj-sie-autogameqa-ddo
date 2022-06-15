import os
from typing import List, Tuple

from PIL import Image
import numpy as np
import pandas
import torch

from ..utils import Env, Step, Agent
from .config import PGConfig
from ..config import Config


CONTROLS: List[List[int]] = []


class PGStep:
    def __init__(self, frame_path: str, controls: List[int]) -> None:
        img = Image.open(frame_path)
        self.frame = np.array(img)
        self.controls = controls
        if controls not in CONTROLS:
            CONTROLS.append(controls)
        self.action = CONTROLS.index(controls)


class Trajectory:
    def __init__(self, dir_path: str, csv_path: str) -> None:
        self.steps = []
        self.observations = []
        csv = pandas.read_csv(csv_path, delimiter=",")
        for idx in range(len(csv)):
            frame_path = csv.iloc[idx, 0]
            controls = csv.iloc[idx, 1:].tolist()
            self.steps.append(PGStep(os.path.join(dir_path, frame_path), controls))
        for step in range(len(self.steps)):
            ob = self._obs(step)
            self.observations.append(ob)

    def  obs(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.observations[idx]

    def  _obs(self, idx: int) -> Tuple[torch.Tensor, int]:
        action = self.steps[idx].action
        frames = []
        for i  in PGConfig.frame_obs_idx:
            fi = idx + i
            if fi < 0:
                fi = 0
            frames.append(self.steps[fi].frame)
        return (
            torch.tensor(frames),
            action
        )


class ExpertData(Env):
    def __init__(self, csv_list: List[str]) -> None:
        self.trajectories = []
        for csv_path in csv_list:
            dir_path = os.path.dirname(csv_path)
            traj = Trajectory(dir_path, csv_path)
            self.trajectories.append(traj)

    def eval_agent(self, agent: Agent) -> float:
        # TODO add evaluation
        return 0.

    def batch(self, batch_size: int) -> List[Step]:
        all_obs = []
        all_actions = []
        for bi in range(batch_size):
            obs = []
            actions = []
            traj = self.trajectories[np.random.randint(len(self.trajectories))]
            start = np.random.randint(len(traj.steps) - Config.nsteps)
            for s in range(Config.nsteps):
                o, a = traj.obs(start + s)
                obs.append(o)
                actions.append(a)
            all_obs.append(obs)
            all_actions.append(actions)
        batch = []
        for s in range(Config.nsteps):
            step = Step(
                torch.stack([obs[s] for obs in all_obs]),
                torch.tensor([actions[s] for actions in all_actions], device=Config.device).long(),
            )
            batch.append(step)
        return batch

