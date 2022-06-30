import os
import pathlib
from typing import List, Tuple, Dict

from PIL import Image
import numpy as np
import pandas
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from ..utils import Env, Step, Agent
from ..recorder import Recorder
from .config import PGConfig
from .network import PGAgent
from ..config import Config


CONTROLS: List[List[int]] = []
CONTROLS_COUNT: Dict[str, int] = {}
TOTAL_STEPS = 0


class PGStep:
    def __init__(self, frame_path: str, controls: List[int], transform: transforms.Compose, save_npy: bool = True, force: bool = False) -> None:
        global TOTAL_STEPS
        self.transform = transform
        self.frame_path = frame_path
        self.controls = controls
        TOTAL_STEPS += 1
        if controls not in CONTROLS:
            CONTROLS.append(controls)
            CONTROLS_COUNT[str(controls)] = 0
        self.action = CONTROLS.index(controls)
        CONTROLS_COUNT[str(controls)] += 1
        if save_npy:
            self.save_frame_npy(force)

    def npy_path(self) -> str:
        return self.frame_path + ".npy"

    def load_frame_from_img(self) -> torch.Tensor:
        img = Image.open(self.frame_path)
        frame = self.transform(img)
        return frame

    def save_frame_npy(self, force: bool = False) -> None:
        if not os.path.isfile(self.npy_path()) or force:
            frame = self.load_frame_from_img()
            np.save(self.npy_path(), frame.numpy())

    def frame(self, load_npy: bool = True) -> torch.Tensor:
        npy_path = self.frame_path + ".npy"
        if load_npy:
            frame = torch.tensor(np.load(npy_path))
        else:
            frame = self.load_frame_from_img()
        return frame


class Trajectory:
    def __init__(self, img_dir: str, csv_path: str, transform: transforms.Compose, ) -> None:
        self.steps = []
        csv = pandas.read_csv(csv_path, delimiter=",")
        # NOTE current_frame, current_action
        # for idx in range(len(csv)):
        #     frame_path = csv.iloc[idx, 0]
        #     controls = csv.iloc[idx, 1:].tolist()
        #     self.steps.append(PGStep(self.get_path(dir_path, frame_path), controls, transform))

        # NOTE current_frame, previous_action
        frame_path = None
        for idx in range(len(csv)):
            controls = csv.iloc[idx, 1:].tolist()
            if frame_path is not None:
                self.steps.append(PGStep(self.get_path(img_dir, frame_path), controls, transform))
            frame_path = csv.iloc[idx, 0]

    def get_path(self, img_dir: str, frame_path: str) -> str:
        frame_path = frame_path.replace("\\", "/")
        # fp = pathlib.Path(frame_path)
        # return os.path.join(dir_path, pathlib.Path(*fp.parts[1:]))
        return os.path.join(img_dir, frame_path)

    def weight(self, action: int) -> float:
        freq = CONTROLS_COUNT[str(CONTROLS[action])] / TOTAL_STEPS
        if freq > 0.001:
            return 1. - freq
        return 0.

    def obs(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        action = self.steps[idx].action
        frames = []
        for i  in PGConfig.frame_obs_idx:
            fi = idx + i
            if fi < 0:
                fi = 0
            frames.append(self.steps[fi].frame())
        return (
            torch.stack(frames),
            action,
            self.weight(action)
        )


class ExpertData(Env):

    def record(self, trajectory: List[Step], recorder: Recorder, agent: Agent) -> None:
        assert isinstance(agent, PGAgent)
        r = np.random.randint(len(trajectory))
        with torch.no_grad():
            recorder.image(
                agent.extractor.spatial(trajectory[r].current_obs)[0],
                "spatial"
            )
            recorder.image(
                agent.extractor.temporal(trajectory[r].current_obs)[0],
                "temporal"
            )

    def __init__(self, img_dir: str, csv_list: List[str], eval_csv_list: List[str]) -> None:
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(PGConfig.size)
        ])
        self.trajectories = self.get_trajectories(img_dir, csv_list, Config.nsteps)
        self.eval_trajectories = self.get_trajectories(img_dir, eval_csv_list, Config.eval_nsteps)
        print("Number actions : ", len(CONTROLS))
        for key,count in CONTROLS_COUNT.items():
            print(f"{key} : {count/TOTAL_STEPS}")

    def get_trajectories(self, img_dir: str, csv_list: List[str], min_steps: int) -> List[Trajectory]:
        trajectories = []
        for csv_path in tqdm(csv_list):
            try:
                traj = Trajectory(img_dir, csv_path, self.transforms)
                if len(traj.steps) > min_steps:
                    trajectories.append(traj)
            except pandas.errors.EmptyDataError:
                print(csv_path + " is empty")
                pass
        return trajectories

    def batch(self, batch_size: int, nsteps: int) -> List[Step]:
        return self._batch(self.trajectories, batch_size, nsteps)

    def eval_batch(self, batch_size: int, nsteps: int) -> List[Step]:
        return self._batch(self.eval_trajectories, batch_size, nsteps)

    def _batch(self, trajectories: List[Trajectory], batch_size: int, nsteps: int) -> List[Step]:
        all_obs = []
        all_actions = []
        all_weights = []
        for bi in range(batch_size):
            obs = []
            actions = []
            weights = []
            traj = trajectories[np.random.randint(len(trajectories))]
            start = np.random.randint(len(traj.steps) - nsteps)
            for s in range(nsteps):
                o, a, w = traj.obs(start + s)
                obs.append(o)
                actions.append(a)
                weights.append(w)
            all_obs.append(obs)
            all_actions.append(actions)
            all_weights.append(weights)
        batch = []
        for s in range(nsteps):
            step = Step(
                torch.stack([obs[s] for obs in all_obs]),
                torch.tensor([actions[s] for actions in all_actions], device=Config.device).long(),
                torch.tensor([weights[s] for weights in all_weights], device=Config.device)
            )
            batch.append(step)
        return batch

