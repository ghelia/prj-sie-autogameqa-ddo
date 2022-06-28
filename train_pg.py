from typing import List
import os
import argparse

import numpy as np

from ddo.ddo import ddo
from ddo.config import Config
from ddo.recorder import Recorder
from ddo.pseudogame.network import PGAgent
from ddo.pseudogame.data import ExpertData
from ddo.pseudogame.controls import CONTROLS


def get_csvs(path: str) -> List[str]:
    csvs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csvs.append(os.path.join(root, file))
    return csvs


def save_controls(save_path: str) -> None:
    print("controls : ", CONTROLS)
    arr = np.array(CONTROLS)
    np.save(os.path.join(save_path, "controls.npy"), arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDO model for Pseudogame")
    parser.add_argument("imgs", type=str, help="path of the directory containing the frames PNG")
    parser.add_argument("train_csvs", type=str, help="path of the directory containing CSV files of the expert used for training")
    parser.add_argument("eval_csvs", type=str, help="path of the directory containing CSV files of the expert used for evaluation")
    args = parser.parse_args()
    csvs = get_csvs(args.train_csvs)
    eval_csvs = get_csvs(args.eval_csvs)
    print("Expert data : ", csvs)
    print("Expert data (eval) : ", eval_csvs)
    if not os.path.exists(os.path.join("./saves", Config.session)):
        os.makedirs(os.path.join("./saves", Config.session))
    recorder = Recorder(os.path.join("./logs", Config.session))
    save_path = os.path.join("./saves", Config.session)
    data = ExpertData(args.imgs, csvs, eval_csvs)
    save_controls(save_path)
    agent = PGAgent(len(CONTROLS))
    agent.to(Config.device)
    ddo(agent, recorder, save_path, data)
