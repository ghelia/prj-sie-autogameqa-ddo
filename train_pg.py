from typing import List
import os
import argparse

from ddo.ddo import ddo
from ddo.config import Config
from ddo.recorder import Recorder
from ddo.pseudogame.network import PGAgent
from ddo.pseudogame.data import ExpertData


def get_csvs(path: str) -> List[str]:
    csvs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csvs.append(os.path.join(root, file))
    return csvs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDO model for Pseudogame")
    parser.add_argument("datapath", type=str, help="path of the directory containing CSV files of the expert used for training")
    parser.add_argument("eval_datapath", type=str, help="path of the directory containing CSV files of the expert used for evaluation")
    args = parser.parse_args()
    csvs = get_csvs(args.datapath)
    eval_csvs = get_csvs(args.eval_datapath)
    print("Expert data : ", csvs)
    print("Expert data (eval) : ", eval_csvs)
    if not os.path.exists(os.path.join("./saves", Config.session)):
        os.makedirs(os.path.join("./saves", Config.session))
    recorder = Recorder(os.path.join("./logs", Config.session))
    save_path = os.path.join("./saves", Config.session)
    data = ExpertData(csvs, eval_csvs)
    agent = PGAgent()
    agent.to(Config.device)
    ddo(agent, recorder, save_path, data)
