import os
import argparse

from ddo.ddo import ddo
from ddo.config import Config
from ddo.recorder import Recorder
from ddo.pseudogame.network import PGAgent
from ddo.pseudogame.data import ExpertData



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDO model for Pseudogame")
    parser.add_argument("datapath", type=str, help="path of the directory containing CSV files of the expert")
    args = parser.parse_args()
    csvs = []
    for root, dirs, files in os.walk(args.datapath):
        for file in files:
            if file.endswith(".csv"):
                csvs.append(os.path.join(root, file))
    print("Expert data : ", csvs)
    if not os.path.exists(os.path.join("./saves", Config.session)):
        os.makedirs(os.path.join("./saves", Config.session))
    recorder = Recorder(os.path.join("./logs", Config.session))
    save_path = os.path.join("./saves", Config.session)
    data = ExpertData(csvs)
    agent = PGAgent()
    ddo(agent, recorder, save_path, data)
