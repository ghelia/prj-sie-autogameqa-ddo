from typing import MutableMapping, List, Any
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddo.env import Env, Action, Goal
from ddo.network import TaxiAgent
from ddo.config import Config

def plot_probs(ax: Any, all_probs: torch.Tensor, title: str) -> None:
    x_offset = -all_probs.shape[0]/2 * 0.1
    x_axis = np.arange(all_probs.shape[1])
    ax.set_xticks(x_axis)
    ax.set_xticklabels(Action.labels())
    for idx in range(all_probs.shape[0]):
        probs = all_probs[idx]
        ax.bar(x_axis + x_offset, probs.numpy(), 0.1, label=f"option {idx}")
        x_offset += 0.1
    ax.set_title(title)
    ax.set_ylabel("Action probability")

def plot_all_probs(probs_list: List[torch.Tensor], title_list: List[str]) -> None:
    axid = [421, 422, 423, 424, 425, 426, 427, 428, 429]
    assert len(probs_list) == len(title_list)
    for idx in range(len(probs_list)):
        ax = plt.subplot(axid[idx])
        plot_probs(ax, probs_list[idx], title_list[idx])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gcf().legend(handles, labels, loc='upper right')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot options actions distribution of a trained agent on Taxi-V3")
    parser.add_argument("checkpoint", type=str, help="path of the checkpoint of the model")
    args = parser.parse_args()
    env = Env()
    agent = TaxiAgent()
    agent.load_state_dict(torch.load(args.checkpoint))
    probs = []
    labels = []

    special_states = {
        "pickup 1": ((0,0), 0, 1),
        "pickup 2": ((0,4), 1, 3),
        "pickup 3": ((4,0), 2, 1),
        "pickup 4": ((4,3), 3, 1),
        "dropoff 1": ((0,0), 4, 0),
        "dropoff 2": ((0,4), 4, 1),
        "dropoff 3": ((4,0), 4, 2),
        "dropoff 4": ((4,3), 4, 3)
    }
    all_special_labels = {v:k for k,v in special_states.items()}
    for label, state in special_states.items():
        obs = env._tensor(state[0][0], state[0][1], state[1], state[2]).reshape([1, -1])
        all_probs = agent.all_probs(obs)
        probs.append(all_probs.detach())
        labels.append(label)

    plot_all_probs(probs, labels)
