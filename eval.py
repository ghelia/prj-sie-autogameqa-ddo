import os
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddo.env import Env, Action
from ddo.network import TaxiAgent
from ddo.config import Config


def plot_distributions(distributions, keys) -> None:
    x_offset = -Config.noptions/2 * 0.1
    x_axis = np.arange(len(keys))
    plt.xticks(x_axis, keys.keys())
    for option in range(Config.noptions):
        label = f"Option {option}"
        counts = [
            (distributions[key][option] if key in distributions and option in distributions[key] else 0)
            for key in keys.values()
        ]
        plt.bar(x_axis + x_offset, counts, 0.1, label=label)
        x_offset += 0.1

    plt.xlabel("Type of action")
    plt.ylabel("Action count")
    plt.title("Distributions of options during each pickup/dropoff actions")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize a trained agent on Taxi-V3")
    parser.add_argument("checkpoint", type=str, help="path of the checkpoint of the model")
    parser.add_argument("--greedy", action="store_true", help="choose next action greedly with argmax on actions probs (action randomly sampled from actions probs by default)")
    parser.add_argument("--only-display-option", type=int, help="only display the environment whan a specific option is selected (display for all the options by default", default=-1)
    parser.add_argument("--only-use-option", type=int, help="only use a specific option of the model", default=None)
    parser.add_argument("--framerate", type=float, help="duration of one frame when rendering environment", default=.1)
    args = parser.parse_args()
    env = Env()
    agent = TaxiAgent()
    agent.load_state_dict(torch.load(args.checkpoint))
    only = args.only_display_option

    distributions = {}
    all_distributions = {}

    try:
        step = 0
        while True:
            with torch.no_grad():
                obs = env.tensor().reshape([1, -1])
                position = (env.taxi_row, env.taxi_col)
                action = agent.action(
                    obs,
                    greedy=args.greedy,
                    only_option=args.only_use_option
                )

                if action not in all_distributions:
                    all_distributions[action] = {}
                if agent.previous_option not in all_distributions[action]:
                    all_distributions[action][agent.previous_option] = 0
                all_distributions[action][agent.previous_option] += 1

                if (action, position) not in distributions:
                    distributions[(action, position)] = {}
                if agent.previous_option not in distributions[(action, position)]:
                    distributions[(action, position)][agent.previous_option] = 0
                distributions[(action, position)][agent.previous_option] += 1

                env.step(action)
                if env.done:
                    agent.reset()
                if only < 0 or only == agent.current_option:
                    os.system("clear")
                    env.render()
                    print("Rewards : ", env.rewards)
                    print("Option : ", agent.previous_option)
                    print("Option prob : ", agent.meta_prob)
                    print("Action prob : ", agent.action_prob)
                    print("Termination prob : ", agent.termination_prob)
                    time.sleep(args.framerate)
                else:
                    os.system("clear")
                    print("skip")
                step += 1
    except KeyboardInterrupt:
        keys = {
            "pickup 1": (Action.PICKUP, (0,0)),
            "pickup 2": (Action.PICKUP, (4,0)),
            "pickup 3": (Action.PICKUP, (0,4)),
            "pickup 4": (Action.PICKUP, (4,3)),
            "dropoff 1": (Action.DROPOFF, (0,0)),
            "dropoff 2": (Action.DROPOFF, (4,0)),
            "dropoff 3": (Action.DROPOFF, (0,4)),
            "dropoff 4": (Action.DROPOFF, (4,3))
        }
        plot_distributions(distributions, keys)

        keys = {
            "pickup": Action.PICKUP,
            "dropoff": Action.DROPOFF,
            "left": Action.LEFT,
            "right": Action.RIGHT,
            "up": Action.UP,
            "down": Action.DOWN
        }
        plot_distributions(all_distributions, keys)
        exit()
