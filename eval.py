import os
import time
import argparse

import torch

from env import Env
from network import TaxiAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize a trained agent on Taxi-V3")
    parser.add_argument("checkpoint", type=str, help="path of the checkpoint of the model")
    parser.add_argument("--greedy", action="store_true", help="choose next action greedly with argmax on actions probs (action randomly sampled from actions probs by default)")
    parser.add_argument("--only-display-option", type=int, help="only display the environment whan a specific option is selected (display for all the options by default", default=-1)
    args = parser.parse_args()
    env = Env()
    agent = TaxiAgent()
    agent.load_state_dict(torch.load(args.checkpoint))
    only = args.only_display_option

    try:
        step = 0
        while True:
            with torch.no_grad():
                obs = env.tensor().reshape([1, -1])
                action = agent.action(obs, greedy=args.greedy)
                env.step(action)
                if env.done:
                    agent.reset()
                if only < 0 or only == agent.current_option:
                    os.system("clear")
                    env.render()
                    # print("Done : ", env.done)
                    print("Rewards : ", env.rewards)
                    # print("Step : ", step)
                    print("Option : ", agent.previous_option)
                    print("Option prob : ", agent.meta_prob)
                    print("Action prob : ", agent.action_prob)
                    print("Termination prob : ", agent.termination_prob)
                    if only >= 0:
                        time.sleep(.1)
                    else:
                        time.sleep(.1)
                else:
                    os.system("clear")
                    print("skip")
                step += 1
    except KeyboardInterrupt:
        exit()
