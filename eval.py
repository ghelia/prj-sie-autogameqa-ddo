import os
import time

import torch

from env import Env
from network import TaxiAgent

if __name__ == "__main__":
    env = Env()
    agent = TaxiAgent()
    # agent.load_state_dict(torch.load("./saves/05_10_2022, 14:20:49/agent-125.chkpt"))
    agent.load_state_dict(torch.load("./saves/05_10_2022, 15:54:00/agent-300.chkpt"))

    try:
        step = 0
        while True:
            with torch.no_grad():
                os.system("clear")
                obs = env.tensor().reshape([1, -1])
                action = agent.action(obs)
                env.step(action)
                if env.done:
                    agent.reset()
                env.render()
                # print("Done : ", env.done)
                # print("Rewards : ", env.rewards)
                # print("Step : ", step)
                print("Option : ", agent.previous_option)
                print("Option prob : ", agent.meta_prob)
                print("Action prob : ", agent.action_prob)
                print("Termination prob : ", agent.termination_prob)
                time.sleep(0.1)
                step += 1
    except KeyboardInterrupt:
        exit()
