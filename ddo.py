import os
from typing import Tuple, List, Callable

import torch
import numpy as np
from tqdm import tqdm

from config import Config
from network import DDOLoss, TaxiAgent
from utils import Agent
from env import Expert, Env
from recorder import Recorder


def eval_agent(agent: TaxiAgent, expert: Expert, env: Env) -> None:
    env.reset()
    agent.reset()
    success = 0
    for s in range(1000):
        expert_action = expert.action(env)
        obs = env.tensor().reshape([1, -1])
        agent_action = agent.action(obs)
        env.step(expert_action)
        if expert_action == agent_action:
            success += 1
    print("success : ", success/1000)
    print("option selections : ", agent.option_tracker)
    print("option changements : ", agent.option_change_tracker)


def ddo(agent: Agent, recorder: Recorder, save_path: str, batch: Callable) -> None:
    env = Env()
    expert = Expert()
    optimizer = torch.optim.Adam(agent.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Config.learning_rate_decay)
    agent.train()
    loss_func = DDOLoss()

    for E in range(Config.nepoch):
        print(f"")
        print(f"Epoch {E}")
        all_losses = []
        for e in tqdm(range(Config.nsubepoch)):
            optimizer.zero_grad()
            trajectory = batch(Config.batch_size)

            loss = loss_func(trajectory, agent)
            all_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            recorder.scalar(loss.item(), "loss")
            recorder.scalar(scheduler.get_last_lr()[0], "learning rate")
        torch.save(agent.state_dict(), os.path.join(save_path, f'agent-{E}.chkpt'))
        print(f"Loss {np.mean(all_losses)}")
        recorder.gradients_and_weights(agent)
        recorder.end_epoch()
        scheduler.step()
        eval_agent(agent, expert, env)

