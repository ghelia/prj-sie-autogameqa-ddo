import os
from typing import Tuple, List, Callable

import torch
import numpy as np
from tqdm import tqdm

from .config import Config
from .network import DDOLoss, TaxiAgent
from .utils import Agent
from .env import Expert, Env
from .recorder import Recorder


def eval_agent(agent: TaxiAgent, expert: Expert, env: Env, recorder: Recorder) -> float:
    env.reset()
    agent.reset()
    success = 0
    L = 1000
    for s in range(L):
        expert_action = expert.action(env)
        obs = env.tensor().reshape([1, -1])
        agent_action = agent.action(obs, greedy=True)
        env.step(expert_action)
        if expert_action == agent_action:
            success += 1
    success_rate = success/L
    recorder.scalar(success_rate, "evaluation")
    print("success : ", success_rate)
    print("option selections : ", agent.option_tracker)
    print("option changements : ", agent.option_change_tracker)
    return success_rate


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
        all_kl_losses = []
        for e in tqdm(range(Config.nsubepoch)):
            optimizer.zero_grad()
            trajectory = batch(Config.batch_size)

            loss, kl_loss = loss_func(trajectory, agent)
            all_losses.append(loss.item())
            all_kl_losses.append(kl_loss.item())
            (loss + Config.kl_divergence_factor*kl_loss).backward()
            optimizer.step()
            recorder.scalar(loss.item(), "loss")
            recorder.scalar(kl_loss.item(), "kl_loss")
            recorder.scalar(scheduler.get_last_lr()[0], "learning rate")
        print(f"Loss {np.mean(all_losses)}")
        print(f"KL Loss {np.mean(all_kl_losses)}")
        recorder.gradients_and_weights(agent)
        success_rate = eval_agent(agent, expert, env, recorder)
        scheduler.step()
        recorder.end_epoch()
        torch.save(agent.state_dict(), os.path.join(save_path, f'agent-{E}-{success_rate}.chkpt'))

