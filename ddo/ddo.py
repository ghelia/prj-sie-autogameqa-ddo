import os
from typing import Tuple, List, Callable

import torch
import numpy as np
from tqdm import tqdm

from .config import Config
from .network import DDOLoss
from .taxi.network import TaxiAgent
from .utils import Agent, Env
from .recorder import Recorder


def ddo(agent: Agent, recorder: Recorder, save_path: str, env: Env) -> None:
    optimizer = torch.optim.Adam(agent.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Config.learning_rate_decay)
    agent.train()
    loss_func = DDOLoss()

    for E in range(Config.nepoch):
        print(f"")
        print(f"Epoch {E}")
        all_losses = []
        all_kl_losses = []
        print("train")
        for e in tqdm(range(Config.nsubepoch)):
            optimizer.zero_grad()
            trajectory = env.batch(Config.batch_size, Config.nsteps)
            if e == 0:
                env.record(trajectory, recorder, agent)
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
        success_rate = env.eval_agent(agent, Config.neval, Config.eval_nsteps)
        recorder.scalar(success_rate, "evaluation")
        scheduler.step()
        recorder.end_epoch()
        torch.save(agent.state_dict(), os.path.join(save_path, f'agent-{E}-{success_rate}.chkpt'))

