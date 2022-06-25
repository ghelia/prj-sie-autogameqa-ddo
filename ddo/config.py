import torch
from datetime import datetime


class Config:

    learning_rate = 0.00001
    learning_rate_decay = 0.995
    batch_size = 2
    nsteps = 30
    noptions = 4
    nepoch = 100000
    nsubepoch = 3
    epsilon = 1e-12

    useless_switch_factor = 0.15
    kl_divergence_factor = 0. # 0.001

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    session = datetime.now().strftime(f"KL{kl_divergence_factor}_%m_%d_%Y, %H:%M:%S")
