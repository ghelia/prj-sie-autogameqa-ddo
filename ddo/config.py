import torch
from datetime import datetime


class Config:

    learning_rate = 0.015
    learning_rate_decay = 0.995
    batch_size = 2
    nsteps = 3
    noptions = 4
    nepoch = 100000
    nsubepoch = 5
    epsilon = 1e-12
    neval = 3
    eval_nsteps = 10

    useless_switch_factor = 0.15
    kl_divergence_factor = 0. # 0.001

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    session = datetime.now().strftime(f"%m_%d_%Y__%Hh%Mm%Ss")
