import torch


class Config:

    learning_rate = 0.005
    learning_rate_decay = 0.995
    batch_size = 10
    nsteps = 100
    noptions = 4
    nepoch = 100000
    nsubepoch = 10
    epsilon = 1e-12

    useless_switch_factor = 0.33
    kl_divergence_factor = 0.1

    taxi_nrow = 5
    taxi_ncol = 5
    taxi_npassenger_pos = 5
    taxi_ndestination = 4
    taxi_init_std = 1.
    taxi_expert_epsilon = 0.05

    taxi_row_offset = 0
    taxi_col_offset = taxi_nrow
    taxi_passenger_offset = taxi_col_offset + taxi_ncol
    taxi_destination_offset = taxi_passenger_offset + taxi_ndestination

    taxi_ninputs = taxi_nrow + taxi_ncol + taxi_npassenger_pos + taxi_ndestination
    taxi_nactions = 6
    taxi_hidden_layer = 32

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
