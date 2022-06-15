class TaxiConfig:
    nrow = 5
    ncol = 5
    npassenger_pos = 5
    ndestination = 4
    expert_epsilon = 0.05
    row_offset = 0
    col_offset = nrow
    passenger_offset = col_offset + ncol
    destination_offset = passenger_offset + ndestination
    ninputs = nrow + ncol + npassenger_pos + ndestination
    nactions = 6
    hidden_layer = [32, 32]
    init_std = 1.
