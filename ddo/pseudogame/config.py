class PGConfig:

    classifier_learning_rate = 0.0001
    classifier_nepoch = 10
    classifier_nsteps = 50

    frame_obs_idx = [-3, -1, 0]
    size = 224
    hidden_layer = [64, 32]
    nfeatures = 2000
    init_std = 0.5
    min_frequency = 0.0005
