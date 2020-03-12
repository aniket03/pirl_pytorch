import random
import numpy as np
import torch


def set_random_generators_seed():

    # Set up random seed to 1008. Do not change the random seed.
    # Yes, these are all necessary when you run experiments!
    seed = 1008
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
