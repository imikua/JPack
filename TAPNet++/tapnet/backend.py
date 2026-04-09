import random

import jittor as jt
import numpy as np


def clone_tensor(x):
    if hasattr(x, "clone"):
        return x.clone()
    return jt.array(x)


def detach_tensor(x):
    if hasattr(x, "stop_grad"):
        return x.stop_grad()
    if hasattr(x, "detach"):
        return x.detach()
    return x


def nonzero_mask(x):
    if hasattr(x, "ne"):
        try:
            return x.ne(0)
        except TypeError:
            pass
    return x != 0


def set_global_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    jt.set_global_seed(seed)
    return [seed]
