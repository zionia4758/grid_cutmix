import numpy as np
import random
import torch
# import torch.backends.cudnn as cudnn

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    