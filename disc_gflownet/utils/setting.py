import torch
import numpy as np
import random

# global variable
_dev = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]

def set_device(dev):
    """Set the global device."""
    _dev[0] = dev

def get_device():
    """Get the current global device."""
    return _dev[0]



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# tf = lambda x: torch.FloatTensor(x).to(_dev[0])
# tl = lambda x: torch.LongTensor(x).to(_dev[0])

# Helper functions
def tf(x):
    """Convert input to a FloatTensor on the current device."""
    if isinstance(x, list):
        x = np.array(x)
    return torch.FloatTensor(x).to(_dev[0])

def tl(x):
    """Convert input to a LongTensor on the current device."""
    if isinstance(x, list):
        x = np.array(x)
    return torch.LongTensor(x).to(_dev[0])



