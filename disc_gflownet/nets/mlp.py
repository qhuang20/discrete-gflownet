import torch
import torch.nn as nn




# def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
#     return nn.Sequential(*(sum([[nn.Linear(i, o)] + ([act] if n < len(l)-2 else []) for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))


def make_mlp(dims, activation=nn.LeakyReLU(), tail_layers=[]):
    """Constructs MLP with linear layers of sizes specified in dims, activation between them.
    Used to create policy networks that output action logits and flow values."""
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(dims)-2:  # No activation after final layer
            layers.append(activation)
    return nn.Sequential(*(layers + tail_layers))


