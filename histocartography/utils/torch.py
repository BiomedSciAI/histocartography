import torch


def torch_to_numpy(x):
    return x.cpu().detach().numpy()
