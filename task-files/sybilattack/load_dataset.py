import torch
from taskdataset import TaskDataset

def load():
    return torch.load("data/SybilAttack.pt")

