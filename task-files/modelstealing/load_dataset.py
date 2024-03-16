import torch
from taskdataset import TaskDataset

if __name__ == "__main__":
    dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")
    