import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("sybilattack/data/ExampleSybilAttack.pt")

    print(dataset.ids, dataset.imgs, dataset.labels)
