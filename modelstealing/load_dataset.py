import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("modelstealing/data/ExampleModelStealingPub.pt")

    print(dataset.ids, dataset.imgs, dataset.labels)
