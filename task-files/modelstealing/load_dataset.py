import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    pt_dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")
    dataset = TaskDataset()
    dataset.ids = pt_dataset.ids
    dataset.imgs = pt_dataset.imgs
    dataset.labels = pt_dataset.labels
    print(dataset.transform)
    dataset.imgs[0].show()
    dataset.imgs[1].show()
