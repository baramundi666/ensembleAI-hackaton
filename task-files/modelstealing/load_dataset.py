import torch
from taskdataset import TaskDataset
from img_augmentation import crop_img, rotate_img, color_distort, add_gaussian_noise




if __name__ == "__main__":
    pt_dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")
    dataset = TaskDataset()
    dataset.ids = pt_dataset.ids
    dataset.imgs = pt_dataset.imgs
    dataset.labels = pt_dataset.labels

    
