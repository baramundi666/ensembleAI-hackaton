import torch
from taskdataset import TaskDataset
from img_augmentation import crop_img, rotate_img, color_distort, add_gaussian_noise


if __name__ == "__main__":
    dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")
    #dataset = TaskDataset(dataset)

    image = dataset.imgs[3]
    image = add_gaussian_noise(image)
    image.show()

    
