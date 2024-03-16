import torch
from taskdataset import TaskDataset
from img_augmentation import crop_img, rotate_img, color_distort, add_gaussian_noise, sobel_filter

if __name__ == "__main__":
    dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")

    image = sobel_filter(dataset.imgs[0])
    print(dataset.ids[0])
    image.show()
