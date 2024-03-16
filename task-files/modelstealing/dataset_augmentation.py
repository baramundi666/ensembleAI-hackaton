import torch
from taskdataset import TaskDataset
from img_augmentation import crop_img, color_distort


def generate_augmented_dataset(dataset: TaskDataset):
    augmented_dataset = TaskDataset()
    for i in range(len(dataset.labels)):
        img = dataset.imgs[i]
        label = dataset.labels[i]
        _id = dataset.ids[i]

        # here we want to transform each image and keep the label (don't know if id is needed)

        # 1. crop, color
        new_img = crop_img(img)
        new_img = color_distort(new_img)
        augmented_dataset.imgs.append(new_img)
        augmented_dataset.labels.append(label)
        augmented_dataset.ids.append(None)
    return augmented_dataset


if __name__ == "__main__":
    pt_dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")
    dataset = TaskDataset()
    dataset.ids = pt_dataset.ids
    dataset.imgs = pt_dataset.imgs
    dataset.labels = pt_dataset.labels

    augmented_dataset = generate_augmented_dataset(dataset)
