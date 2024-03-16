import torch
from taskdataset import TaskDataset
from img_augmentation import crop_img, color_distort


def augment_dataset(dataset: TaskDataset):
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

    dataset.imgs += augmented_dataset.imgs
    dataset.labels += augmented_dataset.labels
    dataset.ids += augmented_dataset.ids

    return dataset
