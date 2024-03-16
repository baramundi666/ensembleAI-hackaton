import torch
from taskdataset import TaskDataset
from img_augmentation import crop_img, color_distort, rotate_img


def augment_dataset(dataset: TaskDataset):
    augmented_dataset = TaskDataset()
    for i in range(len(dataset.labels)):
        img = dataset.imgs[i]
        label = dataset.labels[i]
        ids = dataset.ids[i]

        # augmenting the img
        crop_color = color_distort(crop_img(img))
        crop_rotate = rotate_img(crop_img(img))

        augmented_dataset.imgs.append(crop_color)
        augmented_dataset.labels.append(label)
        augmented_dataset.ids.append(ids)

        augmented_dataset.imgs.append(crop_rotate)
        augmented_dataset.labels.append(label)
        augmented_dataset.ids.append(ids)

    dataset.imgs += augmented_dataset.imgs
    dataset.labels += augmented_dataset.labels
    dataset.ids += augmented_dataset.ids

    return dataset


if __name__ == "__main__":
    pt_dataset = torch.load("../modelstealing/data/ModelStealingPub.pt")
    dataset = TaskDataset()
    dataset.ids = pt_dataset.ids
    dataset.imgs = pt_dataset.imgs
    dataset.labels = pt_dataset.labels

    augmented_dataset = augment_dataset(dataset)

