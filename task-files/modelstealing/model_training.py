from enum import Enum
import torch
from torchvision.transforms import transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from dataset_augmentation import augment_dataset
from taskdataset import TaskDataset
from utils import map_labels, shuffle_multiple_lists, generate_model_name


class ScalingMode(Enum):
    NONE = 1
    UNDER = 2
    UPPER = 3


class ScalingMethod(Enum):
    RANDOM = 1
    INTERVAL = 2


# example scaling config
example_scaling_config = {
    'mode': ScalingMode.UNDER,
    'method': ScalingMethod.RANDOM
}


def scale_dataset(dataset: TaskDataset, scaling_config):
    if scaling_config['method'] == ScalingMethod.INTERVAL:
        start, end = scaling_config['range']
        dataset.imgs = dataset.imgs[start:end]
        dataset.labels = dataset.labels[start:end]
        dataset.ids = dataset.ids[start:end]
        return dataset

    elif scaling_config['method'] == ScalingMethod.RANDOM:
        count = scaling_config['count']
        new_imgs, new_labels, new_ids = shuffle_multiple_lists(dataset.imgs, dataset.labels, dataset.ids)
        new_imgs = new_imgs[:count]
        new_labels = new_labels[:count]
        new_ids = new_ids[:count]
        new_dataset = TaskDataset()
        new_dataset.imgs = new_imgs
        new_dataset.labels = new_labels
        new_dataset.ids = new_ids
        return new_dataset
    else:
        return None

def train_model(scaling_config=None):
    # load dataset
    dataset = torch.load("./data/ModelStealingPub.pt")

    dataset = scale_dataset(dataset,scaling_config)

    # dataset scaling (optional)
    if scaling_config is not None:
        dataset = scale_dataset(dataset, scaling_config)

    # augment dataset
    dataset = augment_dataset(dataset)

    # Uzyskaj dostęp do obrazów i etykiet z wczytanego zestawu danych
    imgs = dataset.imgs
    labels = dataset.labels
    images_rgb = []
    for img in imgs:
        if type(img) is not int:
            images_rgb.append(img.convert("RGB"))

    labels_dict = map_labels(labels)
    labels_int = [labels_dict[label] for label in labels]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Konwertuj listy obrazów i etykiet na tensory
    imgs_tensor = torch.stack([transform(img) for img in images_rgb])
    labels_tensor = torch.tensor(labels_int)

    # Twórz DataLoader bezpośrednio z obrazami i etykietami
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(imgs_tensor, labels_tensor),
        batch_size=32, shuffle=True)

    # Zdefiniuj model
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(labels_int))

    # Wybierz urządzenie
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Definiuj funkcję straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Trenuj model
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), f'models/{generate_model_name()}.pt')


if __name__ == '__main__':
    # temporary dataset cut
    # adjust this config before training
    # config = {
    #     'method': ScalingMethod.INTERVAL,
    #     'range': (0, 13)
    # }

    train_model()
