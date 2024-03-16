import torch
from torchvision.transforms import transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from dataset_augmentation import augment_dataset
from taskdataset import TaskDataset

def map_labels(labels):
    d = dict()
    counter = 1
    for label in labels:
        if label not in d:
            d[label] = counter
            counter += 1
    return d


if __name__ == "__main__":
    # Wczytaj zapisany model danych
    dataset = torch.load("data/ModelStealingPub.pt")
    dataset_pt = TaskDataset()
    dataset_pt.ids = dataset.ids[:13000]
    dataset_pt.imgs = dataset.imgs[:13000]
    dataset_pt.labels = dataset.labels[:13000]

    # Rozszerz model
    dataset = augment_dataset(dataset_pt)

    # Uzyskaj dostęp do obrazów i etykiet z wczytanego zestawu danych
    imgs = dataset.imgs
    labels = dataset.labels
    images_rgb = []
    for img in imgs:
        if type(img) is not int:
            images_rgb.append(img.convert("RGB"))

    # labels_int = [i for i in range(len(dataset.))]
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
    torch.save(model.state_dict(), 'wytrenowany_model1.pt')
