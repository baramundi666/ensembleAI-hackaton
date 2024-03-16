import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from taskdataset import TaskDataset
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image

from image_to_vector import ImageToVector
    

dataset = torch.load("data/ModelStealingPub.pt")
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, sep=";")
        self.model_outputs = [torch.tensor(parse_vector(vec)) for vec in self.data['model_output']]
        self.ids = self.data['ID']  
        self.imgs = []
        for i in range(len(self.ids)):
            self.imgs.append(dataset.imgs[self.ids[i]].convert("RGB"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_output = self.model_outputs[idx]
        id = self.ids[idx]
        return model_output, id
    

def parse_vector(vector_string):
    vector_string = vector_string.strip('[]')
    elements = vector_string.split(', ')
    elements = [float(element) for element in elements]
    return elements

my_dataset = MyDataset('outputs/output.csv')
labels = my_dataset.model_outputs

dataset = torch.load("./data/ModelStealingPub.pt")

imgs = [dataset.imgs[i] for i in my_dataset.ids]

model = ImageToVector()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a DataLoader for the dataset
class MyDataset1(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = transforms.ToTensor()(self.imgs[idx].convert("RGB"))
        label = self.labels[idx]
        return img, label

# Create a DataLoader for training
batch_size = 32
dataset = MyDataset1(imgs, labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'image_to_vector_model.pt')




# model = ImageToVector()
# model.load_state_dict(torch.load('image_to_vector_model.pt'))
# model.eval()


# # Preprocess the sample image
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# dataset = torch.load("data/ModelStealingPub.pt")
# imgs = dataset.imgs[1:66]

# imgs_tensor = torch.stack([transform(img.convert("RGB")) for img in imgs])  # Add batch dimension

# # Perform inference to obtain the vector representation
# with torch.no_grad():
#     output_vectors = model(imgs_tensor)

# total_dist = 0.0
# for output_vector, label_vector in zip(output_vectors, labels):
#     distance = torch.norm(output_vector - label_vector)
#     total_dist += distance.item()

# print(f"Total distance: {total_dist:.4f}")

