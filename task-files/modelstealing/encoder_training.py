import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from taskdataset import TaskDataset
from autoencoder import Autoencoder


import pandas as pd

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
autoencoder = Autoencoder()


imgs = dataset.imgs
def get_img():
    return None
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

num_epochs = 50

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    for i in range(len(my_dataset.imgs)):
        img = my_dataset.imgs[i]
        img = torchvision.transforms.ToTensor()(img)
        vector = my_dataset.model_outputs[i]
        optimizer.zero_grad()
        output = autoencoder(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(my_dataset.imgs)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


torch.save(autoencoder.state_dict(), 'models/autoencoder_model.pt')
